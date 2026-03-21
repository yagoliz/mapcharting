import json
import numpy as np
import tensorflow as tf

# Antenna definitions
ASSIGNMENTS = [
    [0, 13, 31, 29, 3, 7, 1, 12],
    [30, 26, 21, 25, 24, 8, 22, 15],
    [28, 5, 10, 14, 6, 2, 16, 18],
    [19, 4, 23, 17, 20, 11, 9, 27],
]

# Antenna counts
ANTENNACOUNT = np.sum([len(antennaArray) for antennaArray in ASSIGNMENTS])

def load_calibrate_timedomain(path, offset_path):
	offsets = None
	with open(offset_path, "r") as offsetfile:
		offsets = json.load(offsetfile)
	
	def record_parse_function(proto):
		record = tf.io.parse_single_example(
			proto,
			{
				"csi": tf.io.FixedLenFeature([], tf.string, default_value=""),
				"pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value=""),
				"time": tf.io.FixedLenFeature([], tf.float32, default_value=0),
			},
		)

		csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type=tf.float32), (ANTENNACOUNT, 1024, 2))
		csi = tf.complex(csi[:, :, 0], csi[:, :, 1])
		csi = tf.signal.fftshift(csi, axes=1)

		position = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3))
		time = tf.ensure_shape(record["time"], ())

		return csi, position[:2], time

	def apply_calibration(csi, pos, time):
		sto_offset = tf.tensordot(tf.constant(offsets["sto"]), 2 * np.pi * tf.range(tf.shape(csi)[1], dtype = np.float32) / tf.cast(tf.shape(csi)[1], np.float32), axes = 0)
		cpo_offset = tf.tensordot(tf.constant(offsets["cpo"]), tf.ones(tf.shape(csi)[1], dtype = np.float32), axes = 0)
		csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))

		return csi, pos, time

	def csi_time_domain(csi, pos, time):
		csi = tf.signal.fftshift(tf.signal.ifft(tf.signal.fftshift(csi, axes=1)),axes=1)

		return csi, pos, time

	def cut_out_taps(tap_start, tap_stop):
		def cut_out_taps_func(csi, pos, time):
			return csi[:,tap_start:tap_stop], pos, time

		return cut_out_taps_func

	def order_by_antenna_assignments(csi, pos, time):
		csi = tf.stack([tf.gather(csi, antenna_inidces) for antenna_inidces in ASSIGNMENTS])
		return csi, pos, time
	
	dataset = tf.data.TFRecordDataset(path)
	
	dataset = dataset.map(record_parse_function, num_parallel_calls = tf.data.AUTOTUNE)
	dataset = dataset.map(apply_calibration, num_parallel_calls = tf.data.AUTOTUNE)
	dataset = dataset.map(csi_time_domain, num_parallel_calls = tf.data.AUTOTUNE)
	dataset = dataset.map(cut_out_taps(507, 520), num_parallel_calls = tf.data.AUTOTUNE)
	dataset = dataset.map(order_by_antenna_assignments, num_parallel_calls = tf.data.AUTOTUNE)

	return dataset