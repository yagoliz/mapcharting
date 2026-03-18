import tensorflow as tf


def record_parse_function(proto, feature_description):
    record = tf.io.parse_single_example(proto, feature_description)

    # Measured carrier frequency offset between MOBTX and each receive antenna.
    cfo = tf.ensure_shape(tf.io.parse_tensor(record["cfo"], out_type=tf.float32), (32))

    # Channel coefficients for all antennas, over all subcarriers, real and imaginary parts
    csi = tf.ensure_shape(
        tf.io.parse_tensor(record["csi"], out_type=tf.float32), (32, 1024, 2)
    )

    # Time in seconds to closest known tachymeter position. Indicates quality of linear interpolation.
    gt_interp_age_tachy = tf.ensure_shape(record["gt-interp-age-tachy"], ())

    # Position of transmitter determined by a tachymeter pointed at a prism mounted on top of the antenna, in meters (X / Y / Z coordinates)
    pos_tachy = tf.ensure_shape(
        tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3)
    )

    # Signal-to-Noise ratio estimates for all antennas
    snr = tf.ensure_shape(tf.io.parse_tensor(record["snr"], out_type=tf.float32), (32))

    # Timestamp since start of measurement campaign, in seconds
    time = tf.ensure_shape(record["time"], ())

    return cfo, csi, gt_interp_age_tachy, pos_tachy, snr, time
