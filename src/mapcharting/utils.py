import numpy as np
import matplotlib.pyplot as plt


def plot_colorized(
    positions: np.ndarray,
    groundtruth_positions: np.ndarray,
    title: str | None = None,
    show: bool = True,
    alpha: float = 1.0,
):
    # Generate RGB colors for datapoints
    center_point = np.zeros(2, dtype=np.float32)
    center_point[0] = 0.5 * (
        np.min(groundtruth_positions[:, 0], axis=0)
        + np.max(groundtruth_positions[:, 0], axis=0)
    )
    center_point[1] = 0.5 * (
        np.min(groundtruth_positions[:, 1], axis=0)
        + np.max(groundtruth_positions[:, 1], axis=0)
    )

    def normalize_data(in_data):
        return (in_data - np.min(in_data)) / (np.max(in_data) - np.min(in_data))

    rgb_values = np.zeros((groundtruth_positions.shape[0], 3))
    rgb_values[:, 0] = 1 - 0.9 * normalize_data(groundtruth_positions[:, 0])
    rgb_values[:, 1] = 0.8 * normalize_data(
        np.square(np.linalg.norm(groundtruth_positions[:, 0:2] - center_point, axis=1))
    )
    rgb_values[:, 2] = 0.9 * normalize_data(groundtruth_positions[:, 1])

    # Plot datapoints
    plt.figure(figsize=(6, 6))
    if title is not None:
        plt.title(title, fontsize=16)
    plt.scatter(
        positions[:, 0], positions[:, 1], c=rgb_values, alpha=alpha, s=10, linewidths=0
    )
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    if show:
        plt.show()


def pad(x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((x.shape[0], 1))])


def transform(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    return unpad(np.dot(pad(x), A))


def unpad(x: np.ndarray) -> np.ndarray:
    return x[:, :-1]


def affine_transform_channel_chart(
    groundtruth_pos: np.ndarray, channel_chart_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    A, res, rank, s = np.linalg.lstsq(
        pad(channel_chart_pos), pad(groundtruth_pos), rcond=None
    )
    return transform(channel_chart_pos, A), A


def affine_transform(pos: np.ndarray, A: np.ndarray) -> np.ndarray:
    return transform(pos, A)
