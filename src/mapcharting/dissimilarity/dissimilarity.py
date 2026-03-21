import multiprocessing as mp

import numpy as np
from scipy.sparse import spmatrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
import torch
from tqdm import tqdm


def adp_dissimilarity(
    csi_np: np.ndarray, device: str = "cpu", chunk_size: int = 64
) -> torch.Tensor:
    N = csi_np.shape[0]
    output = torch.zeros(N, N)  # accumulate on CPU

    for i in range(N):
        # Move only one reference vector to GPU
        h = torch.from_numpy(csi_np[i]).to(device)  # (B, M, T)
        power_h = torch.einsum("bmt,bmt->bt", h, torch.conj(h))  # (B, T)

        # Process the remaining rows in sub-batches
        for j in range(i, N, chunk_size):
            w = torch.from_numpy(csi_np[j : j + chunk_size]).to(device)  # (C, B, M, T)
            power_w = torch.einsum("lbmt,lbmt->lbt", w, torch.conj(w))  # (C, B, T)

            dotproducts = torch.abs(
                torch.square(torch.einsum("bmt,lbmt->lbt", torch.conj(h), w))
            )
            d = torch.sum(1 - dotproducts / torch.real(power_h * power_w), dim=(1, 2))
            d = torch.maximum(d, torch.zeros_like(d)).cpu()

            output[i, j : j + chunk_size] = d

    return output + output.T


def timestamp_dissimilarity(timestamps: np.ndarray) -> np.ndarray:
    return np.abs(np.subtract.outer(timestamps, timestamps))


def get_weighted_graph(matrix: np.ndarray, n_neighbors: int = 20) -> spmatrix:
    nbrs_alg = NearestNeighbors(
        n_neighbors=n_neighbors, metric="precomputed", n_jobs=-1
    )
    nbrs = nbrs_alg.fit(matrix)
    nbg = kneighbors_graph(nbrs, n_neighbors, metric="precomputed", mode="distance")
    return nbg


def _shortest_path_worker(graph, todo_queue, output_queue):
    while True:
        index = todo_queue.get()
        if index == -1:
            output_queue.put((-1, None))
            break

        d = dijkstra(graph, directed=False, indices=index)
        output_queue.put((index, d))


def geodesic_dissimilarity(
    dissimilarity_matrix: np.ndarray, n_neighbors: int = 20
) -> np.ndarray:
    # Step 1: We need the weighted connectivity graph
    graph = get_weighted_graph(dissimilarity_matrix, n_neighbors)

    # Step 2: Dijkstra computation
    output = np.zeros(graph.shape, dtype=np.float32)

    with tqdm(total=graph.shape[0]) as pbar:
        todo_queue = mp.Queue()
        output_queue = mp.Queue()

        for i in range(graph.shape[0]):
            todo_queue.put(i)

        for i in range(mp.cpu_count()):
            todo_queue.put(-1)
            p = mp.Process(target=_shortest_path_worker, args=(graph, todo_queue, output_queue))
            p.start()

        finished_processes = 0
        while finished_processes != mp.cpu_count():
            i, d = output_queue.get()

            if i == -1:
                finished_processes += 1
            else:
                output[i, :] = d
                pbar.update(1)

    return output
