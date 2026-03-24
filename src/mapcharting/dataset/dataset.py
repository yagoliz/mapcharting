import torch
from torch.utils.data import Dataset


class CSIPairsDataset(Dataset):
    def __init__(self, csi, dissimilarity, n_neighbors: int = 50, neighbor_fraction: float = 0.5):
        self.csi = torch.tensor(csi, requires_grad=False)
        self.dissimilarity = torch.tensor(dissimilarity, requires_grad=False)
        self.count = csi.shape[0]
        self.neighbor_fraction = neighbor_fraction

        # Precompute k-nearest neighbors for each point (by dissimilarity, excluding self)
        topk = torch.topk(self.dissimilarity, k=n_neighbors + 1, largest=False, dim=1)
        self.knn_indices = topk.indices[:, 1:]  # shape (N, n_neighbors), exclude self at col 0

    def __len__(self):
        return self.csi.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if torch.rand(1).item() < self.neighbor_fraction:
            # Sample from k-nearest neighbors to provide good local structure signal
            k = self.knn_indices.shape[1]
            nn_pos = torch.randint(0, k, (1,)).item()
            idx2 = self.knn_indices[idx, nn_pos].item()
        else:
            # Random sample for global structure
            idx2 = idx
            while idx2 == idx:
                idx2 = torch.randint(0, self.count, (1,)).item()

        csi_a = self.csi[idx]
        csi_b = self.csi[idx2]

        # Ground-truth dissimilarity between the two samples
        # Assumes dissimilarity is a square matrix (N x N)
        dissimilarity = self.dissimilarity[idx, idx2]

        return csi_a, csi_b, dissimilarity

