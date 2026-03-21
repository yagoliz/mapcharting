import torch
from torch.utils.data import Dataset


class CSIPairsDataset(Dataset):
    def __init__(self, csi, dissimilarity):
        self.csi = torch.tensor(csi, requires_grad=False)
        self.dissimilarity = torch.tensor(dissimilarity, requires_grad=False)
        self.count = csi.shape[0]

    def __len__(self):
        return self.csi.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Pick a second index different from idx
        idx2 = idx
        while idx2 == idx:
            idx2 = torch.randint(0, self.count, (1,)).item()

        csi_a = self.csi[idx]
        csi_b = self.csi[idx2]

        # Ground-truth dissimilarity between the two samples
        # Assumes dissimilarity is a square matrix (N x N)
        dissimilarity = self.dissimilarity[idx, idx2]

        return csi_a, csi_b, dissimilarity

