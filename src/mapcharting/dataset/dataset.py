import torch
from torch.utils.data import Dataset


class CSIPairsDataset(Dataset):
    def __init__(self, csi, dissimilarity):
        self.csi = torch.tensor(csi, requires_grad=False)
        # Replace inf geodesic distances (disconnected nodes) with the max finite value
        # to prevent NaN gradients during training
        d = torch.tensor(dissimilarity, requires_grad=False)
        finite_max = d[torch.isfinite(d)].max()
        self.dissimilarity = torch.where(torch.isfinite(d), d, finite_max)
        self.count = csi.shape[0]

    def __len__(self):
        return self.count

    def __getitem__(self, idx: int) -> tuple:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        idx2 = idx
        while idx2 == idx:
            idx2 = int(torch.randint(0, self.count, (1,)).item())

        csi_a = self.csi[idx]
        csi_b = self.csi[idx2]
        dissimilarity = self.dissimilarity[idx, idx2]

        return csi_a, csi_b, dissimilarity