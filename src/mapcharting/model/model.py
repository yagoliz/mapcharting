import torch
import torch.nn as nn


class FeatureEngineeringLayer(nn.Module):
    def __init__(self):
        super(FeatureEngineeringLayer, self).__init__()

    def forward(self, csi):
        # Compute sample correlations for any combination of two antennas in the whole system
        # for the same datapoint and time tap.
        sample_autocorrelations = torch.einsum(
            "damt,dbnt->dtabmn", csi, torch.conj(csi)
        )
        return torch.stack(
            [torch.real(sample_autocorrelations), torch.imag(sample_autocorrelations)],
            dim=-1,
        )


class ChannelCharter(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(832, 256), nn.ReLU(), nn.BatchNorm1d(256)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.BatchNorm1d(64)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 16), nn.ReLU(), nn.BatchNorm1d(16)
        )
        self.encoder = nn.Sequential(
            self.fc1, self.fc2, self.fc3,
        )
        self.output_layer = nn.Linear(16, 2)

    def forward(self, X):
        X = torch.view_as_real(X)
        X = X.flatten(start_dim=1)
        X = self.encoder(X)
        return self.output_layer(X)

    def save(self, filename=".cache/model.pt2"):
        torch.save(self.state_dict(), filename)

    def load(self, filename=".cache/model.pt2"):
        self.load_state_dict(torch.load(filename, weights_only=True), strict=False)
