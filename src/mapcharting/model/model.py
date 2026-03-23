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
    def __init__(
        self,
    ):
        super().__init__()

        self.feature = FeatureEngineeringLayer()
        self.fc1 = nn.Sequential(
            nn.Linear(26624, 1024), nn.ReLU(), nn.BatchNorm1d(1024)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64)
        )
        self.encoder = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3,
            self.fc4,
            self.fc5,
        )
        self.output_layer = nn.Linear(64, 2)

    def forward(self, X):
        X = self.feature(X)
        X = X.flatten(start_dim=1)
        X = self.encoder(X)
        return self.output_layer(X)

    def save(self, filename=".cache/model.pt2"):
        torch.save(self.state_dict(), filename)

    def load(self, filename=".cache/model.pt2"):
        self.load_state_dict(torch.load(filename, weights_only=True), strict=False)
