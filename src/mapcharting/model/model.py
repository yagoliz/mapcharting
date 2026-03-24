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

        self.feature = FeatureEngineeringLayer()

        # Conv1d over the frequency/tap dimension (t=416).
        # Feature engineering produces [batch, t, a, b, m, n, 2]; we treat
        # the antenna-correlation features (a*b*m*n*2 = 128) as channels.
        # Input to conv: [batch, 128, 416]
        self.conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3),  # → [batch, 128, 208]
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=7, stride=2, padding=3),   # → [batch,  64, 104]
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=7, stride=4, padding=3),    # → [batch,  32,  26]
            nn.ReLU(),
        )

        # After conv: 32 * 26 = 832
        self.encoder = nn.Sequential(
            nn.Linear(832, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 64),  nn.ReLU(), nn.BatchNorm1d(64),
        )
        self.output_layer = nn.Linear(64, 2)

    def forward(self, X):
        X = self.feature(X)                   # [batch, t, a, b, m, n, 2]
        X = X.reshape(X.shape[0], X.shape[1], -1)  # [batch, t, 128]
        X = X.permute(0, 2, 1)               # [batch, 128, t]  — channels first for Conv1d
        X = self.conv(X)
        X = X.flatten(start_dim=1)           # [batch, 832]
        X = self.encoder(X)
        return self.output_layer(X)

    def save(self, filename=".cache/model.pt2"):
        torch.save(self.state_dict(), filename)

    def load(self, filename=".cache/model.pt2"):
        self.load_state_dict(torch.load(filename, weights_only=True), strict=False)
