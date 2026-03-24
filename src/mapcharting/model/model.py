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

        # Feature engineering produces [batch, t, a, b, m, n, 2].
        # We reshape to [batch, t, antenna_features] then permute to
        # [batch, antenna_features, t] for Conv1d.
        # Conv reduces the large antenna-correlation channel dim (2048),
        # AdaptiveAvgPool1d aggregates across the t (tap) dimension.
        self.conv = nn.Sequential(
            nn.Conv1d(2048, 512, kernel_size=3, padding=1),  # [batch, 512, t]
            nn.ReLU(),
            nn.Conv1d(512, 128, kernel_size=3, padding=1),   # [batch, 128, t]
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                          # [batch, 128, 1]
        )

        # After flatten: [batch, 128]
        self.encoder = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
        )
        self.output_layer = nn.Linear(64, 2)

    def forward(self, X):
        X = self.feature(X)                        # [batch, t, a, b, m, n, 2]
        X = X.reshape(X.shape[0], X.shape[1], -1)  # [batch, t, antenna_features]
        X = X.permute(0, 2, 1)                     # [batch, antenna_features, t]
        X = self.conv(X)
        X = X.flatten(start_dim=1)                 # [batch, 128]
        X = self.encoder(X)
        return self.output_layer(X)

    def save(self, filename=".cache/model.pt2"):
        torch.save(self.state_dict(), filename)

    def load(self, filename=".cache/model.pt2"):
        self.load_state_dict(torch.load(filename, weights_only=True), strict=False)
