import torch


def siamese_loss(y_true, y_pred, dissimilarity_margin):
	y_true = torch.reshape(y_true, (-1,))
	pos_A, pos_B = (y_pred[:,:2], y_pred[:,2:])
	distances_pred = torch.sqrt(torch.sum(torch.square(pos_A - pos_B), dim=1) + 1e-8)

	return torch.mean(torch.square(distances_pred - y_true) / (y_true + dissimilarity_margin))