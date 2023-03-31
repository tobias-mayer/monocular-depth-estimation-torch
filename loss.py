imoprt torch
import torch.nn.functional as F

def _compute_image_gradient(img):
    batch_size, channels, height, width = img.shape

    dy = img[..., 1:, :] - img[..., :-1, :]
    dx = img[..., :, 1:] - img[..., :, :-1]

    shapey = [batch_size, channels, 1, width]
    dy = torch.cat([dy, torch.zeros(shapey, device=img.device, dtype=img.dtype)], dim=2)
    dy = dy.view(img.shape)

    shapex = [batch_size, channels, height, 1]
    dx = torch.cat([dx, torch.zeros(shapex, device=img.device, dtype=img.dtype)], dim=3)
    dx = dx.view(img.shape)

    return dy, dx

def _l_depth(y_pred, y):
    return F.l1_loss(y_pred, y)

def _l_grad(y_pred, y):
    dy_true, dx_true = _compute_image_gradient(y)
    dy_pred, dx_pred = image_gradient(y_pred)

    return torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), axis=-1)

def _l_ssim(y_pred, y):
    pass

def depth_loss(y_pred, y, lam=0.1):
    return lam * _l_depth(y_pred, y) + _l_grad(y_pred, y) + _l_ssim(y_pred, y)

