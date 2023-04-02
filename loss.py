import torch
import torch.nn.functional as F

def _compute_image_gradient(img):
    '''
    Compute the gradient of an image.
    Arguments:
        img: a tensor representing the image (batch_size, channels, height, width).
    Returns:
        A tuple of two tensors representing the vertical and horizontal gradients of the image.
    '''
    # Extract the shape of the image tensor
    batch_size, channels, height, width = img.shape

    # Compute the vertical and horizontal gradients of the image using finite differences
    dy = img[..., 1:, :] - img[..., :-1, :]
    dx = img[..., :, 1:] - img[..., :, :-1]

    # Add zero padding to the vertical and horizontal gradients to match the original image shape
    shapey = [batch_size, channels, 1, width]
    dy = torch.cat([dy, torch.zeros(shapey, device=img.device, dtype=img.dtype)], dim=2)
    dy = dy.view(img.shape)

    shapex = [batch_size, channels, height, 1]
    dx = torch.cat([dx, torch.zeros(shapex, device=img.device, dtype=img.dtype)], dim=3)
    dx = dx.view(img.shape)

    # Return the vertical and horizontal gradients as a tuple
    return dy, dx

def _l_depth(y_pred, y):
    '''
    Compute the L1 loss between two images.
    Arguments:
        y_pred: a tensor representing the predicted image (batch_size, channels, height, width).
        y: a tensor representing the ground truth image (batch_size, channels, height, width).
    Returns:
        The L1 loss between the two images (a scalar).
    '''
    # Compute the L2 loss between the predicted and ground truth images
    return F.l1_loss(y_pred, y)

def _l_grad(y_pred, y):
    '''
    Compute the L1 loss between the gradients of two images.
    Arguments:
        y_pred: a tensor representing the predicted image (batch_size, channels, height, width).
        y: a tensor representing the ground truth image (batch_size, channels, height, width).
    Returns:
        The L1 loss between the gradients of the two images (a scalar).
    '''
    # Compute the gradients of the ground truth image batch
    dy_true, dx_true = _compute_image_gradient(y)

    # Compute the gradients of the predicted image batch
    dy_pred, dx_pred = _compute_image_gradient(y_pred)

    # Compute the L1 loss between the gradients
    return torch.mean(torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true))

def _l_ssim(img1, img2, window_size=11, sigma=1.5, L=1):
    '''
    Compute the Structural Similarity Index Measure (SSIM) between two images.
    Arguments:
        img1: a tensor representing the first image (batch_size, channels, height, width).
        img2: a tensor representing the second image (batch_size, channels, height, width).
        window_size: the size of the sliding window used for comparison (default: 11).
        sigma: the standard deviation of the Gaussian filter applied to the window (default: 1.5).
        L: the dynamic range of the pixel values (default: 1).
    Returns:
        The SSIM value between the two images (a scalar).
    '''
    
    # Create a sliding window for the SSIM computation
    window = torch.ones((1, 1, window_size, window_size)).to(img1.device)
    
    # Compute the means of the two images
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.shape[1])
    
    # Compute the variances and covariances of the two images
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.shape[1]) - mu1 * mu1
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.shape[1]) - mu2 * mu2
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1 * mu2
    
    # Compute the SSIM values
    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2
    ssim = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Compute the mean SSIM value across all channels and pixels
    return torch.mean(ssim)

def depth_loss(y_pred, y, lam=0.1):
    '''
    Compute the depth loss between two images which is defined as the sum of three loss functions.
    Arguments:
        y_pred: a tensor representing the predicted image (batch_size, channels, height, width).
        y: a tensor representing the ground truth image (batch_size, channels, height, width).
        lam: a float representing the weight of the L1 loss term (default: 0.1).
    Returns:
        The depth loss between the two images (a scalar).
    '''
    # Compute the L1 loss between the predicted and ground truth images, scaled by lam
    l1_loss = lam * _l_depth(y_pred, y)
    print(f'l1: {l1_loss}')

    # Compute the L1 loss between the gradients of the predicted and ground truth images
    grad_loss = _l_grad(y_pred, y)
    print(f'grad: {grad_loss}')

    # Compute the structural similarity index (SSIM) between the predicted and ground truth images
    ssim_loss = _l_ssim(y_pred, y)
    print(f'ssim: {ssim_loss}')

    # Compute the weighted sum of the three loss terms
    return l1_loss + grad_loss + ssim_loss

