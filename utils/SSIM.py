# Ana Harris 27/01/2023
# Reference: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
# Rewritten, instead of 2D images we use 3D images
# SSIM is composed of 3 components: Luminance; contrast; and structure.


import torch  
import torch.nn.functional as F 
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """Create 1-D Gaussian kernel with shape (1, 1, size)
    Args:
      size: the size of the Gaussian kernel
      sigma: sigma of normal distribution
    Returns:
      1D kernel (1, 1,1, size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0).unsqueeze(0)


def _gaussian_filter(inputs: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """Apply 1D Gaussian kernel to inputs images
    Args:
      inputs: a batch of images in shape (C,D,H,W)
      win: 1-D Gaussian kernel
    Returns:
      blurred images
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    channel = inputs.shape[0]
    outputs = inputs
    for i, s in enumerate(inputs.shape[2:]):
        #if s >= win.shape[-1]:
        outputs = F.conv3d(
            outputs.double(),
            weight=win.transpose(2 + i, -1).double(),
            stride=1,
            padding=win.shape[2 + i] // 2,
            groups=channel,
        )
    return outputs


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    max_value: float = 1.0,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
) -> torch.Tensor:
    """Computes structural similarity index metric (SSIM)

    Reference: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py

    Args:
      x: images in the format of (C,H,W,D)
      y: images in the format of (C,H,W,D)
      max_value: the maximum value of the images (usually 1.0 or 255.0)
      size_average: return SSIM average of all images
      win_size: the size of gauss kernel
      win_sigma: sigma of normal distribution
      win: 1-D gauss kernel. if None, a new kernel will be created according to
          win_size and win_sigma
      K1: scalar constant
      K2: scalar constant
    Returns:
      SSIM value(s)
    """
    assert x.shape == y.shape, "input images should have the same dimensions."

    # remove dimensions that has size 1, except the batch and channel dimensions
    for d in range(2, x.ndim):
        x = x.squeeze(dim=d)
        y = y.squeeze(dim=d)


    assert win_size % 2 == 1, f"win_size should be odd, but got {win_size}."

    win = _gaussian_kernel_1d(win_size, win_sigma)
    win = win.repeat([x.shape[1]] + [1] * (len(x.shape) - 1))

    compensation = 1.0
    eps = 1e-8

    C1 = (K1 * max_value) ** 2
    C2 = (K2 * max_value) ** 2
    C3 = C2/2

    win = win.to(x.device, dtype=x.dtype)

    mu1 = _gaussian_filter(x, win)
    mu2 = _gaussian_filter(y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (_gaussian_filter(x * x, win) - mu1_sq)
    sigma1 = torch.sqrt((sigma1_sq + eps))
    sigma2_sq = compensation * (_gaussian_filter(y * y, win) - mu2_sq)
    sigma2 = torch.sqrt((sigma2_sq + eps))
    sigma12 = compensation * (_gaussian_filter(x * y, win) - mu1_mu2)

    structural = (sigma12 + C3)/(sigma1*sigma2 + C3) 
    luminosity = (2*mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1)
    contrast = (2*sigma1*sigma2 + C2)/(sigma1_sq + sigma2_sq + C2)

    return luminosity,contrast,structural


def compute_ssim(img1,img2,mask,map=None):
    luminosity,contrast,structural = ssim(img1,img2)
    luminosity = luminosity.squeeze().numpy()
    luminosity[mask == 0] = 0
    luminosity_score = np.mean(luminosity[luminosity!=0])


    contrast = contrast.squeeze().numpy()
    contrast[mask == 0] = 0
    contrast_score = np.mean(contrast[contrast!=0])


    structural = structural.squeeze().numpy()
    structural[mask == 0] = 0
    structural_score = np.mean(structural[structural!=0])

    if map:
      ssim_map = luminosity * structural * contrast
      return luminosity_score,contrast_score,structural_score,ssim_map,structural
    
    else:
      return luminosity_score,contrast_score,structural_score


def maps(diff,indexes=[150,90,90],method='dssim'):
  figure, axes = plt.subplots(
      nrows=1,
      ncols=4,
      figsize=(15,5),
      gridspec_kw={
          "width_ratios": [1,1,1,0.05],
          "wspace": 0.005,
      },
      squeeze=False,
      dpi=120,
  )

  vmax = 1
  vcenter = vmax/2
  norm = colors.TwoSlopeNorm(vmin=0,vcenter=vcenter, vmax=vmax)
  color = 'jet'

  title = f'{method} map'

  figure.patch.set_facecolor("white")
  
  axes = axes.flatten()
  axes[0].imshow(diff[indexes[0],:,:], cmap = color, norm=norm, origin='lower',interpolation="none")
  axes[0].axis('off')
  axes[1].imshow(diff[:,indexes[1],:].T, cmap = color, norm=norm, origin='lower',interpolation="none")
  axes[1].axis('off')
  axes[2].imshow(diff[:,:,indexes[2]], cmap = color, norm=norm, origin='lower',interpolation="none")
  axes[2].axis('off')
  
  figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(color)), cax=axes[3])
  figure.suptitle(title)
  
  return figure