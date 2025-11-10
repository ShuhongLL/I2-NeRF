import os
import enum
import math
import logging
import collections
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from math import exp
from torch.autograd import Variable
from PIL import ExifTags
from PIL import Image
from internal import vis
from matplotlib import cm


class Timing:
    """
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print(self.name, "elapsed", self.start.elapsed_time(self.end), "ms")


def handle_exception(exc_type, exc_value, exc_traceback):
    logging.error("Error!", exc_info=(exc_type, exc_value, exc_traceback))


def nan_sum(x):
    return (torch.isnan(x) | torch.isinf(x)).sum()


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class DataSplit(enum.Enum):
    """Dataset split."""
    TRAIN = 'train'
    TEST = 'test'


class BatchingMethod(enum.Enum):
    """Draw rays randomly from a single image or all images, in each batch."""
    ALL_IMAGES = 'all_images'
    SINGLE_IMAGE = 'single_image'


def open_file(pth, mode='r'):
    return open(pth, mode=mode)


def file_exists(pth):
    return os.path.exists(pth)


def listdir(pth):
    return os.listdir(pth)


def isdir(pth):
    return os.path.isdir(pth)


def makedirs(pth):
    os.makedirs(pth, exist_ok=True)


def load_img(pth):
    """Load an image and cast to float32."""
    image = np.array(Image.open(pth), dtype=np.float32)
    return image


def load_exif(pth):
    """Load EXIF data for an image."""
    with open_file(pth, 'rb') as f:
        image_pil = Image.open(f)
        exif_pil = image_pil._getexif()  # pylint: disable=protected-access
        if exif_pil is not None:
            exif = {
                ExifTags.TAGS[k]: v for k, v in exif_pil.items() if k in ExifTags.TAGS
            }
        else:
            exif = {}
    return exif


def save_img_u8(img, pth):
    """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
        pth, 'PNG')


def save_img_f32(depthmap, pth, p=0.5):
    """Save an image (probably a depthmap) to disk as a float32 TIFF."""
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(pth, 'TIFF')
    

def find_closest_factors(batch_size):
    # Start from the square root of the batch_size and go downwards
    for i in range(int(math.sqrt(batch_size)), 0, -1):
        if batch_size % i == 0:
            return i, batch_size // i


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(rendered, target, window, window_size, channel, desired_mean, contrast_factor, size_average=True, stride=None):
    """
    Compute SSIM between rendered and target images, but compensate the target's
    local luminance and contrast (variance) on the fly. The idea is to linearly transform 
    the target's local mean and variance to match the desired normal-light conditions.
    
    Parameters:
      rendered       : The output image from your network (assumed to be "normal-light")
      target         : The low-light input image
      window         : The convolution window (e.g., a Gaussian kernel) for local statistics
      window_size    : The size (assumed square) of the window
      channel        : The number of channels in the images
      desired_mean   : The desired global mean for a normal-light image (e.g., 0.5 for [0,1] images)
      contrast_factor: A scaling factor to boost the contrast of the target. 
                       (The target’s variance is multiplied by contrast_factor^2.)
      size_average   : Whether to average the resulting SSIM map
      stride         : Stride to use in convolution (if any)
      
    Returns:
      The (compensated) SSIM value.
    """
    # Compute local means with convolution
    mu1 = F.conv2d(rendered, window, padding=window_size//2, groups=channel, stride=stride)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel, stride=stride)
    

    global_target_mean = target.mean()
    # Adjust (compensate) target's local mean:
    # This simulates converting the low-light target to the desired normal-light levels.
    mu2_comp = (mu2 - global_target_mean) * contrast_factor + desired_mean

    mu1_sq = mu1.pow(2)
    mu2_comp_sq = mu2_comp.pow(2)
    
    sigma1_sq = F.conv2d(rendered * rendered, window, padding=window_size//2, groups=channel, stride=stride) - mu1_sq
    # Compute local variance for target image using its original local mean.
    # Note: sigma2_sq = E[target^2] - (mu2)^2.
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel, stride=stride) - F.conv2d(target, window, padding=window_size//2, groups=channel, stride=stride).pow(2)
    # Adjust the target's local variance: variance scales by (contrast_factor)^2.
    sigma2_sq_comp = sigma2_sq * (contrast_factor ** 2)
    
    # Compute local covariance between rendered and target.
    # Standard formulation: sigma12 = E[rendered * target] - mu1 * mu2.
    sigma12 = F.conv2d(rendered * target, window, padding=window_size//2, groups=channel, stride=stride) - mu1 * mu2
    # Adjust the covariance: for a linear transform, covariance scales linearly with contrast_factor.
    sigma12_comp = sigma12 * contrast_factor

    # Constants for numerical stability (as in the original SSIM formulation)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Compute the luminance term using the compensated local mean
    luminance = (2 * mu1 * mu2_comp + C1) / (mu1_sq + mu2_comp_sq + C1)
    
    # Compute the contrast-structure term using the compensated variance and covariance
    contrast_structure = (2 * sigma12_comp + C2) / (sigma1_sq + sigma2_sq_comp + C2)
    
    ssim_map = luminance * contrast_structure
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, target_mean, contrast_factor, window_size = 3, size_average = True, stride=3):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.stride = stride
        self.window = create_window(window_size, self.channel)
        self.target_mean = target_mean
        self.contrast_factor = contrast_factor

    def forward(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.target_mean,
                     self.contrast_factor, self.size_average, stride=self.stride)


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class S3IM(torch.nn.Module):
    r"""Implements Stochastic Structural SIMilarity(S3IM) algorithm.
    It is proposed in the ICCV2023 paper  
    `S3IM: Stochastic Structural SIMilarity and Its Unreasonable Effectiveness for Neural Fields`.

    Arguments:
        kernel_size (int): kernel size in ssim's convolution(default: 4)
        stride (int): stride in ssim's convolution(default: 4)
        repeat_time (int): repeat time in re-shuffle virtual patch(default: 10)
        patch_height (height): height of virtual patch(default: 64)
        patch_width (height): width of virtual patch(default: 64)
    """
    def __init__(self, target_mean, contrast_factor, kernel_size=4, stride=4, repeat_time=10):
        super(S3IM, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.repeat_time = repeat_time
        self.ssim_loss = SSIM(target_mean, contrast_factor, window_size=self.kernel_size,
                              stride=self.stride)
        
    def forward(self, src_vec, tar_vec):
        src_vec = src_vec.squeeze() # [batch_size, 1, 1, 3] -> [batch_size, 3]
        tar_vec = tar_vec.squeeze()
        cur_batch_size = tar_vec.shape[0]
        patch_height, patch_width = find_closest_factors(cur_batch_size)
        index_list = []
        for i in range(self.repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, patch_height, patch_width * self.repeat_time)
        src_patch = src_all.permute(1, 0).reshape(1, 3, patch_height, patch_width * self.repeat_time)
        loss = (1 - self.ssim_loss(src_patch, tar_patch))
        return loss


class Structure_Loss(torch.nn.Module):
    def __init__(self, contrast):
        super(Structure_Loss, self).__init__()
        self.kernel_left = torch.FloatTensor([-1, 1, 0]).unsqueeze(0).unsqueeze(0)
        self.kernel_right = torch.FloatTensor([0, 1, -1]).unsqueeze(0).unsqueeze(0)
        self.weight_left = torch.nn.Parameter(data=self.kernel_left, requires_grad=False)
        self.weight_right = torch.nn.Parameter(data=self.kernel_right, requires_grad=False)
        self.pool = torch.nn.AvgPool1d(4)
        self.contrast = contrast
    
    def forward(self, rendered, target):
        rendered_mean = torch.mean(rendered, dim=-1).permute(2, 1, 0) # [batch_size, 1, 1] -> [1, 1, batch_size]
        target_mean = torch.mean(target, dim=-1).permute(2, 1, 0) # [batch_size, 1, 1] -> [1, 1, batch_size]

        target_pool, rendered_pool = self.pool(target_mean), self.pool(rendered_mean)

        target_left = F.conv1d(target_pool, self.weight_left.to(target_pool.device), padding=1)
        target_right = F.conv1d(target_pool, self.weight_right.to(target_pool.device), padding=1)

        rendered_left = F.conv1d(rendered_pool, self.weight_left.to(rendered_pool.device), padding=1)
        rendered_right = F.conv1d(rendered_pool, self.weight_right.to(rendered_pool.device), padding=1)

        D_left = torch.pow(self.contrast * target_left - rendered_left, 2)
        D_right = torch.pow(self.contrast * target_right - rendered_right, 2)
        
        return torch.mean(D_left + D_right)
    

def load_bcp(image, bcp_kernel_size=7, use_atmospheric_light=False):
    bright = cal_bright_channel(image, bcp_kernel_size)
    if use_atmospheric_light:
        A = estimate_atmospheric_light(image, bright)
    else:
        A = np.zeros(3) + 1e-3
    trans = calculate_transmission(image, A, bcp_kernel_size)
    trans_min = calcualte_min_transmission(image, A)
    trans_min = np.clip(trans_min, 0.1, 1)
    refined_trans = guided_filter(image[:, :, 0], trans, r=bcp_kernel_size, eps=1e-3)
    refined_trans = np.clip(refined_trans, trans_min, 1)[..., None] # [h, w, 1]
    J = (image - A[None, None,:]) / refined_trans + A[None, None,:] # [h, w, 3]
    return J, refined_trans


def cal_bright_channel(I, patch_size=5):
    J = np.max(I, axis=2)
    return cv2.dilate(J, np.ones((patch_size, patch_size)))


def estimate_atmospheric_light(I, bright_channel):
    # Using the darkest 0.1% pixels in the bright channel to estimate A
    darkest_percentile = np.percentile(bright_channel, 0.1)
    indices = np.where(bright_channel <= darkest_percentile)
    flat_I = I.reshape(-1, I.shape[2])
    darkest_pixels = flat_I[indices[0], :]
    A = np.mean(darkest_pixels, axis=0)
    return A


def calculate_transmission(img, A, kernel_size=5, omega=0.9):
    normalized = (1 - img) / (1 - A[None, None, :])
    max_color = np.max(normalized, axis=-1)
    # Apply max filter (dilation) to find the maximum over each patch
    max_patch = cv2.dilate(max_color, np.ones((kernel_size, kernel_size), dtype=np.float32))
    t = 1 - omega * max_patch
    return t


def calcualte_min_transmission(img, A):
    A = A[None, None, :]
    trans = (img - A) / (1 - A)
    return np.max(trans, axis=-1)


def guided_filter(I, p, r, eps):
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    ones = np.ones(I.shape[:2], dtype=np.float32)
    N = cv2.boxFilter(ones, -1, (r, r))
    mean_I = cv2.boxFilter(I, -1, (r, r)) / N
    mean_p = cv2.boxFilter(p, -1, (r, r)) / N
    mean_Ip = cv2.boxFilter(I * p, -1, (r, r)) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, -1, (r, r)) / N
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (r, r)) / N
    mean_b = cv2.boxFilter(b, -1, (r, r)) / N
    q = mean_a * I + mean_b
    return q
