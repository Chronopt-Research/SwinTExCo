import sys
import time
import numpy as np
from PIL import Image
from skimage import color
from skimage.transform import resize
import src.data.functional as F
import torch
from torch import nn
import torch.nn.functional as F_torch
import torchvision.transforms.functional as F_torchvision
from numba import cuda, jit
import math
import torchvision.utils as vutils
from torch.autograd import Variable
import cv2

rgb_from_xyz = np.array(
    [
        [3.24048134, -0.96925495, 0.05564664],
        [-1.53715152, 1.87599, -0.20404134],
        [-0.49853633, 0.04155593, 1.05731107],
    ]
)
l_norm, ab_norm = 1.0, 1.0
l_mean, ab_mean = 50.0, 0


import numpy as np
from PIL import Image
from skimage.transform import resize

import numpy as np
from PIL import Image
from skimage.transform import resize

class SquaredPadding:
    def __init__(self, target_size=384, fill_value=0):
        self.target_size = target_size
        self.fill_value = fill_value

    def __call__(self, img, return_pil=True, return_paddings=False, dtype=np.uint8):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        ndim = len(img.shape)
        H, W = img.shape[:2]
        if H > W:
            H_new, W_new = self.target_size, int(W/H*self.target_size)
            # Resize image
            img = resize(img, (H_new, W_new), preserve_range=True).astype(dtype)

            # Padding image
            padded_size = H_new - W_new
            if ndim == 3:
              paddings = [(0, 0), (padded_size // 2, (padded_size // 2) + (padded_size % 2)), (0,0)]
            elif ndim == 2:
              paddings = [(0, 0), (padded_size // 2, (padded_size // 2) + (padded_size % 2))]
            padded_img = np.pad(img, paddings, mode='constant', constant_values=self.fill_value)
        else:
            H_new, W_new = int(H/W*self.target_size), self.target_size
            # Resize image
            img = resize(img, (H_new, W_new), preserve_range=True).astype(dtype)

            # Padding image
            padded_size = W_new - H_new
            if ndim == 3:
              paddings = [(padded_size // 2, (padded_size // 2) + (padded_size % 2)), (0, 0), (0,0)]
            elif ndim == 2:
              paddings = [(padded_size // 2, (padded_size // 2) + (padded_size % 2)), (0, 0)]
            padded_img = np.pad(img, paddings, mode='constant', constant_values=self.fill_value)

        if return_pil:
            padded_img = Image.fromarray(padded_img)

        if return_paddings:
            return padded_img, paddings

        return padded_img
    
class UnpaddingSquare():
    def __call__(self, img, paddings):
        if not isinstance(img, np.ndarray):
          img = np.array(img)
        
        H, W = img.shape[0], img.shape[1]
        (pad_top, pad_bottom), (pad_left, pad_right), _ = paddings
        W_ori = W - pad_left - pad_right
        H_ori = H - pad_top - pad_bottom
        
        return img[pad_top:pad_top+H_ori, pad_left:pad_left+W_ori, :]

class UnpaddingSquare_Tensor():
    def __call__(self, img, paddings):
        H, W = img.shape[1], img.shape[2]
        (pad_top, pad_bottom), (pad_left, pad_right), _ = paddings
        W_ori = W - pad_left - pad_right
        H_ori = H - pad_top - pad_bottom
        
        return img[:, pad_top:pad_top+H_ori, pad_left:pad_left+W_ori]
    
class ResizeFlow(object):
    def __init__(self, target_size=(224,224)):
        self.target_size = target_size
        pass
    
    def __call__(self, flow):
        return F_torch.interpolate(flow.unsqueeze(0), self.target_size, mode='bilinear', align_corners=True).squeeze(0)

class SquaredPaddingFlow(object):
    def __init__(self, fill_value=0):
        self.fill_value = fill_value
        
    def __call__(self, flow):
        H, W = flow.size(1), flow.size(2)
    
        if H > W:
            # Padding flow
            padded_size = H - W
            paddings = (padded_size // 2, (padded_size // 2) + (padded_size % 2), 0, 0)
            padded_img = F_torch.pad(flow, paddings, value=self.fill_value)
        else:
            # Padding flow
            padded_size = W - H
            paddings = (0, 0, padded_size // 2, (padded_size // 2) + (padded_size % 2))
            padded_img = F_torch.pad(flow, paddings, value=self.fill_value)

        return padded_img
    

def gray2rgb_batch(l):
    # gray image tensor to rgb image tensor
    l_uncenter = uncenter_l(l)
    l_uncenter = l_uncenter / (2 * l_mean)
    return torch.cat((l_uncenter, l_uncenter, l_uncenter), dim=1)

def batch_lab2rgb_transpose_mc(img_l_mc, img_ab_mc, nrow=8):
    if isinstance(img_l_mc, Variable):
        img_l_mc = img_l_mc.data.cpu()
    if isinstance(img_ab_mc, Variable):
        img_ab_mc = img_ab_mc.data.cpu()

    if img_l_mc.is_cuda:
        img_l_mc = img_l_mc.cpu()
    if img_ab_mc.is_cuda:
        img_ab_mc = img_ab_mc.cpu()

    assert img_l_mc.dim() == 4 and img_ab_mc.dim() == 4, "only for batch input"

    img_l = img_l_mc * l_norm + l_mean
    img_ab = img_ab_mc * ab_norm + ab_mean
    pred_lab = torch.cat((img_l, img_ab), dim=1)
    grid_lab = vutils.make_grid(pred_lab, nrow=nrow).numpy().astype("float64")
    return (np.clip(color.lab2rgb(grid_lab.transpose((1, 2, 0))), 0, 1) * 255).astype("uint8")


def vgg_preprocess(tensor):
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1)
    return tensor_bgr_ml * 255


def tensor_lab2rgb(input):
    """
    n * 3* h *w
    """
    input_trans = input.transpose(1, 2).transpose(2, 3)  # n * h * w * 3
    L, a, b = (
        input_trans[:, :, :, 0:1],
        input_trans[:, :, :, 1:2],
        input_trans[:, :, :, 2:],
    )
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    neg_mask = z.data < 0
    z[neg_mask] = 0
    xyz = torch.cat((x, y, z), dim=3)

    mask = xyz.data > 0.2068966
    mask_xyz = xyz.clone()
    mask_xyz[mask] = torch.pow(xyz[mask], 3.0)
    mask_xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787
    mask_xyz[:, :, :, 0] = mask_xyz[:, :, :, 0] * 0.95047
    mask_xyz[:, :, :, 2] = mask_xyz[:, :, :, 2] * 1.08883

    rgb_trans = torch.mm(mask_xyz.view(-1, 3), torch.from_numpy(rgb_from_xyz).type_as(xyz)).view(
        input.size(0), input.size(2), input.size(3), 3
    )
    rgb = rgb_trans.transpose(2, 3).transpose(1, 2)

    mask = rgb > 0.0031308
    mask_rgb = rgb.clone()
    mask_rgb[mask] = 1.055 * torch.pow(rgb[mask], 1 / 2.4) - 0.055
    mask_rgb[~mask] = rgb[~mask] * 12.92

    neg_mask = mask_rgb.data < 0
    large_mask = mask_rgb.data > 1
    mask_rgb[neg_mask] = 0
    mask_rgb[large_mask] = 1
    return mask_rgb


###### loss functions ######
def feature_normalize(feature_in):
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm


# denormalization for l
def uncenter_l(l):
    return l * l_norm + l_mean


def get_grid(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))

    return torch.cat([torchHorizontal, torchVertical], 1)


class WarpingLayer(nn.Module):
    def __init__(self, device):
        super(WarpingLayer, self).__init__()
        self.device = device

    def forward(self, x, flow):
        """
        It takes the input image and the flow and warps the input image according to the flow

        Args:
          x: the input image
          flow: the flow tensor, which is a 4D tensor of shape (batch_size, 2, height, width)

        Returns:
          The warped image
        """
        # WarpingLayer uses F.grid_sample, which expects normalized grid
        # we still output unnormalized flow for the convenience of comparing EPEs with FlowNet2 and original code
        # so here we need to denormalize the flow
        flow_for_grip = torch.zeros_like(flow).to(self.device)
        flow_for_grip[:, 0, :, :] = flow[:, 0, :, :] / ((flow.size(3) - 1.0) / 2.0)
        flow_for_grip[:, 1, :, :] = flow[:, 1, :, :] / ((flow.size(2) - 1.0) / 2.0)

        grid = (get_grid(x).to(self.device) + flow_for_grip).permute(0, 2, 3, 1)
        return F_torch.grid_sample(x, grid, align_corners=True)


class CenterPad_threshold(object):
    def __init__(self, image_size, threshold=3 / 4):
        self.height = image_size[0]
        self.width = image_size[1]
        self.threshold = threshold

    def __call__(self, image):
        # pad the image to 16:9
        # pad height
        I = np.array(image)

        # for padded input
        height_old = np.size(I, 0)
        width_old = np.size(I, 1)
        old_size = [height_old, width_old]
        height = self.height
        width = self.width
        I_pad = np.zeros((height, width, np.size(I, 2)))

        ratio = height / width

        if height_old / width_old == ratio:
            if height_old == height:
                return Image.fromarray(I.astype(np.uint8))
            new_size = [int(x * height / height_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            return Image.fromarray(I_resize.astype(np.uint8))

        if height_old / width_old > self.threshold:
            width_new, height_new = width_old, int(width_old * self.threshold)
            height_margin = height_old - height_new
            height_crop_start = height_margin // 2
            I_crop = I[height_crop_start : (height_crop_start + height_new), :, :]
            I_resize = resize(I_crop, [height, width], mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)

            return Image.fromarray(I_resize.astype(np.uint8))

        if height_old / width_old > ratio:  # pad the width and crop
            new_size = [int(x * width / width_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_height = (height_resize - height) // 2
            I_pad[:, :, :] = I_resize[start_height : (start_height + height), :, :]
        else:  # pad the height and crop
            new_size = [int(x * height / height_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_width = (width_resize - width) // 2
            I_pad[:, :, :] = I_resize[:, start_width : (start_width + width), :]

        return Image.fromarray(I_pad.astype(np.uint8))


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        inputs[0:1, :, :] = F.normalize(inputs[0:1, :, :], 50, 1)
        inputs[1:3, :, :] = F.normalize(inputs[1:3, :, :], (0, 0), (1, 1))
        return inputs


class RGB2Lab(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        normed_inputs = np.float32(inputs) / 255.0
        rgb_inputs = cv2.cvtColor(normed_inputs, cv2.COLOR_RGB2LAB)
        return rgb_inputs


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return F.to_mytensor(inputs)


class CenterPad(object):
    def __init__(self, image_size):
        self.height = image_size[0]
        self.width = image_size[1]

    def __call__(self, image):
        # pad the image to 16:9
        # pad height
        I = np.array(image)

        # for padded input
        height_old = np.size(I, 0)
        width_old = np.size(I, 1)
        old_size = [height_old, width_old]
        height = self.height
        width = self.width
        I_pad = np.zeros((height, width, np.size(I, 2)))

        ratio = height / width
        if height_old / width_old == ratio:
            if height_old == height:
                return Image.fromarray(I.astype(np.uint8))
            new_size = [int(x * height / height_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            return Image.fromarray(I_resize.astype(np.uint8))

        if height_old / width_old > ratio:  # pad the width and crop
            new_size = [int(x * width / width_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_height = (height_resize - height) // 2
            I_pad[:, :, :] = I_resize[start_height : (start_height + height), :, :]
        else:  # pad the height and crop
            new_size = [int(x * height / height_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_width = (width_resize - width) // 2
            I_pad[:, :, :] = I_resize[:, start_width : (start_width + width), :]

        return Image.fromarray(I_pad.astype(np.uint8))


class CenterPadCrop_numpy(object):
    """
    pad the image according to the height
    """

    def __init__(self, image_size):
        self.height = image_size[0]
        self.width = image_size[1]

    def __call__(self, image, threshold=3 / 4):
        # pad the image to 16:9
        # pad height
        I = np.array(image)
        # for padded input
        height_old = np.size(I, 0)
        width_old = np.size(I, 1)
        old_size = [height_old, width_old]
        height = self.height
        width = self.width
        padding_size = width
        if image.ndim == 2:
            I_pad = np.zeros((width, width))
        else:
            I_pad = np.zeros((width, width, I.shape[2]))

        ratio = height / width
        if height_old / width_old == ratio:
            return I

        # if height_old / width_old > threshold:
        #     width_new, height_new = width_old, int(width_old * threshold)
        #     height_margin = height_old - height_new
        #     height_crop_start = height_margin // 2
        #     I_crop = I[height_start : (height_start + height_new), :]
        #     I_resize = resize(
        #         I_crop, [height, width], mode="reflect", preserve_range=True, clip=False, anti_aliasing=True
        #     )
        #     return I_resize

        if height_old / width_old > ratio:  # pad the width and crop
            new_size = [int(x * width / width_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_height = (height_resize - height) // 2
            start_height_block = (padding_size - height) // 2
            if image.ndim == 2:
                I_pad[start_height_block : (start_height_block + height), :] = I_resize[
                    start_height : (start_height + height), :
                ]
            else:
                I_pad[start_height_block : (start_height_block + height), :, :] = I_resize[
                    start_height : (start_height + height), :, :
                ]
        else:  # pad the height and crop
            new_size = [int(x * height / height_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_width = (width_resize - width) // 2
            start_width_block = (padding_size - width) // 2
            if image.ndim == 2:
                I_pad[:, start_width_block : (start_width_block + width)] = I_resize[:, start_width : (start_width + width)]

            else:
                I_pad[:, start_width_block : (start_width_block + width), :] = I_resize[
                    :, start_width : (start_width + width), :
                ]

        crop_start_height = (I_pad.shape[0] - height) // 2
        crop_start_width = (I_pad.shape[1] - width) // 2

        if image.ndim == 2:
            return I_pad[crop_start_height : (crop_start_height + height), crop_start_width : (crop_start_width + width)]
        else:
            return I_pad[crop_start_height : (crop_start_height + height), crop_start_width : (crop_start_width + width), :]


@jit(nopython=True, nogil=True)
def biInterpolation_cpu(distorted, i, j):
        i = np.uint16(i)
        j = np.uint16(j)
        Q11 = distorted[j, i]
        Q12 = distorted[j, i + 1]
        Q21 = distorted[j + 1, i]
        Q22 = distorted[j + 1, i + 1]

        return np.int8(
            Q11 * (i + 1 - i) * (j + 1 - j) + Q12 * (i - i) * (j + 1 - j) + Q21 * (i + 1 - i) * (j - j) + Q22 * (i - i) * (j - j)
        )

@jit(nopython=True, nogil=True)
def iterSearchShader_cpu(padu, padv, xr, yr, W, H, maxIter, precision):
    # print('processing location', (xr, yr))
    #
    if abs(padu[yr, xr]) < precision and abs(padv[yr, xr]) < precision:
        return xr, yr

        # Our initialize method in this paper, can see the overleaf for detail
    if (xr + 1) <= (W - 1):
        dif = padu[yr, xr + 1] - padu[yr, xr]
    else:
        dif = padu[yr, xr] - padu[yr, xr - 1]
    u_next = padu[yr, xr] / (1 + dif)
    if (yr + 1) <= (H - 1):
        dif = padv[yr + 1, xr] - padv[yr, xr]
    else:
        dif = padv[yr, xr] - padv[yr - 1, xr]
    v_next = padv[yr, xr] / (1 + dif)
    i = xr - u_next
    j = yr - v_next
    i_int = int(i)
    j_int = int(j)

    # The same as traditional iterative search method
    for _ in range(maxIter):
        if not 0 <= i <= (W - 1) or not 0 <= j <= (H - 1):
            return i, j

        u11 = padu[j_int, i_int]
        v11 = padv[j_int, i_int]

        u12 = padu[j_int, i_int + 1]
        v12 = padv[j_int, i_int + 1]

        int1 = padu[j_int + 1, i_int]
        v21 = padv[j_int + 1, i_int]

        int2 = padu[j_int + 1, i_int + 1]
        v22 = padv[j_int + 1, i_int + 1]

        u = (
            u11 * (i_int + 1 - i) * (j_int + 1 - j)
            + u12 * (i - i_int) * (j_int + 1 - j)
            + int1 * (i_int + 1 - i) * (j - j_int)
            + int2 * (i - i_int) * (j - j_int)
        )

        v = (
            v11 * (i_int + 1 - i) * (j_int + 1 - j)
            + v12 * (i - i_int) * (j_int + 1 - j)
            + v21 * (i_int + 1 - i) * (j - j_int)
            + v22 * (i - i_int) * (j - j_int)
        )

        i_next = xr - u
        j_next = yr - v

        if abs(i - i_next) < precision and abs(j - j_next) < precision:
            return i, j

        i = i_next
        j = j_next

    # if the search doesn't converge within max iter, it will return the last iter result
    return i_next, j_next

@jit(nopython=True, nogil=True)
def iterSearch_cpu(distortImg, resultImg, padu, padv, W, H, maxIter=5, precision=1e-2):
    for xr in range(W):
        for yr in range(H):
            # (xr, yr) is the point in result image, (i, j) is the search result in distorted image
            i, j = iterSearchShader_cpu(padu, padv, xr, yr, W, H, maxIter, precision)

            # reflect the pixels outside the border
            if i > W - 1:
                i = 2 * W - 1 - i
            if i < 0:
                i = -i
            if j > H - 1:
                j = 2 * H - 1 - j
            if j < 0:
                j = -j

            # Bilinear interpolation to get the pixel at (i, j) in distorted image
            resultImg[yr, xr, 0] = biInterpolation_cpu(
                distortImg[:, :, 0],
                i,
                j,
            )
            resultImg[yr, xr, 1] = biInterpolation_cpu(
                distortImg[:, :, 1],
                i,
                j,
            )
            resultImg[yr, xr, 2] = biInterpolation_cpu(
                distortImg[:, :, 2],
                i,
                j,
            )
    return None


def forward_mapping_cpu(source_image, u, v, maxIter=5, precision=1e-2):
    """
    warp the image according to the forward flow
    u: horizontal
    v: vertical
    """
    H = source_image.shape[0]
    W = source_image.shape[1]

    distortImg = np.array(np.zeros((H + 1, W + 1, 3)), dtype=np.uint8)
    distortImg[0:H, 0:W] = source_image[0:H, 0:W]
    distortImg[H, 0:W] = source_image[H - 1, 0:W]
    distortImg[0:H, W] = source_image[0:H, W - 1]
    distortImg[H, W] = source_image[H - 1, W - 1]

    padu = np.array(np.zeros((H + 1, W + 1)), dtype=np.float32)
    padu[0:H, 0:W] = u[0:H, 0:W]
    padu[H, 0:W] = u[H - 1, 0:W]
    padu[0:H, W] = u[0:H, W - 1]
    padu[H, W] = u[H - 1, W - 1]

    padv = np.array(np.zeros((H + 1, W + 1)), dtype=np.float32)
    padv[0:H, 0:W] = v[0:H, 0:W]
    padv[H, 0:W] = v[H - 1, 0:W]
    padv[0:H, W] = v[0:H, W - 1]
    padv[H, W] = v[H - 1, W - 1]

    resultImg = np.array(np.zeros((H, W, 3)), dtype=np.uint8)
    iterSearch_cpu(distortImg, resultImg, padu, padv, W, H, maxIter, precision)
    return resultImg

class Distortion_with_flow_cpu(object):
    """Elastic distortion"""

    def __init__(self, maxIter=3, precision=1e-3):
        self.maxIter = maxIter
        self.precision = precision

    def __call__(self, inputs, dx, dy):
        inputs = np.array(inputs)
        shape = inputs.shape[0], inputs.shape[1]
        remap_image = forward_mapping_cpu(inputs, dy, dx, maxIter=self.maxIter, precision=self.precision)

        return Image.fromarray(remap_image)

@cuda.jit(device=True)
def biInterpolation_gpu(distorted, i, j):
    i = int(i)
    j = int(j)
    Q11 = distorted[j, i]
    Q12 = distorted[j, i + 1]
    Q21 = distorted[j + 1, i]
    Q22 = distorted[j + 1, i + 1]

    return np.int8(
        Q11 * (i + 1 - i) * (j + 1 - j) + Q12 * (i - i) * (j + 1 - j) + Q21 * (i + 1 - i) * (j - j) + Q22 * (i - i) * (j - j)
    )

@cuda.jit(device=True)
def iterSearchShader_gpu(padu, padv, xr, yr, W, H, maxIter, precision):
    # print('processing location', (xr, yr))
    #
    if abs(padu[yr, xr]) < precision and abs(padv[yr, xr]) < precision:
        return xr, yr

        # Our initialize method in this paper, can see the overleaf for detail
    if (xr + 1) <= (W - 1):
        dif = padu[yr, xr + 1] - padu[yr, xr]
    else:
        dif = padu[yr, xr] - padu[yr, xr - 1]
    u_next = padu[yr, xr] / (1 + dif)
    if (yr + 1) <= (H - 1):
        dif = padv[yr + 1, xr] - padv[yr, xr]
    else:
        dif = padv[yr, xr] - padv[yr - 1, xr]
    v_next = padv[yr, xr] / (1 + dif)
    i = xr - u_next
    j = yr - v_next
    i_int = int(i)
    j_int = int(j)

    # The same as traditional iterative search method
    for _ in range(maxIter):
        if not 0 <= i <= (W - 1) or not 0 <= j <= (H - 1):
            return i, j

        u11 = padu[j_int, i_int]
        v11 = padv[j_int, i_int]

        u12 = padu[j_int, i_int + 1]
        v12 = padv[j_int, i_int + 1]

        int1 = padu[j_int + 1, i_int]
        v21 = padv[j_int + 1, i_int]

        int2 = padu[j_int + 1, i_int + 1]
        v22 = padv[j_int + 1, i_int + 1]

        u = (
            u11 * (i_int + 1 - i) * (j_int + 1 - j)
            + u12 * (i - i_int) * (j_int + 1 - j)
            + int1 * (i_int + 1 - i) * (j - j_int)
            + int2 * (i - i_int) * (j - j_int)
        )

        v = (
            v11 * (i_int + 1 - i) * (j_int + 1 - j)
            + v12 * (i - i_int) * (j_int + 1 - j)
            + v21 * (i_int + 1 - i) * (j - j_int)
            + v22 * (i - i_int) * (j - j_int)
        )

        i_next = xr - u
        j_next = yr - v

        if abs(i - i_next) < precision and abs(j - j_next) < precision:
            return i, j

        i = i_next
        j = j_next

    # if the search doesn't converge within max iter, it will return the last iter result
    return i_next, j_next

@cuda.jit
def iterSearch_gpu(distortImg, resultImg, padu, padv, W, H, maxIter=5, precision=1e-2):
    
    start_x, start_y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)
    
    for xr in range(start_x, W, stride_x):
        for yr in range(start_y, H, stride_y):

            i,j = iterSearchShader_gpu(padu, padv, xr, yr, W, H, maxIter, precision)

            if i > W - 1:
                i = 2 * W - 1 - i
            if i < 0:
                i = -i
            if j > H - 1:
                j = 2 * H - 1 - j
            if j < 0:
                j = -j

            resultImg[yr, xr,0] = biInterpolation_gpu(distortImg[:,:,0], i, j)
            resultImg[yr, xr,1] = biInterpolation_gpu(distortImg[:,:,1], i, j)
            resultImg[yr, xr,2] = biInterpolation_gpu(distortImg[:,:,2], i, j)
    return None

def forward_mapping_gpu(source_image, u, v, maxIter=5, precision=1e-2):
    """
    warp the image according to the forward flow
    u: horizontal
    v: vertical
    """
    H = source_image.shape[0]
    W = source_image.shape[1]

    resultImg = np.array(np.zeros((H, W, 3)), dtype=np.uint8)

    distortImg = np.array(np.zeros((H + 1, W + 1, 3)), dtype=np.uint8)
    distortImg[0:H, 0:W] = source_image[0:H, 0:W]
    distortImg[H, 0:W] = source_image[H - 1, 0:W]
    distortImg[0:H, W] = source_image[0:H, W - 1]
    distortImg[H, W] = source_image[H - 1, W - 1]

    padu = np.array(np.zeros((H + 1, W + 1)), dtype=np.float32)
    padu[0:H, 0:W] = u[0:H, 0:W]
    padu[H, 0:W] = u[H - 1, 0:W]
    padu[0:H, W] = u[0:H, W - 1]
    padu[H, W] = u[H - 1, W - 1]

    padv = np.array(np.zeros((H + 1, W + 1)), dtype=np.float32)
    padv[0:H, 0:W] = v[0:H, 0:W]
    padv[H, 0:W] = v[H - 1, 0:W]
    padv[0:H, W] = v[0:H, W - 1]
    padv[H, W] = v[H - 1, W - 1]

    padu = cuda.to_device(padu)
    padv = cuda.to_device(padv)
    distortImg = cuda.to_device(distortImg)
    resultImg = cuda.to_device(resultImg)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(W / threadsperblock[0])
    blockspergrid_y = math.ceil(H / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)


    iterSearch_gpu[blockspergrid, threadsperblock](distortImg, resultImg, padu, padv, W, H, maxIter, precision)
    resultImg = resultImg.copy_to_host()
    return resultImg

class Distortion_with_flow_gpu(object):

    def __init__(self, maxIter=3, precision=1e-3):
        self.maxIter = maxIter
        self.precision = precision
    
    def __call__(self, inputs, dx, dy):
        inputs = np.array(inputs)
        shape = inputs.shape[0], inputs.shape[1]
        remap_image = forward_mapping_gpu(inputs, dy, dx, maxIter=self.maxIter, precision=self.precision)

        return Image.fromarray(remap_image)

def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, "rb")
    try:
        magic = np.fromfile(f, np.float32, count=1)[0]  # For Python3.x
    except:
        magic = np.fromfile(f, np.float32, count=1)  # For Python2.x
    data2d = None
    if (202021.25 != magic)and(123.25!=magic):
        print("Magic number incorrect. Invalid .flo file")
    elif (123.25==magic):
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        # print("Reading %d x %d flo file" % (h, w))
        data2d = np.fromfile(f, np.float16, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    elif (202021.25 == magic):
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        # print("Reading %d x %d flo file" % (h, w))
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d.astype(np.float32)

class LossHandler:
    def __init__(self):
        self.loss_dict = {}
        self.count_sample = 0

    def add_loss(self, key, loss):
        if key not in self.loss_dict:
            self.loss_dict[key] = 0
        self.loss_dict[key] += loss

    def get_loss(self, key):
        return self.loss_dict[key] / self.count_sample

    def count_one_sample(self):
        self.count_sample += 1

    def reset(self):
        self.loss_dict = {}
        self.count_sample = 0


class TimeHandler:
    def __init__(self):
        self.time_handler = {}

    def compute_time(self, key):
        if key not in self.time_handler:
            self.time_handler[key] = time.time()
            return None
        else:
            return time.time() - self.time_handler.pop(key)


def print_num_params(model, is_trainable=False):
    model_name = model.__class__.__name__.ljust(30)

    if is_trainable:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"| TRAINABLE | {model_name} | {('{:,}'.format(num_params)).rjust(10)} |")
    else:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"|  GENERAL  | {model_name} | {('{:,}'.format(num_params)).rjust(10)} |")

    return num_params
