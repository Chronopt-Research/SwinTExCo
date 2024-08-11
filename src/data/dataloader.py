import numpy as np
import pandas as pd
from src.utils import (
    CenterPadCrop_numpy,
    Distortion_with_flow_cpu,
    Distortion_with_flow_gpu,
    Normalize,
    RGB2Lab,
    ToTensor,
    Normalize,
    RGB2Lab,
    ToTensor,
    CenterPad,
    read_flow,
    SquaredPadding
)
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from numpy import random
import os
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import map_coordinates
import glob



def image_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class CenterCrop(object):
    """
    center crop the numpy array
    """

    def __init__(self, image_size):
        self.h0, self.w0 = image_size

    def __call__(self, input_numpy):
        if input_numpy.ndim == 3:
            h, w, channel = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0, channel))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0, :
            ]
        else:
            h, w = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0
            ]
        return output_numpy


class VideosDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        video_data_root,
        flow_data_root,
        mask_data_root,
        imagenet_folder,
        annotation_file_path,
        image_size,
        num_refs=5, # max = 20
        image_transform=None,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
    ):
        self.video_data_root = video_data_root
        self.flow_data_root = flow_data_root
        self.mask_data_root = mask_data_root
        self.imagenet_folder = imagenet_folder
        self.image_transform = image_transform
        self.CenterPad = CenterPad(image_size)
        self.Resize = transforms.Resize(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = transforms.CenterCrop(image_size)
        self.SquaredPadding = SquaredPadding(image_size[0])
        self.num_refs = num_refs

        assert os.path.exists(self.video_data_root), "find no video dataroot"
        assert os.path.exists(self.flow_data_root), "find no flow dataroot"
        assert os.path.exists(self.imagenet_folder), "find no imagenet folder"
        # self.epoch = epoch
        self.image_pairs = pd.read_csv(annotation_file_path, dtype=str)
        self.real_len = len(self.image_pairs)
        # self.image_pairs = pd.concat([self.image_pairs] * self.epoch, ignore_index=True)
        self.real_reference_probability = real_reference_probability
        self.nonzero_placeholder_probability = nonzero_placeholder_probability
        print("##### parsing image pairs in %s: %d pairs #####" % (video_data_root, self.__len__()))

    def __getitem__(self, index):
        (
            video_name,
            prev_frame,
            current_frame,
            flow_forward_name,
            mask_name,
            reference_1_name,
            reference_2_name,
            reference_3_name,
            reference_4_name,
            reference_5_name
        ) = self.image_pairs.iloc[index, :5+self.num_refs].values.tolist()

        video_path = os.path.join(self.video_data_root, video_name)
        flow_path = os.path.join(self.flow_data_root, video_name)
        mask_path = os.path.join(self.mask_data_root, video_name)
        
        prev_frame_path = os.path.join(video_path, prev_frame)
        current_frame_path = os.path.join(video_path, current_frame)
        list_frame_path = glob.glob(os.path.join(video_path, '*'))
        list_frame_path.sort()
        
        reference_1_path = os.path.join(self.imagenet_folder, reference_1_name)
        reference_2_path = os.path.join(self.imagenet_folder, reference_2_name)
        reference_3_path = os.path.join(self.imagenet_folder, reference_3_name)
        reference_4_path = os.path.join(self.imagenet_folder, reference_4_name)
        reference_5_path = os.path.join(self.imagenet_folder, reference_5_name)
        
        flow_forward_path = os.path.join(flow_path, flow_forward_name)
        mask_path = os.path.join(mask_path, mask_name)
        
        #reference_gt_1_path = prev_frame_path
        #reference_gt_2_path = current_frame_path
        try:
            I1 = Image.open(prev_frame_path).convert("RGB")
            I2 = Image.open(current_frame_path).convert("RGB")
            try:
                I_reference_video = Image.open(list_frame_path[0]).convert("RGB") # Get first frame
            except:
                I_reference_video = Image.open(current_frame_path).convert("RGB") # Get current frame if error
            
            reference_list = [reference_1_path, reference_2_path, reference_3_path, reference_4_path, reference_5_path]
            while reference_list: # run until getting the colorized reference
                reference_path = random.choice(reference_list)
                I_reference_video_real = Image.open(reference_path)
                if I_reference_video_real.mode == 'L':
                    reference_list.remove(reference_path)
                else:
                    break
            if not reference_list:
                I_reference_video_real = I_reference_video

            flow_forward = read_flow(flow_forward_path)  # numpy

            mask = Image.open(mask_path)  # PIL
            mask = self.Resize(mask)
            mask = np.array(mask)
            # mask = self.SquaredPadding(mask, return_pil=False, return_paddings=False)
            # binary mask
            mask[mask < 240] = 0
            mask[mask >= 240] = 1
            mask = self.ToTensor(mask)
            
            # transform
            I1 = self.image_transform(I1)
            I2 = self.image_transform(I2)
            I_reference_video = self.image_transform(I_reference_video)
            I_reference_video_real = self.image_transform(I_reference_video_real)
            flow_forward = self.ToTensor(flow_forward)
            flow_forward = self.Resize(flow_forward)#, return_pil=False, return_paddings=False, dtype=np.float32)
            

            if np.random.random() < self.real_reference_probability:
                I_reference_output = I_reference_video_real  # Use reference from imagenet
                placeholder = torch.zeros_like(I1)
                self_ref_flag = torch.zeros_like(I1)
            else:
                I_reference_output = I_reference_video  # Use reference from ground truth
                placeholder = I2 if np.random.random() < self.nonzero_placeholder_probability else torch.zeros_like(I1)
                self_ref_flag = torch.ones_like(I1)

            outputs = [
                I1,
                I2,
                I_reference_output,
                flow_forward,
                mask,
                placeholder,
                self_ref_flag,
                video_name + prev_frame,
                video_name + current_frame,
                reference_path
            ]

        except Exception as e:
            print("error in reading image pair: %s" % str(self.image_pairs[index]))
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return len(self.image_pairs)


def parse_imgnet_images(pairs_file):
    pairs = []
    with open(pairs_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("|")
            image_a = line[0]
            image_b = line[1]
            pairs.append((image_a, image_b))
    return pairs


class VideosDataset_ImageNet(data.Dataset):
    def __init__(
        self,
        imagenet_data_root,
        pairs_file,
        image_size,
        transforms_imagenet=None,
        distortion_level=3,
        brightnessjitter=0,
        nonzero_placeholder_probability=0.5,
        extra_reference_transform=None,
        real_reference_probability=1,
        distortion_device='cpu'
    ):
        self.imagenet_data_root = imagenet_data_root
        self.image_pairs = pd.read_csv(pairs_file, names=['i1', 'i2'])
        self.transforms_imagenet_raw = transforms_imagenet
        self.extra_reference_transform = transforms.Compose(extra_reference_transform)
        self.real_reference_probability = real_reference_probability
        self.transforms_imagenet = transforms.Compose(transforms_imagenet)
        self.image_size = image_size
        self.real_len = len(self.image_pairs)
        self.distortion_level = distortion_level
        self.distortion_transform = Distortion_with_flow_cpu() if distortion_device == 'cpu' else Distortion_with_flow_gpu()
        self.brightnessjitter = brightnessjitter
        self.flow_transform = transforms.Compose([CenterPadCrop_numpy(self.image_size), ToTensor()])
        self.nonzero_placeholder_probability = nonzero_placeholder_probability
        self.ToTensor = ToTensor()
        self.Normalize = Normalize()
        print("##### parsing imageNet pairs in %s: %d pairs #####" % (imagenet_data_root, self.__len__()))

    def __getitem__(self, index):
        pa, pb = self.image_pairs.iloc[index].values.tolist()
        if np.random.random() > 0.5:
            pa, pb = pb, pa

        image_a_path = os.path.join(self.imagenet_data_root, pa)
        image_b_path = os.path.join(self.imagenet_data_root, pb)

        I1 = image_loader(image_a_path)
        I2 = I1
        I_reference_video = I1
        I_reference_video_real = image_loader(image_b_path)
        # print("i'm here get image 2")
        # generate the flow
        alpha = np.random.rand() * self.distortion_level
        distortion_range = 50
        random_state = np.random.RandomState(None)
        shape = self.image_size[0], self.image_size[1]
        # dx: flow on the vertical direction; dy: flow on the horizontal direction
        forward_dx = (
            gaussian_filter((random_state.rand(*shape) * 2 - 1), distortion_range, mode="constant", cval=0) * alpha * 1000
        )
        forward_dy = (
            gaussian_filter((random_state.rand(*shape) * 2 - 1), distortion_range, mode="constant", cval=0) * alpha * 1000
        )
        # print("i'm here get image 3")
        for transform in self.transforms_imagenet_raw:
            if type(transform) is RGB2Lab:
                I1_raw = I1
            I1 = transform(I1)
        for transform in self.transforms_imagenet_raw:
            if type(transform) is RGB2Lab:
                I2 = self.distortion_transform(I2, forward_dx, forward_dy)
                I2_raw = I2
            I2 = transform(I2)
        # print("i'm here get image 4")
        I2[0:1, :, :] = I2[0:1, :, :] + torch.randn(1) * self.brightnessjitter

        I_reference_video = self.extra_reference_transform(I_reference_video)
        for transform in self.transforms_imagenet_raw:
            I_reference_video = transform(I_reference_video)
        
        I_reference_video_real = self.transforms_imagenet(I_reference_video_real)
        # print("i'm here get image 5")
        flow_forward_raw = np.stack((forward_dy, forward_dx), axis=-1)
        flow_forward = self.flow_transform(flow_forward_raw)

        # update the mask for the pixels on the border
        grid_x, grid_y = np.meshgrid(np.arange(self.image_size[0]), np.arange(self.image_size[1]), indexing="ij")
        grid = np.stack((grid_y, grid_x), axis=-1)
        grid_warp = grid + flow_forward_raw
        location_y = grid_warp[:, :, 0].flatten()
        location_x = grid_warp[:, :, 1].flatten()
        I2_raw = np.array(I2_raw).astype(float)
        I21_r = map_coordinates(I2_raw[:, :, 0], np.stack((location_x, location_y)), cval=-1).reshape(
            (self.image_size[0], self.image_size[1])
        )
        I21_g = map_coordinates(I2_raw[:, :, 1], np.stack((location_x, location_y)), cval=-1).reshape(
            (self.image_size[0], self.image_size[1])
        )
        I21_b = map_coordinates(I2_raw[:, :, 2], np.stack((location_x, location_y)), cval=-1).reshape(
            (self.image_size[0], self.image_size[1])
        )
        I21_raw = np.stack((I21_r, I21_g, I21_b), axis=2)
        mask = np.ones((self.image_size[0], self.image_size[1]))
        mask[(I21_raw[:, :, 0] == -1) & (I21_raw[:, :, 1] == -1) & (I21_raw[:, :, 2] == -1)] = 0
        mask[abs(I21_raw - I1_raw).sum(axis=-1) > 50] = 0
        mask = self.ToTensor(mask)
        # print("i'm here get image 6")
        if np.random.random() < self.real_reference_probability:
            I_reference_output = I_reference_video_real
            placeholder = torch.zeros_like(I1)
            self_ref_flag = torch.zeros_like(I1)
        else:
            I_reference_output = I_reference_video
            placeholder = I2 if np.random.random() < self.nonzero_placeholder_probability else torch.zeros_like(I1)
            self_ref_flag = torch.ones_like(I1)

        # except Exception as e:
        #     if combo_path is not None:
        #         print("problem in ", combo_path)
        #     print("problem in, ", image_a_path)
        #     print(e)
        #     return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        # print("i'm here get image 7")
        return [I1, I2, I_reference_output, flow_forward, mask, placeholder, self_ref_flag, "holder", pb, pa]

    def __len__(self):
        return len(self.image_pairs)