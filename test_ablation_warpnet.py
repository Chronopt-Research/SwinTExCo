import numpy as np
import os
import argparse
import torch
from tqdm import tqdm
from PIL import Image
from PIL import ImageEnhance as IE
from collections import OrderedDict
import torchvision.transforms as T
from utils.convert_folder_to_video import convert_frames_to_video

from src.models.CNN.ColorVidNet import ColorVidNet
from src.models.vit.embed import SwinModel
from src.models.CNN.NonlocalNet import AblationWarpNet as WarpNet
from src.models.CNN.FrameColor import frame_colorization
from src.utils import (
    RGB2Lab,
    ToTensor,
    Normalize,
    uncenter_l,
    tensor_lab2rgb
)

from src.models.vit.utils import load_params

def save_frames(predicted_rgb, output_path, frame_name):
    if predicted_rgb is not None:
        os.makedirs(os.path.join(output_path), exist_ok=True)
        predicted_rgb = np.transpose(predicted_rgb, (1,2,0))
        pil_img = Image.fromarray(predicted_rgb)
        pil_img.save(os.path.join(output_path, frame_name))
        
def upscale_image(large_IA_l, I_current_ab_predict):
    H, W = large_IA_l.shape[2:]
    large_current_ab_predict = torch.nn.functional.interpolate(I_current_ab_predict, 
                                                               size=(H,W), 
                                                               mode="bilinear", 
                                                               align_corners=False)
    large_IA_l = torch.cat((large_IA_l, large_current_ab_predict.cpu()), dim=1)
    large_current_rgb_predict = tensor_lab2rgb(large_IA_l)
    return large_current_rgb_predict

def preprocess_reference(img):
    color_enhancer = IE.Color(img)
    img = color_enhancer.enhance(1.5)
    return img

def main(args):
    frames_list = os.listdir(args.input_video)
    frames_list.sort()
    
    # Preprocess reference image
    frame_ref = Image.open(args.input_ref).convert('RGB')
    frame_ref = preprocess_reference(frame_ref)
    
    I_last_lab_predict = None

    IB_lab = image_preprocessor(frame_ref)
    IB_lab = IB_lab.unsqueeze(0).to(device)

    with torch.no_grad():
        I_reference_lab = IB_lab
        I_reference_l = I_reference_lab[:, 0:1, :, :]
        I_reference_ab = I_reference_lab[:, 1:3, :, :]
        I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1)).to(device)
        features_B = embed_net(I_reference_rgb)
    
    for frame_name in tqdm(frames_list):
        curr_frame = Image.open(os.path.join(args.input_video, frame_name)).convert("RGB")
        large_IA_lab = ToTensor()(RGB2Lab()(curr_frame)).unsqueeze(0)
        large_IA_l = large_IA_lab[:, 0:1, :, :]
        
        IA_lab = image_preprocessor(curr_frame)
        IA_lab = IA_lab.unsqueeze(0).to(device)
        IA_l = IA_lab[:, 0:1, :, :]
        if I_last_lab_predict is None:
                I_last_lab_predict = torch.zeros_like(IA_lab).to(device)
        

        with torch.no_grad():
            I_current_ab_predict, _ = frame_colorization(
                IA_l,
                I_reference_lab,
                I_last_lab_predict,
                features_B,
                embed_net,
                nonlocal_net,
                colornet,
                luminance_noise=0,
                temperature=1e-10,
                joint_training=False
            )
            I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)
            
        IA_predict_rgb = upscale_image(large_IA_l, I_current_ab_predict)
        IA_predict_rgb = (IA_predict_rgb.squeeze(0).cpu().numpy() * 255.)
        IA_predict_rgb = np.clip(IA_predict_rgb, 0, 255).astype(np.uint8)
        save_frames(IA_predict_rgb, args.output_video, frame_name)
    
    if args.export_video:
        convert_frames_to_video(args.output_video, os.path.join(args.output_video, 'output_video.mp4'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--input_ref", type=str, required=True)
    parser.add_argument("--output_video", type=str, default="output/")
    parser.add_argument("--export_video", action="store_true")
    parser.add_argument("--backbone", type=str, default="swinv2-cr-t-224")
    parser.add_argument("--weight_path", type=str, default="checkpoints/epoch_20/")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Initialize preprocessor
    image_preprocessor = T.Compose([
        T.Resize((224,224)),
        RGB2Lab(),
        ToTensor(),
        Normalize()
    ])
    
    # Initialize models
    embed_net=SwinModel(pretrained_model=args.backbone, device=device).to(device)
    nonlocal_net = WarpNet(feature_channel=128).to(device)
    colornet=ColorVidNet(7).to(device)

    embed_net.eval()
    nonlocal_net.eval()
    colornet.eval()

    # Load weights
    embed_net_params = load_params(os.path.join(args.weight_path, "embed_net.pth"),device=device)
    nonlocal_net_params = load_params(os.path.join(args.weight_path, "nonlocal_net.pth"),device=device)
    colornet_params = load_params(os.path.join(args.weight_path, "colornet.pth"),device=device)


    embed_net.load_state_dict(embed_net_params, strict=True)
    nonlocal_net.load_state_dict(nonlocal_net_params, strict=True)
    colornet.load_state_dict(colornet_params, strict=True)
    
    main(args)