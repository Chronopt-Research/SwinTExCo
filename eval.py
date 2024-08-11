from src.metrics import calc_ssim, calc_psnr, LPIPS_utils, FID_utils, calc_cdc
import argparse
import glob
from PIL import Image
from tqdm import tqdm
import torch
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='input')
    parser.add_argument('--gt_folder_name', type=str, default='gt')
    parser.add_argument('--pred_folder_name', type=str, default='pred')
    parser.add_argument('--log_metric_file', type=str, default='log_metric.csv')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    F = open(args.log_metric_file, 'w')
    F.write('video_name,frame_number,lpips,fid,ssim,psnr,cdc\n')
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    video_list = os.listdir(args.input_dir)
    
    lpips_utils = LPIPS_utils(device=device)
    fid_utils = FID_utils(device=device)
    
    for video_name in tqdm(video_list, desc='Processing videos'):
        gt_folder_path = os.path.join(args.input_dir, video_name, args.gt_folder_name)
        pred_folder_path = os.path.join(args.input_dir, video_name, args.pred_folder_name)
        
        gt_list = glob.glob(os.path.join(gt_folder_path, '*.png')) \
            + glob.glob(os.path.join(gt_folder_path, '*.jpg')) \
            + glob.glob(os.path.join(gt_folder_path, '*.jpeg'))
        pred_list = glob.glob(os.path.join(pred_folder_path, '*.png')) \
            + glob.glob(os.path.join(pred_folder_path, '*.jpg')) \
            + glob.glob(os.path.join(pred_folder_path, '*.jpeg'))
        
        gt_list.sort()
        pred_list.sort()
        
        if len(gt_list) != len(pred_list):
            print(f'Warning: the number of frames in gt and pred of {video_name} is not equal!')
            continue
            
        lpips_list = []
        fid_list = []
        ssim_list = []
        psnr_list = []
        cdc_list = []
        
        for frame_idx, (gt_path, pred_path) in tqdm(enumerate(zip(gt_list, pred_list)), desc='Calculating metrics'):
            gt_img = Image.open(gt_path)
            pred_img = Image.open(pred_path)
            
            lpips_list.append(lpips_utils.compare_lpips(pred_img, gt_img))
            fid_list.append(fid_utils.score(pred_img, gt_img))
            ssim_list.append(calc_ssim(pred_img, gt_img))
            psnr_list.append(calc_psnr(pred_img, gt_img))
            
            F.write(f'{video_name},{frame_idx},{lpips_list[-1]},{fid_list[-1]},{ssim_list[-1]},{psnr_list[-1]},{0}\n')

        
        cdc_list.append(calc_cdc(pred_folder_path))
        F.write(f'{video_name},-1,{np.mean(lpips_list)},{np.mean(fid_list)},{np.mean(ssim_list)},{np.mean(psnr_list)},{cdc_list[-1]}\n')
        F.flush()
    F.close()