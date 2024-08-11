import os
import sys
import wandb
import argparse
from tqdm import tqdm
from datetime import datetime
from zoneinfo import ZoneInfo
from time import gmtime, strftime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.transforms as torch_transforms
from torchvision.utils import make_grid

from src.losses import (
    ContextualLoss,
    ContextualLoss_forward,
    Perceptual_loss,
    consistent_loss_fn,
    discriminator_loss_fn,
    generator_loss_fn,
    l1_loss_fn,
    smoothness_loss_fn,
)
from src.models.CNN.GAN_models import Discriminator_x64_224
from src.models.CNN.ColorVidNet import ColorVidNet
from src.models.CNN.FrameColor import frame_colorization
from src.models.CNN.NonlocalNet import WeightedAverage_color, NonlocalWeightedAverage, WarpNet
from src.models.vit.embed import SwinModel
from src.data.dataloader import VideosDataset, VideosDataset_ImageNet
from src.utils import CenterPad_threshold
from src.utils import (
    RGB2Lab,
    ToTensor,
    Normalize,
    LossHandler,
    WarpingLayer,
    uncenter_l,
    tensor_lab2rgb,
    print_num_params
)
from src.models.vit.utils import load_params
from src.scheduler import PolynomialLR

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


parser = argparse.ArgumentParser()
parser.add_argument("--video_data_root_list", type=str)
parser.add_argument("--flow_data_root_list", type=str)
parser.add_argument("--mask_data_root_list", type=str)
parser.add_argument("--data_root_imagenet", type=str)
parser.add_argument("--annotation_file_path_list", type=str)
parser.add_argument("--imagenet_pairs_file", type=str)
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--image_size", nargs='+', type=int, default=[224, 224])
parser.add_argument("--ic", type=int, default=7)
parser.add_argument("--epoch", type=int, default=40)
parser.add_argument("--resume_epoch", type=int, default=0)
parser.add_argument("--resume", action='store_true')
parser.add_argument("--pretrained_model", default='swinv2-cr-t-224', type=str)
parser.add_argument("--load_pretrained_model", action='store_true')
parser.add_argument("--pretrained_model_dir", type=str, default='ckpt')
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--lr_step", type=int, default=1)
parser.add_argument("--lr_gamma", type=float, default=0.9)
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
parser.add_argument("--checkpoint_step", type=int, default=500)
parser.add_argument("--real_reference_probability", type=float, default=0.7)
parser.add_argument("--nonzero_placeholder_probability", type=float, default=0.0)
parser.add_argument("--domain_invariant", action='store_true')
parser.add_argument("--weigth_l1", type=float, default=2.0)
parser.add_argument("--weight_contextual", type=float, default="0.5")
parser.add_argument("--weight_perceptual", type=float, default="0.02")
parser.add_argument("--weight_smoothness", type=float, default="5.0")
parser.add_argument("--weight_gan", type=float, default="0.5")
parser.add_argument("--weight_nonlocal_smoothness", type=float, default="0.0")
parser.add_argument("--weight_nonlocal_consistent", type=float, default="0.0")
parser.add_argument("--weight_consistent", type=float, default="0.05")
parser.add_argument("--luminance_noise", type=float, default="2.0")
parser.add_argument("--permute_data", action='store_true')
parser.add_argument("--contextual_loss_direction", type=str, default="forward", help="forward or backward matching")
# parser.add_argument("--batch_accum_size", type=int, default=10)
parser.add_argument("--epoch_train_discriminator", type=int, default=3)
parser.add_argument("--vit_version", type=str, default="vit_tiny_patch16_384")

# parser.add_argument("--use_feature_transform", action='store_true')
# parser.add_argument("--head_out_idx", type=str, default="8,9,10,11")
parser.add_argument("--use_dummy", action='store_true')
parser.add_argument("--use_wandb", action='store_true')
parser.add_argument("--wandb_token", type=str, default="")
parser.add_argument("--wandb_name", type=str, default="")
    
def prepare_dataloader_ddp(dataset, batch_size=4, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            pin_memory=pin_memory, 
                            num_workers=num_workers, 
                            sampler=sampler)
    return dataloader

def is_master_process():
    ddp_rank = int(os.environ['RANK'])
    return ddp_rank == 0

def ddp_setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    return local_rank

def ddp_cleanup():
    dist.destroy_process_group()

def load_data():
    transforms_video = [
        torch_transforms.Resize(opt.image_size),
        RGB2Lab(),
        ToTensor(),
        Normalize(),
    ]

    train_dataset_videos = [
        VideosDataset(
            video_data_root=video_data_root,
            flow_data_root=flow_data_root,
            mask_data_root=mask_data_root,
            imagenet_folder=opt.data_root_imagenet,
            annotation_file_path=annotation_file_path,
            image_size=opt.image_size,
            image_transform=torch_transforms.Compose(transforms_video),
            real_reference_probability=opt.real_reference_probability,
            nonzero_placeholder_probability=opt.nonzero_placeholder_probability,
        )
        for video_data_root, flow_data_root, mask_data_root, annotation_file_path in zip(opt.video_data_root_list, 
                                                                                         opt.flow_data_root_list, 
                                                                                         opt.mask_data_root_list,
                                                                                         opt.annotation_file_path_list)
    ]

    transforms_imagenet = [CenterPad_threshold(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
    extra_reference_transform = [
        torch_transforms.RandomHorizontalFlip(0.5),
        torch_transforms.RandomResizedCrop(480, (0.98, 1.0), ratio=(0.8, 1.2)),
    ]

    train_dataset_imagenet = VideosDataset_ImageNet(
        imagenet_data_root=opt.data_root_imagenet,
        pairs_file=opt.imagenet_pairs_file,
        image_size=opt.image_size,
        transforms_imagenet=transforms_imagenet,
        distortion_level=4,
        brightnessjitter=5,
        nonzero_placeholder_probability=opt.nonzero_placeholder_probability,
        extra_reference_transform=extra_reference_transform,
        real_reference_probability=opt.real_reference_probability,
    )
    dataset_combined = ConcatDataset(train_dataset_videos + [train_dataset_imagenet])
    data_loader = prepare_dataloader_ddp(dataset_combined,
                                         batch_size=opt.batch_size,
                                         pin_memory=False, 
                                         num_workers=opt.workers)
    return data_loader

def save_checkpoints(saved_path):
    # Make directory if the folder doesn't exists
    os.makedirs(saved_path, exist_ok=True)
    
    # Save model
    torch.save(
            nonlocal_net.module.state_dict(),
            os.path.join(saved_path, "nonlocal_net.pth"),
    )
    torch.save(
        colornet.module.state_dict(),
        os.path.join(saved_path, "colornet.pth"),
    )
    torch.save(
        discriminator.module.state_dict(),
        os.path.join(saved_path, "discriminator.pth"),
    )
    torch.save(
        embed_net.state_dict(), 
        os.path.join(saved_path, "embed_net.pth")
    )
    
    # Save learning state for restoring train
    learning_state = {
        "epoch": epoch_num,
        "total_iter": total_iter,
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "optimizer_schedule_g": step_optim_scheduler_g.state_dict(),
        "optimizer_schedule_d": step_optim_scheduler_d.state_dict(),
    }
    
    torch.save(learning_state, os.path.join(saved_path, "learning_state.pth"))  
    
def training_logger():
    if (total_iter % opt.checkpoint_step == 0) or (total_iter == len(data_loader)):
        train_loss_dict = {"train/" + str(k): v / loss_handler.count_sample for k, v in loss_handler.loss_dict.items()}
        train_loss_dict["train/opt_g_lr_1"] = step_optim_scheduler_g.get_last_lr()[0]
        train_loss_dict["train/opt_g_lr_2"] = step_optim_scheduler_g.get_last_lr()[1]
        train_loss_dict["train/opt_d_lr"] = step_optim_scheduler_d.get_last_lr()[0]

        alert_text = f"l1_loss: {l1_loss.item()}\npercep_loss: {perceptual_loss.item()}\nctx_loss: {contextual_loss_total.item()}\ncst_loss: {consistent_loss.item()}\nsm_loss: {smoothness_loss.item()}\ntotal: {total_loss.item()}"

        if opt.use_wandb:
            wandb.log(train_loss_dict)
            wandb.alert(title=f"Progress training #{total_iter}", text=alert_text)

            for idx in range(I_predict_rgb.shape[0]):
                concated_I = make_grid(
                    [(I_predict_rgb[idx] * 255), (I_reference_rgb[idx] * 255), (I_current_rgb[idx] * 255)], nrow=3
                )
                wandb_concated_I = wandb.Image(
                    concated_I,
                    caption="[LEFT] Predict, [CENTER] Reference, [RIGHT] Ground truth\n[REF] {}, [FRAME] {}".format(
                        ref_path[idx], curr_frame_path[idx]
                    ),
                )
                wandb.log({f"example_{idx}": wandb_concated_I})
            
        # Save learning state checkpoint
        # save_checkpoints(os.path.join(opt.checkpoint_dir, 'runs'))
        loss_handler.reset()


def parse(parser, save=True):
    opt = parser.parse_args()
    args = vars(opt)

    print("------------------------------ Options -------------------------------")
    for k, v in sorted(args.items()):
        print("%s: %s" % (str(k), str(v)))
    print("-------------------------------- End ---------------------------------")

    if save:
        file_name = os.path.join("opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(os.path.basename(sys.argv[0]) + " " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")
            opt_file.write("------------------------------ Options -------------------------------\n")
            for k, v in sorted(args.items()):
                opt_file.write("%s: %s\n" % (str(k), str(v)))
            opt_file.write("-------------------------------- End ---------------------------------\n")
    return opt


def gpu_setup():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cudnn.benchmark = True
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    return device


if __name__ == "__main__":
    ####### SETUP #######
    torch.multiprocessing.set_start_method("spawn", force=True)
    # =============== GET PARSER OPTION ================
    opt = parse(parser)
    opt.video_data_root_list = opt.video_data_root_list.split(",")
    opt.flow_data_root_list = opt.flow_data_root_list.split(",")
    opt.mask_data_root_list = opt.mask_data_root_list.split(",")
    opt.annotation_file_path_list = opt.annotation_file_path_list.split(",")
    # opt.gpu_ids = list(map(int, opt.gpu_ids.split(",")))
    
    # =================== INIT WANDB ===================
    if opt.use_wandb:
        print("Save images to Wandb")
        if opt.wandb_token != "":
            try:
                wandb.login(key=opt.wandb_token)
            except:
                pass
    if opt.use_wandb:
        wandb.init(
            project="video-colorization",
            group=f"{opt.wandb_name} {datetime.now(tz=ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y/%m/%d_%H-%M-%S')}"
        )

    ###### SETUP DEVICE ######
    local_rank = ddp_setup()
    data_loader = load_data()

    colornet = DDP(ColorVidNet(opt.ic).to(local_rank),  device_ids=[local_rank], output_device=local_rank)
    nonlocal_net = DDP(WarpNet().to(local_rank), device_ids=[local_rank], output_device=local_rank)
    discriminator = DDP(Discriminator_x64_224(ndf=64).to(local_rank), device_ids=[local_rank], output_device=local_rank)
    weighted_layer_color = WeightedAverage_color().to(local_rank)
    nonlocal_weighted_layer = NonlocalWeightedAverage().to(local_rank)
    warping_layer = WarpingLayer(device=local_rank).to(local_rank)
    embed_net = SwinModel(pretrained_model=opt.pretrained_model, device=local_rank).to(local_rank)
    
    if is_master_process():
        # Print number of parameters
        print("-" * 59)
        print("|    TYPE   |          Model name            | Num params |")
        print("-" * 59)
        
        colornet_params = print_num_params(colornet)
        nonlocal_net_params = print_num_params(nonlocal_net)
        discriminator_params = print_num_params(discriminator)
        weighted_layer_color_params = print_num_params(weighted_layer_color)
        nonlocal_weighted_layer_params = print_num_params(nonlocal_weighted_layer)
        warping_layer_params = print_num_params(warping_layer)
        embed_net_params = print_num_params(embed_net)
        print("-" * 59)
        print(
            f"|   TOTAL   |                                | {('{:,}'.format(colornet_params+nonlocal_net_params+discriminator_params+weighted_layer_color_params+nonlocal_weighted_layer_params+warping_layer_params+embed_net_params)).rjust(10)} |"
        )
        print("-" * 59)
    
    if opt.use_wandb:
        wandb.watch(discriminator, log="all", log_freq=opt.checkpoint_step, idx=0)
        wandb.watch(embed_net, log="all", log_freq=opt.checkpoint_step, idx=1)
        wandb.watch(colornet, log="all", log_freq=opt.checkpoint_step, idx=2)
        wandb.watch(nonlocal_net, log="all", log_freq=opt.checkpoint_step, idx=3)

    

    ###### DEFINE LOSS FUNCTIONS ######
    perceptual_loss_fn = Perceptual_loss(opt.domain_invariant, opt.weight_perceptual)
    contextual_loss = ContextualLoss().to(local_rank)
    contextual_forward_loss = ContextualLoss_forward().to(local_rank)
    ###### DEFINE OPTIMIZERS ######
    optimizer_g = optim.AdamW(
        [
            {"params": nonlocal_net.parameters(), "lr": opt.lr},
            {"params": colornet.parameters(), "lr": 2 * opt.lr}
        ],
        betas=(0.5, 0.999),
        eps=1e-5,
        amsgrad=True,
    )
    
    optimizer_d = optim.AdamW(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=opt.lr,
        betas=(0.5, 0.999),
        amsgrad=True,
    )

    step_optim_scheduler_g = PolynomialLR(
        optimizer_g,
        step_size=opt.lr_step,
        iter_warmup=0,
        iter_max=len(data_loader) * opt.epoch,
        power=0.9,
        min_lr=1e-8
    )
    step_optim_scheduler_d = PolynomialLR(
        optimizer_d,
        step_size=opt.lr_step,
        iter_warmup=0,
        iter_max=len(data_loader) * opt.epoch,
        power=0.9,
        min_lr=1e-8
    )
    ###### DEFINE OTHERS ######
    downsampling_by2 = nn.AvgPool2d(kernel_size=2).to(local_rank)
    # timer_handler = TimeHandler()
    loss_handler = LossHandler()
    ###### TRAIN ######
    
    # USE PRETRAINED OR NOT
    if opt.load_pretrained_model:
        nonlocal_net.load_state_dict(load_params(os.path.join(opt.pretrained_model_dir, "nonlocal_net.pth"), 
                                                 local_rank, 
                                                 has_module=True))
        colornet.load_state_dict(load_params(os.path.join(opt.pretrained_model_dir, "colornet.pth"),
                                             local_rank,
                                             has_module=True))
        discriminator.load_state_dict(load_params(os.path.join(opt.pretrained_model_dir, "discriminator.pth"),
                                                  local_rank,
                                                  has_module=True))
        embed_net_params = load_params(os.path.join(opt.pretrained_model_dir, "embed_net.pth"),
                                       local_rank,
                                       has_module=False)
        
        embed_net.load_state_dict(embed_net_params)
        
        learning_checkpoint = torch.load(os.path.join(opt.pretrained_model_dir, "learning_state.pth"))
        optimizer_g.load_state_dict(learning_checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(learning_checkpoint["optimizer_d"])
        step_optim_scheduler_g.load_state_dict(learning_checkpoint["optimizer_schedule_g"])
        step_optim_scheduler_d.load_state_dict(learning_checkpoint["optimizer_schedule_d"])
        total_iter = learning_checkpoint['total_iter']
        start_epoch = learning_checkpoint['epoch']+1
    else:
        total_iter = 0
        start_epoch = 1
        
    
        
    for epoch_num in range(start_epoch, opt.epoch+1):
        data_loader.sampler.set_epoch(epoch_num-1)
        
        if is_master_process():
            train_progress_bar = tqdm(
                data_loader,
                desc =f'Epoch {epoch_num}[Training]',
                position = 0,
                leave = False
            )
        else:
            train_progress_bar = data_loader
        for iter, sample in enumerate(train_progress_bar):
            # timer_handler.compute_time("load_sample")
            total_iter += 1
            # =============== LOAD DATA SAMPLE ================
            (
                I_last_lab,  ######## (3, H, W)
                I_current_lab,  ##### (3, H, W)
                I_reference_lab,  ### (3, H, W)
                flow_forward,  ###### (2, H, W)
                mask,  ############## (1, H, W)
                placeholder_lab,  ### (3, H, W)
                self_ref_flag,  ##### (3, H, W)
                prev_frame_path,
                curr_frame_path,
                ref_path,
            ) = sample

            I_last_lab = I_last_lab.to(local_rank)
            I_current_lab = I_current_lab.to(local_rank)
            I_reference_lab = I_reference_lab.to(local_rank)
            flow_forward = flow_forward.to(local_rank)
            mask = mask.to(local_rank)
            placeholder_lab = placeholder_lab.to(local_rank)
            self_ref_flag = self_ref_flag.to(local_rank)

            I_last_l = I_last_lab[:, 0:1, :, :]
            I_last_ab = I_last_lab[:, 1:3, :, :]
            I_current_l = I_current_lab[:, 0:1, :, :]
            I_current_ab = I_current_lab[:, 1:3, :, :]
            I_reference_l = I_reference_lab[:, 0:1, :, :]
            I_reference_ab = I_reference_lab[:, 1:3, :, :]
            I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))

            features_B = embed_net(I_reference_rgb)
            B_feat_0, B_feat_1, B_feat_2, B_feat_3 = features_B

            # ================== COLORIZATION ==================
            # The last frame
            I_last_ab_predict, I_last_nonlocal_lab_predict = frame_colorization(
                IA_l=I_last_l,
                IB_lab=I_reference_lab,
                IA_last_lab=placeholder_lab,
                features_B=features_B,
                embed_net=embed_net,
                colornet=colornet,
                nonlocal_net=nonlocal_net,
                luminance_noise=opt.luminance_noise,
            )
            I_last_lab_predict = torch.cat((I_last_l, I_last_ab_predict), dim=1)

            # The current frame
            I_current_ab_predict, I_current_nonlocal_lab_predict = frame_colorization(
                IA_l=I_current_l,
                IB_lab=I_reference_lab,
                IA_last_lab=I_last_lab_predict,
                features_B=features_B,
                embed_net=embed_net,
                colornet=colornet,
                nonlocal_net=nonlocal_net,
                luminance_noise=opt.luminance_noise,
            )
            I_current_lab_predict = torch.cat((I_last_l, I_current_ab_predict), dim=1)

            # ================ UPDATE GENERATOR ================
            if opt.weight_gan > 0:
                optimizer_g.zero_grad()
                optimizer_d.zero_grad()
                fake_data_lab = torch.cat(
                    (
                        uncenter_l(I_current_l),
                        I_current_ab_predict,
                        uncenter_l(I_last_l),
                        I_last_ab_predict,
                    ),
                    dim=1,
                )
                real_data_lab = torch.cat(
                    (
                        uncenter_l(I_current_l),
                        I_current_ab,
                        uncenter_l(I_last_l),
                        I_last_ab,
                    ),
                    dim=1,
                )

                if opt.permute_data:
                    batch_index = torch.arange(-1, opt.batch_size - 1, dtype=torch.long)
                    real_data_lab = real_data_lab[batch_index, ...]

                discriminator_loss = discriminator_loss_fn(real_data_lab, fake_data_lab, discriminator)
                discriminator_loss.backward()
                optimizer_d.step()

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            # ================== COMPUTE LOSS ==================
            # L1 loss
            l1_loss = l1_loss_fn(I_current_ab, I_current_ab_predict) * opt.weigth_l1

            # Generator_loss. TODO: freeze this to train some first epoch
            if epoch_num > opt.epoch_train_discriminator:
                generator_loss = generator_loss_fn(real_data_lab, fake_data_lab, discriminator, opt.weight_gan, local_rank)

            # Perceptual Loss
            I_predict_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_current_l), I_current_ab_predict), dim=1))
            pred_feat_0, pred_feat_1, pred_feat_2, pred_feat_3 = embed_net(I_predict_rgb)

            I_current_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_current_l), I_current_ab), dim=1))
            A_feat_0, _, _, A_feat_3 = embed_net(I_current_rgb)

            perceptual_loss = perceptual_loss_fn(A_feat_3, pred_feat_3)

            # Contextual Loss
            contextual_style5_1 = torch.mean(contextual_forward_loss(pred_feat_3, B_feat_3.detach())) * 8
            contextual_style4_1 = torch.mean(contextual_forward_loss(pred_feat_2, B_feat_2.detach())) * 4
            contextual_style3_1 = torch.mean(contextual_forward_loss(pred_feat_1, B_feat_1.detach())) * 2
            contextual_style2_1 = torch.mean(contextual_forward_loss(pred_feat_0, B_feat_0.detach()))

            contextual_loss_total = (
                contextual_style5_1 + contextual_style4_1 + contextual_style3_1 + contextual_style2_1
            ) * opt.weight_contextual

            # Consistent Loss
            consistent_loss = consistent_loss_fn(
                I_current_lab_predict,
                I_last_ab_predict,
                I_current_nonlocal_lab_predict,
                I_last_nonlocal_lab_predict,
                flow_forward,
                mask,
                warping_layer,
                weight_consistent=opt.weight_consistent,
                weight_nonlocal_consistent=opt.weight_nonlocal_consistent,
                device=local_rank,
            )

            # Smoothness loss
            smoothness_loss = smoothness_loss_fn(
                I_current_l,
                I_current_lab,
                I_current_ab_predict,
                A_feat_0,
                weighted_layer_color,
                nonlocal_weighted_layer,
                weight_smoothness=opt.weight_smoothness,
                weight_nonlocal_smoothness=opt.weight_nonlocal_smoothness,
                device=local_rank
            )

            # Total loss
            total_loss = l1_loss + perceptual_loss + contextual_loss_total + consistent_loss + smoothness_loss
            if epoch_num > opt.epoch_train_discriminator:
                total_loss += generator_loss

            # Add loss to loss handler
            loss_handler.add_loss(key="total_loss", loss=total_loss.item())
            loss_handler.add_loss(key="l1_loss", loss=l1_loss.item())
            loss_handler.add_loss(key="perceptual_loss", loss=perceptual_loss.item())
            loss_handler.add_loss(key="contextual_loss", loss=contextual_loss_total.item())
            loss_handler.add_loss(key="consistent_loss", loss=consistent_loss.item())
            loss_handler.add_loss(key="smoothness_loss", loss=smoothness_loss.item())
            loss_handler.add_loss(key="discriminator_loss", loss=discriminator_loss.item())
            if epoch_num > opt.epoch_train_discriminator:
                loss_handler.add_loss(key="generator_loss", loss=generator_loss.item())
            loss_handler.count_one_sample()

            total_loss.backward()
            
            optimizer_g.step()
            step_optim_scheduler_g.step()
            step_optim_scheduler_d.step()
            
            training_logger()

        ####
        if is_master_process():
            save_checkpoints(os.path.join(opt.checkpoint_dir, f"epoch_{epoch_num}"))
    ####
    if opt.use_wandb:
        wandb.finish()
    ddp_cleanup()
