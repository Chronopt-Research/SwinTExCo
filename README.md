<h1 align="center">SwinTExCo: Exemplar-based Video Colorization using Swin Transformer</h1>

<p align="center">Paper (Update soon) | <a href="https://huggingface.co/spaces/chronopt-research/SwinTExCo">🤗 Demo (CPU)</a> | <a href="https://huggingface.co/spaces/chronopt-research/SwinTExCo">🤗 Demo (GPU)</a></p>

<p align="center">
This is the official implementation of paper: <b>"SwinTExCo: Exemplar-based Video Colorization using Swin Transformer"</b></p>

> Video colorization represents a compelling domain within the field of Computer Vision. The traditional approach in this field relies on Convolutional Neural Networks (CNNs) to extract features from each video frame and employs a recurrent network to learn information between video frames. While demonstrating considerable success in colorization, most traditional CNNs suffer from a limited receptive field size, capturing local information within a fixed-sized window. Consequently, they struggle to directly grasp long-range dependencies or pixel relationships that span large image or video frame areas. To address this limitation, recent advancements in the field have leveraged Vision Transformers (ViT) and their variants to enhance performance. This article introduces Swin Transformer Exemplar-based Video Colorization (SwinTExCo), an end-to-end model for the video colorization process that incorporates the Swin Transformer architecture as the backbone. The experimental results demonstrate that our proposed method outperforms many other state-of-the-art methods in both quantitative and qualitative metrics. The achievements of this research have significant implications for the domain of documentary and history video restoration, contributing to the broader goal of preserving cultural heritage and facilitating a deeper understanding of historical events through enhanced audiovisual materials.

# News
- **Aug 1, 2024** - First decision from Expert Systems with Applications (ESwA) Journal.
- **Dec 12, 2023** - Submitted the manuscript to Expert Systems with Applications (ESwA) Journal.

# Setup
## Dataset
### Structures
The dataset contains multiple of sub-datasets which are mentioned in the paper. Each sub-dataset contains videos (already extracted to list of frames) with additional files: optical flow (.flo), mask (.pgm) and an annotation file (.csv).

Beside that, the dataset includes a txt file which are pairs of images collected from [ImageNet](https://www.image-net.org/) dataset.

### Download link
- [OneDrive](https://1drv.ms/f/s!Au00COvcS5dxgbYJp-mjOTgr2oP5OA?e=CmTT5m)


## Checkpoints
Update soon!

## Create environment
```bash
# Create env using conda
conda create -n swintexco python=3.10 -y
conda activate swintexco

# Install necessary packages
python -m pip install -r requirements.txt
```

## Train model

```bash
python train.py --video_data_root_list VIDEO_DATA_ROOT_LIST \
                --flow_data_root_list FLOW_DATA_ROOT_LIST \
                --mask_data_root_list MASK_DATA_ROOT_LIST \
                --data_root_imagenet DATA_ROOT_IMAGENET \
                --annotation_file_path_list ANNOTATION_FILE_PATH_LIST \
                --imagenet_pairs_file IMAGENET_PAIRS_FILE \
                --workers WORKERS \
                --batch_size BATCH_SIZE \
                --image_size IMAGE_SIZE [IMAGE_SIZE ...] \
                --ic IC \
                --epoch EPOCH \
                --resume_epoch RESUME_EPOCH \ 
                --resume \
                --pretrained_model PRETRAINED_MODEL \
                --load_pretrained_model \
                --pretrained_model_dir PRETRAINED_MODEL_DIR \
                --lr LR \
                --beta1 BETA1 \
                --lr_step LR_STEP \
                --lr_gamma LR_GAMMA \
                --checkpoint_dir CHECKPOINT_DIR \
                --checkpoint_step CHECKPOINT_STEP \
                --real_reference_probability REAL_REFERENCE_PROBABILITY \
                --nonzero_placeholder_probability NONZERO_PLACEHOLDER_PROBABILITY \
                --domain_invariant \
                --weigth_l1 WEIGTH_L1 \
                --weight_contextual WEIGHT_CONTEXTUAL \
                --weight_perceptual WEIGHT_PERCEPTUAL \
                --weight_smoothness WEIGHT_SMOOTHNESS \
                --weight_gan WEIGHT_GAN \
                --weight_nonlocal_smoothness WEIGHT_NONLOCAL_SMOOTHNESS \
                --weight_nonlocal_consistent WEIGHT_NONLOCAL_CONSISTENT \
                --weight_consistent WEIGHT_CONSISTENT \
                --luminance_noise LUMINANCE_NOISE \
                --permute_data \
                --contextual_loss_direction CONTEXTUAL_LOSS_DIRECTION \
                --epoch_train_discriminator EPOCH_TRAIN_DISCRIMINATOR \
                --vit_version VIT_VERSION \
                --use_dummy \
                --use_wandb \
                --wandb_token WANDB_TOKEN \
                --wandb_name WANDB_NAME
```

## Test model
```bash
python --input_video INPUT_VIDEO \
       --input_ref INPUT_REF \
       --output_video OUTPUT_VIDEO \
       --export_video \
       --backbone BACKBONE \
       --weight_path WEIGHT_PATH \
       --device DEVICE
```

## Results of Qualitative Evaluation
[Google Drive](https://1drv.ms/f/s!Au00COvcS5dxgpYWWmxlQGZjJmvQGQ?e=FD1hse)

# Citation
```
```
