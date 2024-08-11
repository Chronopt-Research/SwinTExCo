from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import numpy as np
import lpips
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
import torch.nn as nn
import cv2
from scipy import stats
import os

def calc_ssim(pred_image, gt_image):
    '''
    Structural Similarity Index (SSIM) is a perceptual metric that quantifies the image quality degradation that is
    caused by processing such as data compression or by losses in data transmission.
    
    # Arguments
        img1: PIL.Image
        img2: PIL.Image
    # Returns
        ssim: float (-1.0, 1.0)
    '''
    pred_image = np.array(pred_image.convert('RGB')).astype(np.float32)
    gt_image = np.array(gt_image.convert('RGB')).astype(np.float32)
    ssim = structural_similarity(pred_image, gt_image, channel_axis=2, data_range=255.)
    return ssim

def calc_psnr(pred_image, gt_image):
    '''
    Peak Signal-to-Noise Ratio (PSNR) is an expression for the ratio between the maximum possible value (power) of a signal
    and the power of distorting noise that affects the quality of its representation.
    
    # Arguments
        img1: PIL.Image
        img2: PIL.Image
    # Returns
        psnr: float
    '''
    pred_image = np.array(pred_image.convert('RGB')).astype(np.float32)
    gt_image = np.array(gt_image.convert('RGB')).astype(np.float32)
    
    psnr = peak_signal_noise_ratio(gt_image, pred_image, data_range=255.)
    return psnr

class LPIPS_utils:
    def __init__(self, device = 'cuda'):
        self.loss_fn = lpips.LPIPS(net='vgg', spatial=True)  # Can set net = 'squeeze' or 'vgg'or 'alex'
        self.loss_fn = self.loss_fn.to(device)
        self.device = device
    
    def compare_lpips(self,img_fake, img_real, data_range=255.):         # input: torch 1 c h w    / h w c
        img_fake = torch.from_numpy(np.array(img_fake).astype(np.float32)/data_range)
        img_real = torch.from_numpy(np.array(img_real).astype(np.float32)/data_range)
        if img_fake.ndim==3:
            img_fake = img_fake.permute(2,0,1).unsqueeze(0)
            img_real = img_real.permute(2,0,1).unsqueeze(0)
        img_fake = img_fake.to(self.device)
        img_real = img_real.to(self.device)
        
        dist = self.loss_fn.forward(img_fake,img_real)
        return dist.mean().item()

class FID_utils(nn.Module):
    """Class for computing the Fréchet Inception Distance (FID) metric score.
    It is implemented as a class in order to hold the inception model instance
    in its state.
    Parameters
    ----------
    resize_input : bool (optional)
        Whether or not to resize the input images to the image size (299, 299)
        on which the inception model was trained. Since the model is fully
        convolutional, the score also works without resizing. In literature
        and when working with GANs people tend to set this value to True,
        however, for internal evaluation this is not necessary.
    device : str or torch.device
        The device on which to run the inception model.
    """

    def __init__(self, resize_input=True, device="cuda"):
        super(FID_utils, self).__init__()
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.model = InceptionV3(resize_input=resize_input).to(device)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model = InceptionV3([block_idx]).to(device)
        self.model = self.model.eval()

    def get_activations(self,batch):                   # 1 c h w
        with torch.no_grad():
            pred = self.model(batch)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            #pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            print("error in get activations!")
        #pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        return pred


    def _get_mu_sigma(self, batch,data_range):
        """Compute the inception mu and sigma for a batch of images.
        Parameters
        ----------
        images : np.ndarray
            A batch of images with shape (n_images,3, width, height).
        Returns
        -------
        mu : np.ndarray
            The array of mean activations with shape (2048,).
        sigma : np.ndarray
            The covariance matrix of activations with shape (2048, 2048).
        """
        # forward pass
        if batch.ndim ==3 and batch.shape[2]==3:
            batch=batch.permute(2,0,1).unsqueeze(0) 
        batch /= data_range           
        #batch = torch.tensor(batch)#.unsqueeze(1).repeat((1, 3, 1, 1))
        batch = batch.to(self.device, torch.float32)
        #(activations,) = self.model(batch)
        activations = self.get_activations(batch)
        activations = activations.detach().cpu().numpy().squeeze(3).squeeze(2)

        # compute statistics
        mu = np.mean(activations,axis=0)
        sigma = np.cov(activations, rowvar=False)

        return mu, sigma

    def score(self, images_1, images_2, data_range=255.):
        """Compute the FID score.
        The input batches should have the shape (n_images,3, width, height). or (h,w,3)
        Parameters
        ----------
        images_1 : np.ndarray
            First batch of images.
        images_2 : np.ndarray
            Section batch of images.
        Returns
        -------
        score : float
            The FID score.
        """
        images_1 = torch.from_numpy(np.array(images_1).astype(np.float32))
        images_2 = torch.from_numpy(np.array(images_2).astype(np.float32))
        images_1 = images_1.to(self.device)
        images_2 = images_2.to(self.device)
        
        mu_1, sigma_1 = self._get_mu_sigma(images_1,data_range)
        mu_2, sigma_2 = self._get_mu_sigma(images_2,data_range)
        score = calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)

        return score

def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * stats.entropy(p, M) + 0.5 * stats.entropy(q, M)


def compute_JS_bgr(input_dir, dilation=1):
    input_img_list = os.listdir(input_dir)
    input_img_list.sort()
    # print(input_img_list)

    hist_b_list = []   # [img1_histb, img2_histb, ...]
    hist_g_list = []
    hist_r_list = []
    
    for img_name in input_img_list:
        # print(os.path.join(input_dir, img_name))
        img_in = cv2.imread(os.path.join(input_dir, img_name))
        H, W, C = img_in.shape
        
        hist_b = cv2.calcHist([img_in], [0], None, [256], [0,256]) # B
        hist_g = cv2.calcHist([img_in], [1], None, [256], [0,256]) # G
        hist_r = cv2.calcHist([img_in], [2], None, [256], [0,256]) # R
        
        hist_b = hist_b / (H * W)
        hist_g = hist_g / (H * W)
        hist_r = hist_r / (H * W)
        
        hist_b_list.append(hist_b)
        hist_g_list.append(hist_g)
        hist_r_list.append(hist_r)
    
    JS_b_list = []
    JS_g_list = []
    JS_r_list = []
    
    for i in range(len(hist_b_list)):
        if i + dilation > len(hist_b_list) - 1:
            break
        hist_b_img1 = hist_b_list[i]
        hist_b_img2 = hist_b_list[i + dilation]     
        JS_b = JS_divergence(hist_b_img1, hist_b_img2)
        JS_b_list.append(JS_b)
        
        hist_g_img1 = hist_g_list[i]
        hist_g_img2 = hist_g_list[i+dilation]     
        JS_g = JS_divergence(hist_g_img1, hist_g_img2)
        JS_g_list.append(JS_g)
        
        hist_r_img1 = hist_r_list[i]
        hist_r_img2 = hist_r_list[i+dilation]     
        JS_r = JS_divergence(hist_r_img1, hist_r_img2)
        JS_r_list.append(JS_r)
        
    return JS_b_list, JS_g_list, JS_r_list


def calc_cdc(vid_folder, dilation=[1, 2, 4], weight=[1/3, 1/3, 1/3]):
    mean_b, mean_g, mean_r = 0, 0, 0
    for d, w in zip(dilation, weight):
        JS_b_list_one, JS_g_list_one, JS_r_list_one = compute_JS_bgr(vid_folder, d)
        mean_b += w * np.mean(JS_b_list_one)
        mean_g += w * np.mean(JS_g_list_one)
        mean_r += w * np.mean(JS_r_list_one)
    
    cdc = np.mean([mean_b, mean_g, mean_r])
    return cdc       
    
    
    
    
    