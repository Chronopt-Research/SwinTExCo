import torch
import torch.nn as nn
from src.utils import feature_normalize


### START### CONTEXTUAL LOSS ####
class ContextualLoss(nn.Module):
    """
    input is Al, Bl, channel = 1, range ~ [0, 255]
    """

    def __init__(self):
        super(ContextualLoss, self).__init__()
        return None

    def forward(self, X_features, Y_features, h=0.1, feature_centering=True):
        """
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        """
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]

        # to normalized feature vectors
        if feature_centering:
            X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(
                dim=-1
            )
            Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(
                dim=-1
            )
        X_features = feature_normalize(X_features).view(
            batch_size, feature_depth, -1
        )  # batch_size * feature_depth * feature_size^2
        Y_features = feature_normalize(Y_features).view(
            batch_size, feature_depth, -1
        )  # batch_size * feature_depth * feature_size^2

        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - torch.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-5)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = torch.exp((1 - d_norm) / h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)

        # contextual loss per sample
        CX = torch.mean(torch.max(A_ij, dim=1)[0], dim=-1)
        return -torch.log(CX)


class ContextualLoss_forward(nn.Module):
    """
    input is Al, Bl, channel = 1, range ~ [0, 255]
    """

    def __init__(self):
        super(ContextualLoss_forward, self).__init__()
        return None

    def forward(self, X_features, Y_features, h=0.1, feature_centering=True):
        """
        X_features&Y_features are are feature vectors or feature 2d array
        h: bandwidth
        return the per-sample loss
        """
        batch_size = X_features.shape[0]
        feature_depth = X_features.shape[1]

        # to normalized feature vectors
        if feature_centering:
            X_features = X_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(
                dim=-1
            )
            Y_features = Y_features - Y_features.view(batch_size, feature_depth, -1).mean(dim=-1).unsqueeze(dim=-1).unsqueeze(
                dim=-1
            )
        X_features = feature_normalize(X_features).view(
            batch_size, feature_depth, -1
        )  # batch_size * feature_depth * feature_size^2
        Y_features = feature_normalize(Y_features).view(
            batch_size, feature_depth, -1
        )  # batch_size * feature_depth * feature_size^2

        # conine distance = 1 - similarity
        X_features_permute = X_features.permute(0, 2, 1)  # batch_size * feature_size^2 * feature_depth
        d = 1 - torch.matmul(X_features_permute, Y_features)  # batch_size * feature_size^2 * feature_size^2

        # normalized distance: dij_bar
        d_norm = d / (torch.min(d, dim=-1, keepdim=True)[0] + 1e-5)  # batch_size * feature_size^2 * feature_size^2

        # pairwise affinity
        w = torch.exp((1 - d_norm) / h)
        A_ij = w / torch.sum(w, dim=-1, keepdim=True)

        # contextual loss per sample
        CX = torch.mean(torch.max(A_ij, dim=-1)[0], dim=1)
        return -torch.log(CX)


### END### CONTEXTUAL LOSS ####


##########################


def mse_loss_fn(input, target=0):
    return torch.mean((input - target) ** 2)


### START### PERCEPTUAL LOSS ###
def Perceptual_loss(domain_invariant, weight_perceptual):
    instancenorm = nn.InstanceNorm2d(512, affine=False)

    def __call__(A_relu5_1, predict_relu5_1):
        if domain_invariant:
            feat_loss = (
                mse_loss_fn(instancenorm(predict_relu5_1), instancenorm(A_relu5_1.detach())) * weight_perceptual * 1e5 * 0.2
            )
        else:
            feat_loss = mse_loss_fn(predict_relu5_1, A_relu5_1.detach()) * weight_perceptual
        return feat_loss

    return __call__


### END### PERCEPTUAL LOSS ###


def l1_loss_fn(input, target=0):
    return torch.mean(torch.abs(input - target))


### END#################


### START### ADVERSIAL LOSS ###
def generator_loss_fn(real_data_lab, fake_data_lab, discriminator, weight_gan, device):
    if weight_gan > 0:
        y_pred_fake, _ = discriminator(fake_data_lab)
        y_pred_real, _ = discriminator(real_data_lab)

        y = torch.ones_like(y_pred_real)
        generator_loss = (
            (
                torch.mean((y_pred_real - torch.mean(y_pred_fake) + y) ** 2)
                + torch.mean((y_pred_fake - torch.mean(y_pred_real) - y) ** 2)
            )
            / 2
            * weight_gan
        )
        return generator_loss

    return torch.Tensor([0]).to(device)


def discriminator_loss_fn(real_data_lab, fake_data_lab, discriminator):
    y_pred_fake, _ = discriminator(fake_data_lab.detach())
    y_pred_real, _ = discriminator(real_data_lab.detach())

    y = torch.ones_like(y_pred_real)
    discriminator_loss = (
        torch.mean((y_pred_real - torch.mean(y_pred_fake) - y) ** 2)
        + torch.mean((y_pred_fake - torch.mean(y_pred_real) + y) ** 2)
    ) / 2
    return discriminator_loss


### END### ADVERSIAL LOSS #####


def consistent_loss_fn(
    I_current_lab_predict,
    I_last_ab_predict,
    I_current_nonlocal_lab_predict,
    I_last_nonlocal_lab_predict,
    flow_forward,
    mask,
    warping_layer,
    weight_consistent=0.02,
    weight_nonlocal_consistent=0.0,
    device="cuda",
):
    def weighted_mse_loss(input, target, weights):
        out = (input - target) ** 2
        out = out * weights.expand_as(out)
        return out.mean()

    def consistent():
        I_current_lab_predict_warp = warping_layer(I_current_lab_predict, flow_forward)
        I_current_ab_predict_warp = I_current_lab_predict_warp[:, 1:3, :, :]
        consistent_loss = weighted_mse_loss(I_current_ab_predict_warp, I_last_ab_predict, mask) * weight_consistent
        return consistent_loss

    def nonlocal_consistent():
        I_current_nonlocal_lab_predict_warp = warping_layer(I_current_nonlocal_lab_predict, flow_forward)
        nonlocal_consistent_loss = (
            weighted_mse_loss(
                I_current_nonlocal_lab_predict_warp[:, 1:3, :, :],
                I_last_nonlocal_lab_predict[:, 1:3, :, :],
                mask,
            )
            * weight_nonlocal_consistent
        )

        return nonlocal_consistent_loss

    consistent_loss = consistent() if weight_consistent else torch.Tensor([0]).to(device)
    nonlocal_consistent_loss = nonlocal_consistent() if weight_nonlocal_consistent else torch.Tensor([0]).to(device)

    return consistent_loss + nonlocal_consistent_loss


### END### CONSISTENCY LOSS #####


### START### SMOOTHNESS LOSS ###
def smoothness_loss_fn(
    I_current_l,
    I_current_lab,
    I_current_ab_predict,
    A_relu2_1,
    weighted_layer_color,
    nonlocal_weighted_layer,
    weight_smoothness=5.0,
    weight_nonlocal_smoothness=0.0,
    device="cuda",
):
    def smoothness(scale_factor=1.0):
        I_current_lab_predict = torch.cat((I_current_l, I_current_ab_predict), dim=1)
        IA_ab_weighed = weighted_layer_color(
            I_current_lab,
            I_current_lab_predict,
            patch_size=3,
            alpha=10,
            scale_factor=scale_factor,
        )
        smoothness_loss = (
            mse_loss_fn(
                nn.functional.interpolate(I_current_ab_predict, scale_factor=scale_factor),
                IA_ab_weighed,
            )
            * weight_smoothness
        )

        return smoothness_loss

    def nonlocal_smoothness(scale_factor=0.25, alpha_nonlocal_smoothness=0.5):
        nonlocal_smooth_feature = feature_normalize(A_relu2_1)
        I_current_lab_predict = torch.cat((I_current_l, I_current_ab_predict), dim=1)
        I_current_ab_weighted_nonlocal = nonlocal_weighted_layer(
            I_current_lab_predict,
            nonlocal_smooth_feature.detach(),
            patch_size=3,
            alpha=alpha_nonlocal_smoothness,
            scale_factor=scale_factor,
        )
        nonlocal_smoothness_loss = (
            mse_loss_fn(
                nn.functional.interpolate(I_current_ab_predict, scale_factor=scale_factor),
                I_current_ab_weighted_nonlocal,
            )
            * weight_nonlocal_smoothness
        )
        return nonlocal_smoothness_loss

    smoothness_loss = smoothness() if weight_smoothness else torch.Tensor([0]).to(device)
    nonlocal_smoothness_loss = nonlocal_smoothness() if weight_nonlocal_smoothness else torch.Tensor([0]).to(device)

    return smoothness_loss + nonlocal_smoothness_loss


### END### SMOOTHNESS LOSS #####
