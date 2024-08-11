import torch
from src.utils import *
from src.models.vit.vit import FeatureTransform


def warp_color(
    IA_l,
    IB_lab,
    features_B,
    embed_net,
    nonlocal_net,
    temperature=0.01,
):
    IA_rgb_from_gray = gray2rgb_batch(IA_l)

    with torch.no_grad():
        A_feat0, A_feat1, A_feat2, A_feat3 = embed_net(IA_rgb_from_gray)
        B_feat0, B_feat1, B_feat2, B_feat3 = features_B

    A_feat0 = feature_normalize(A_feat0)
    A_feat1 = feature_normalize(A_feat1)
    A_feat2 = feature_normalize(A_feat2)
    A_feat3 = feature_normalize(A_feat3)

    B_feat0 = feature_normalize(B_feat0)
    B_feat1 = feature_normalize(B_feat1)
    B_feat2 = feature_normalize(B_feat2)
    B_feat3 = feature_normalize(B_feat3)

    return nonlocal_net(
        IB_lab,
        A_feat0,
        A_feat1,
        A_feat2,
        A_feat3,
        B_feat0,
        B_feat1,
        B_feat2,
        B_feat3,
        temperature=temperature,
    )


def frame_colorization(
    IA_l,
    IB_lab,
    IA_last_lab,
    features_B,
    embed_net,
    nonlocal_net,
    colornet,
    joint_training=True,
    luminance_noise=0,
    temperature=0.01,
):
    if luminance_noise:
        IA_l = IA_l + torch.randn_like(IA_l, requires_grad=False) * luminance_noise

    with torch.autograd.set_grad_enabled(joint_training):
        nonlocal_BA_lab, similarity_map = warp_color(
            IA_l,
            IB_lab,
            features_B,
            embed_net,
            nonlocal_net,
            temperature=temperature,
        )
        nonlocal_BA_ab = nonlocal_BA_lab[:, 1:3, :, :]
        IA_ab_predict = colornet(
            torch.cat(
                (IA_l, nonlocal_BA_ab, similarity_map, IA_last_lab),
                dim=1,
            )
        )

    return IA_ab_predict, nonlocal_BA_lab