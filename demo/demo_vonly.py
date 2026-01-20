# Copyright (c) Facebook, Inc. and its affiliates.
# Grad-CAM for Mask2Former semantic logits with class-id alignment (only this file changed)

import argparse
import glob
import multiprocessing as mp
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import cv2
import numpy as np
import tqdm
import torch
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# 输出CAM梯度图
def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Mask2Former demo with Grad-CAM (class-aligned)")
    parser.add_argument("--config-file", required=True, help="path to config")
    parser.add_argument("--input", nargs="+", required=True, help="images or a glob pattern")
    parser.add_argument("--output", required=True, help="dir to save results")
    # 允许你传“类名”或“原始ID”或“连续ID”
    parser.add_argument("--target-class", required=True,
                        help="class spec: name (e.g. 'rail-track') or dataset id (int) or contiguous id (c:int)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="transparency level for overlay, 0=only original, 1=only heatmap")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    return parser


def resolve_target_contiguous_id(target_spec, metadata, num_channels):
    """
    将用户输入（类名 / 原始ID / 连续ID）映射到模型输出通道的 contiguous id。
    - 类名：在 metadata.stuff_classes（或 thing_classes）里找下标
    - 原始ID：用 metadata.stuff_dataset_id_to_contiguous_id（或 thing_*）做映射
    - 连续ID：以 c:前缀传入，如 c:11，直接使用 11
    """
    # 优先使用 stuff 类（语义分割）
    names = getattr(metadata, "stuff_classes", None)
    dsid2cid = getattr(metadata, "stuff_dataset_id_to_contiguous_id", None)
    if names is None:  # 回退（不太常见）
        names = getattr(metadata, "thing_classes", None)
        dsid2cid = getattr(metadata, "thing_dataset_id_to_contiguous_id", None)

    # 连续ID，格式 c:11
    if isinstance(target_spec, str) and target_spec.startswith("c:"):
        cid = int(target_spec[2:])
        if 0 <= cid < num_channels:
            return cid
        raise ValueError(f"contiguous id {cid} out of range [0,{num_channels-1}]")

    # 纯数字：尝试按“原始ID”解释；否则当作连续ID兜底
    if isinstance(target_spec, str) and target_spec.isdigit():
        dsid = int(target_spec)
        if isinstance(dsid2cid, dict) and dsid in dsid2cid:
            return dsid2cid[dsid]
        # 兜底：当作连续ID使用（当你确认传的就是连续ID时）
        if 0 <= dsid < num_channels:
            return dsid
        raise ValueError(
            f"'{target_spec}' 既不在 stuff/thing_dataset_id_to_contiguous_id 中，也不在 [0,{num_channels-1}] 内；"
            f"请改用类名或以 'c:<cid>' 传连续ID"
        )

    # 非数字：按“类名”解析
    if isinstance(names, (list, tuple)) and isinstance(target_spec, str):
        if target_spec in names:
            return names.index(target_spec)
        else:
            # 宽松匹配（忽略大小写、空白）
            norm = [n.lower().replace(" ", "").replace("-", "").replace("_", "") for n in names]
            key = target_spec.lower().replace(" ", "").replace("-", "").replace("_", "")
            if key in norm:
                return norm.index(key)

    raise ValueError(
        f"无法解析 target-class='{target_spec}'。建议使用：\n"
        f"- 类名（如 'rail-track'，以你数据注册时的名字为准）\n"
        f"- 原始ID（纯数字，代码会用 dataset_id_to_contiguous_id 做映射）\n"
        f"- 连续ID（加前缀 'c:'，如 'c:11'）"
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger()
    cfg = setup_cfg(args)
    os.makedirs(args.output, exist_ok=True)

    # 初始化 predictor / model
    demo = VisualizationDemo(cfg)
    model = demo.predictor.model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 读取元数据（用 TEST 的第一个数据集）
    ds_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else cfg.DATASETS.TRAIN[0]
    meta = MetadataCatalog.get(ds_name)

    # 注册 Grad-CAM hook 到 pixel decoder 的 mask_features（与语义 logits 直接相关）
    acts_holder, grads_holder = {}, {}

    def fw_hook(m, i, o):
        acts_holder["acts"] = o  # [B, Cmask, Hf, Wf]
    def bw_hook(m, gin, gout):
        grads_holder["grads"] = gout[0]  # same shape as acts

    handle_fw = model.sem_seg_head.pixel_decoder.mask_features.register_forward_hook(fw_hook)
    handle_bw = model.sem_seg_head.pixel_decoder.mask_features.register_full_backward_hook(bw_hook)

    # 处理输入列表/通配符
    inputs = []
    if len(args.input) == 1 and ("*" in args.input[0] or "?" in args.input[0]):
        inputs = glob.glob(os.path.expanduser(args.input[0]))
    else:
        inputs = [os.path.expanduser(x) for x in args.input]
    assert len(inputs), "No input images found."

    for path in tqdm.tqdm(inputs):
        img = read_image(path, format="BGR")  # HWC, uint8
        H, W = img.shape[:2]
        img_t = torch.as_tensor(img.astype("float32").transpose(2, 0, 1)) / 255.0  # 1x3xHxW
        img_t = img_t.unsqueeze(0).to(device)
        img_t.requires_grad_(True)

        # ===== 手动走一遍：backbone -> pixel decoder -> transformer decoder =====
        with torch.enable_grad():
            features = model.backbone(img_t)  # dict(res2..res5)
            mask_features, transformer_out, multi_scale_feats = model.sem_seg_head.pixel_decoder.forward_features(features)
            # decoder 预测（得到 queries 的类别分数 & mask）
            pred = model.sem_seg_head.predictor(multi_scale_feats, mask_features)  # dict with pred_logits, pred_masks

        # 用官方聚合方式得到每类语义 logits: [B,C,H',W']
        # 注意 pred_logits 的最后一维包含 no-object 类，需去掉
        cls_prob = pred["pred_logits"].softmax(dim=-1)[..., :-1]          # [B,Q,C]
        masks_q = pred["pred_masks"]                                       # [B,Q,H',W']
        sem_logits = torch.einsum("bqc,bqhw->bchw", cls_prob, masks_q)     # [B,C,H',W']

        # 解析 contiguous id（输出通道顺序）——支持类名/原始ID/连续ID
        C = sem_logits.shape[1]
        try:
            target_cid = resolve_target_contiguous_id(args.target_class, meta, C)
        except Exception as e:
            # 打印一下可用的类名帮助调试
            names = getattr(meta, "stuff_classes", None) or getattr(meta, "thing_classes", [])
            print(f"\n[Class Mapping Help] classes (contiguous order):")
            for i, n in enumerate(names):
                print(f"  cid={i:2d}  name={n}")
            raise

        # 选择该类的语义 logit，做 Grad-CAM
        score = sem_logits[:, target_cid].mean()  # scalar
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        acts = acts_holder.get("acts", None)    # [B,Cmask,Hf,Wf]
        grads = grads_holder.get("grads", None)
        if (acts is None) or (grads is None):
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients. "
                               "Check that mask_features is used in forward path.")

        # Grad-CAM 权重 & CAM
        weights = grads.mean(dim=(2, 3), keepdim=True)                     # [B,Cmask,1,1]
        cam = F.relu((acts * weights).sum(dim=1, keepdim=True))            # [B,1,Hf,Wf]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam_map = (cam[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(cam_map, cv2.COLORMAP_JET)
        overlay = ((1 - args.alpha) * img + args.alpha * heatmap).clip(0, 255).astype(np.uint8)

        # 输出文件名
        base = os.path.basename(path)
        root, _ = os.path.splitext(base)
        out_path = os.path.join(args.output, f"{root}_gradcam_{args.target_class}.png")
        cv2.imwrite(out_path, overlay)

    handle_fw.remove()
    handle_bw.remove()
