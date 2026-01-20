# Copyright (c) Facebook, Inc. and its affiliates.
# Layer-wise semantic mask visualization for Mask2Former (per-class, no Grad-CAM)
# 修正版 v2：
# - 使用与 DefaultPredictor 一致的预处理与后处理流程
# - 修复 CPU/GPU 张量冲突 (pixel_mean/std)
# - 支持 --classes -1 输出所有类别
# - 每层输出“pre / layer1.. / final”语义掩码，可与官方推理结果对齐

import argparse
import glob
import multiprocessing as mp
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import cv2
import tqdm
import torch
import numpy as np
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.structures import ImageList
from detectron2.modeling.postprocessing import sem_seg_postprocess
import detectron2.data.transforms as T

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Mask2Former layer-wise per-class visualization (fixed)")
    parser.add_argument("--config-file", required=True, help="path to config yaml")
    parser.add_argument("--input", nargs="+", required=True, help="images or a glob pattern")
    parser.add_argument("--output", required=True, help="directory to save results")
    parser.add_argument("--target-image", default=None, help="analyze only one image (filename prefix)")
    parser.add_argument("--classes", type=str, default="-1",
                        help="-1 for all classes, or comma-separated list like '0,1,12'")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    return parser


def parse_classes_arg(arg_str):
    arg_str = arg_str.strip()
    if arg_str == "" or arg_str == "-1":
        return [-1]
    return [int(x) for x in arg_str.split(",") if x.strip() != ""]


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger()
    cfg = setup_cfg(args)
    os.makedirs(args.output, exist_ok=True)

    # === 模型 ===
    demo = VisualizationDemo(cfg)
    model = demo.predictor.model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === 元信息（用于类别命名） ===
    ds_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else cfg.DATASETS.TRAIN[0]
    meta = MetadataCatalog.get(ds_name)
    class_names = getattr(meta, "stuff_classes", None) or getattr(meta, "thing_classes", None)
    if class_names is None:
        class_names = []

    # === 输入解析 ===
    if len(args.input) == 1 and ("*" in args.input[0] or "?" in args.input[0]):
        inputs = glob.glob(os.path.expanduser(args.input[0]))
    else:
        inputs = [os.path.expanduser(x) for x in args.input]
    if args.target_image:
        inputs = [p for p in inputs if os.path.basename(p).startswith(args.target_image)]
    assert len(inputs), "No matching input images found."

    target_classes_arg = parse_classes_arg(args.classes)

    # === 使用 DefaultPredictor 的增强与归一化流程 ===
    aug = demo.predictor.aug
    pixel_mean = model.pixel_mean.to(device)
    pixel_std = model.pixel_std.to(device)
    size_div = model.backbone.size_divisibility

    for path in tqdm.tqdm(inputs):
        # 读取原图
        img = read_image(path, format="BGR")  # HWC
        orig_h, orig_w = img.shape[:2]

        # === 与 DefaultPredictor 一致的预处理 ===
        aug_input = T.AugInput(img)
        _ = aug(aug_input)
        img_resized = aug_input.image

        # 转 tensor 并移动到 GPU，再归一化
        image = torch.as_tensor(img_resized.astype("float32").transpose(2, 0, 1)).to(device)
        image = (image - pixel_mean) / pixel_std

        # padding 对齐
        images = ImageList.from_tensors([image], size_divisibility=size_div)
        proc_h, proc_w = images.tensor.shape[-2:]

        # === 前向推理 ===
        with torch.no_grad():
            features = model.backbone(images.tensor)
            mask_features, transformer_out, multi_scale_feats = model.sem_seg_head.pixel_decoder.forward_features(features)
            pred = model.sem_seg_head.predictor(multi_scale_feats, mask_features)

        # === 提取各层输出 ===
        layer_heads = []
        layer_names = []
        if "aux_outputs" in pred and isinstance(pred["aux_outputs"], list) and len(pred["aux_outputs"]) > 0:
            layer_heads.append((pred["aux_outputs"][0]["pred_logits"], pred["aux_outputs"][0]["pred_masks"]))
            layer_names.append("pre")
            for li in range(1, len(pred["aux_outputs"])):
                layer_heads.append((pred["aux_outputs"][li]["pred_logits"], pred["aux_outputs"][li]["pred_masks"]))
                layer_names.append(f"layer{li}")
        layer_heads.append((pred["pred_logits"], pred["pred_masks"]))
        layer_names.append("final")

        # === 输出目录 ===
        base = os.path.splitext(os.path.basename(path))[0]
        save_dir = os.path.join(args.output, f"{base}_layerwise_perclass")
        os.makedirs(save_dir, exist_ok=True)

        # === 每层语义聚合 ===
        for lname, (logits, masks) in zip(layer_names, layer_heads):
            # logits: [B,Q,C+1], masks: [B,Q,Hm,Wm]
            cls_prob = logits.softmax(dim=-1)[..., :-1]   # [B,Q,C]
            mask_prob = masks.sigmoid()                   # [B,Q,Hm,Wm]

            # 语义聚合： per-class 概率图
            sem_prob = torch.einsum("bqc,bqhw->bchw", cls_prob, mask_prob)  # [B,C,Hm,Wm]

            # 上采样到增强后尺寸
            sem_prob_up = F.interpolate(sem_prob, size=(proc_h, proc_w), mode="bilinear", align_corners=False)
            # 用官方后处理恢复原图比例
            sem_prob_orig = sem_seg_postprocess(sem_prob_up[0], (proc_h, proc_w), orig_h, orig_w)  # [C,H,W]

            # 类别预测
            sem_pred_class = sem_prob_orig.argmax(dim=0).cpu().numpy()
            C = cls_prob.shape[-1]

            if len(target_classes_arg) == 1 and target_classes_arg[0] == -1:
                target_class_ids = list(range(C))
            else:
                target_class_ids = [c for c in target_classes_arg if 0 <= c < C]

            # === 可视化每类 ===
            for cid in target_class_ids:
                mask_bin = (sem_pred_class == cid).astype(np.uint8)
                if mask_bin.sum() == 0:
                    continue
                overlay = img.copy()
                overlay[mask_bin == 1] = [0, 0, 255]  # BGR 红色

                if cid < len(class_names):
                    cname = class_names[cid].replace(" ", "_")
                    out_name = f"{base}_{lname}_class{cid:02d}_{cname}.png"
                else:
                    out_name = f"{base}_{lname}_class{cid:02d}.png"

                cv2.imwrite(os.path.join(save_dir, out_name), overlay)

        print(f"[Saved] layers={len(layer_names)} at {save_dir}")

    print("[Done] Per-class, layer-wise visualization (fixed) complete.")
