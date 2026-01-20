# Copyright (c) Facebook, Inc. and its affiliates.
# Simplified demo: visualize query masks for each decoder layer (no Grad-CAM)

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
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
#输出每层decoder的query注意力分布图

def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Mask2Former decoder query visualization")
    parser.add_argument("--config-file", required=True, help="path to config yaml")
    parser.add_argument("--input", nargs="+", required=True, help="images or a glob pattern")
    parser.add_argument("--output", required=True, help="directory to save results")
    parser.add_argument("--target-image", default=None, help="analyze only one image (filename prefix)")
    parser.add_argument("--devide", type=int, default=5,
                        help="transparency level for overlay, 0=only original, 1=only heatmap")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger()
    cfg = setup_cfg(args)
    os.makedirs(args.output, exist_ok=True)

    # === 初始化模型 ===
    demo = VisualizationDemo(cfg)
    model = demo.predictor.model
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === 数据集元信息（不直接用，只为兼容） ===
    ds_name = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else cfg.DATASETS.TRAIN[0]
    meta = MetadataCatalog.get(ds_name)

    # === 解析输入文件 ===
    inputs = []
    if len(args.input) == 1 and ("*" in args.input[0] or "?" in args.input[0]):
        inputs = glob.glob(os.path.expanduser(args.input[0]))
    else:
        inputs = [os.path.expanduser(x) for x in args.input]

    if args.target_image:
        inputs = [p for p in inputs if os.path.basename(p).startswith(args.target_image)]
    assert len(inputs), "No matching input images found."

    # === 主循环 ===
    for path in tqdm.tqdm(inputs):
        img = read_image(path, format="BGR")
        H, W = img.shape[:2]
        img_t = torch.as_tensor(img.astype("float32").transpose(2, 0, 1)).unsqueeze(0).to(device)

        # 前向推理
        with torch.no_grad():
            features = model.backbone(img_t)
            mask_features, transformer_out, multi_scale_feats = model.sem_seg_head.pixel_decoder.forward_features(features)
            pred = model.sem_seg_head.predictor(multi_scale_feats, mask_features)

        # 检查 pixel_decoder 输出中是否包含 all_layer_masks
        if "all_layer_masks" not in pred:
            raise RuntimeError(
                "pixel_decoder.py 未修改，缺少 all_layer_masks。请确认在 forward() 中添加：\n"
                "out['all_layer_masks'] = predictions_mask"
            )

        all_masks = pred["all_layer_masks"]
        num_layers = len(all_masks)
        num_queries = all_masks[0].shape[1]
        chosen_queries = [q for q in range(num_queries) if q % args.devide == 0]

        base = os.path.splitext(os.path.basename(path))[0]
        save_dir = os.path.join(args.output, f"{base}_layerwise")
        os.makedirs(save_dir, exist_ok=True)

        for q in chosen_queries:
            for i, mask in enumerate(all_masks):
                # === 1. 取出该层该 query 的 mask 并 Sigmoid ===
                m = torch.sigmoid(mask[0, q]).cpu().numpy()

                # === 2. 二值化（0.5阈值） ===
                m_bin = (m > 0.5).astype(np.uint8)  # 0或1的二值掩码

                # === 3. 调整到原图大小 ===
                m_bin = cv2.resize(m_bin, (W, H), interpolation=cv2.INTER_NEAREST)

                # === 4. 在原图上绘制红色区域 ===
                overlay = img.copy()
                overlay[m_bin == 1] = [0, 0, 255]  # BGR: 红色

                # === 5. 可选：把红色区域与原图混合一点透明度 ===
                # 如果想保持一点原图信息，可以用这行代替上面那行
                # overlay[m_bin == 1] = 0.5 * img[m_bin == 1] + 0.5 * np.array([0,0,255])

                # === 6. 保存结果 ===
                cv2.imwrite(os.path.join(save_dir, f"{base}_q{q:03d}_decoder{i}.png"), overlay)

        print(f"[Saved] {len(chosen_queries)} queries × {num_layers} layers at {save_dir}")

    print("[Done] Query-wise decoder visualization complete.")
