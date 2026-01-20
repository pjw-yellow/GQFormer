import os
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog


def load_sem_seg_dataset(image_dir, mask_dir):
    """
    Detectron2 语义分割数据加载函数
    """
    dataset_dicts = []

    for fname in sorted(os.listdir(image_dir)):
        if not (fname.endswith(".jpg") or fname.endswith(".png")):
            continue

        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname.replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        if image is None:
            continue

        height, width = image.shape[:2]

        record = {
            "file_name": img_path,
            "image_id": fname,
            "height": height,
            "width": width,
            "sem_seg_file_name": mask_path,
        }
        dataset_dicts.append(record)

    return dataset_dicts


def register_my_dataset():
    """
    Registration of the RailSem19 Semantic Segmentation Dataset
    """
    root_img = "/pengjiawei/DataSet/myr19pros/img"
    root_mask = "/pengjiawei/DataSet/myr19pros/ann"

    # ================= 类别（顺序 = 类别 ID） =================
    stuff_classes = [
        "road",            # 0
        "sidewalk",        # 1
        "construction",    # 2
        "tram-track",      # 3
        "fence",           # 4
        "pole",            # 5
        "traffic-light",   # 6
        "traffic-sign",    # 7
        "vegetation",      # 8
        "terrain",         # 9
        "sky",             # 10
        "human",           # 11
        "rail-track",      # 12
        "car",             # 13
        "truck",           # 14
        "trackbed",        # 15
        "on-rails",        # 16
        "rail-raised",     # 17
        "rail-embedded",   # 18
    ]

    # ================= 颜色（严格按你的 category_mapping） =================
    stuff_colors = [
        [128,  64, 128],  # 0
        [244,  35, 232],  # 1
        [ 70,  70,  70],  # 2
        [192,   0, 128],  # 3
        [190, 153, 153],  # 4
        [153, 153, 153],  # 5
        [250, 170,  30],  # 6
        [220, 220,   0],  # 7
        [107, 142,  35],  # 8
        [152, 251, 152],  # 9
        [ 70, 130, 180],  # 10
        [220,  20,  60],  # 11
        [230, 149, 139],  # 12
        [  0,   0, 142],  # 13
        [  0,   0,  70],  # 14
        [ 90,  40,  40],  # 15
        [  0,  80, 100],  # 16
        [  0, 253, 253],  # 17
        [  0,  68,  62],  # 18
    ]

    # ================= 训练集 =================
    DatasetCatalog.register(
        "my_rs19pros_train",
        lambda: load_sem_seg_dataset(
            os.path.join(root_img, "train"),
            os.path.join(root_mask, "train"),
        )
    )

    MetadataCatalog.get("my_rs19pros_train").set(
        stuff_classes=stuff_classes,
        stuff_colors=stuff_colors,
        evaluator_type="sem_seg",
        ignore_label=255,
    )

    # ================= 验证集 =================
    DatasetCatalog.register(
        "my_rs19pros_val",
        lambda: load_sem_seg_dataset(
            os.path.join(root_img, "val"),
            os.path.join(root_mask, "val"),
        )
    )

    MetadataCatalog.get("my_rs19pros_val").set(
        stuff_classes=stuff_classes,
        stuff_colors=stuff_colors,
        evaluator_type="sem_seg",
        ignore_label=255,
    )


# ⭐ 关键：import 即注册
register_my_dataset()
