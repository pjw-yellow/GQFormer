import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

def register_ade20k_150(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for tag, split in [("train", "training"), ("val", "validation")]:
        img = os.path.join(root, "images", "ADE", split)
        gt  = os.path.join(root, "annotations_detectron2", split)
        name = f"ade20k_150_sem_seg_{tag}"
        DatasetCatalog.register(
            name, lambda x=img, y=gt: load_sem_seg(image_root=x, gt_root=y, image_ext="jpg", gt_ext="png")
        )
        MetadataCatalog.get(name).set(image_root=img, sem_seg_root=gt,
                                      evaluator_type="sem_seg", ignore_label=255)
