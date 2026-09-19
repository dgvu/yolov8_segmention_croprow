from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

IMAGE = "test.png"
MODELS = {
    "pt": "croprow_coco_s.pt",
    "ncnn": "croprow_coco_s_ncnn_model"
}

OUT = Path("compare_mask_debug")
OUT.mkdir(exist_ok=True)

for tag, model_path in MODELS.items():
    print("\n==============================")
    print("Testing:", tag, model_path)
    print("==============================")

    if tag == "ncnn":
        model = YOLO(model_path, task="segment")
    else:
        model = YOLO(model_path)

    results = model(
        IMAGE,
        imgsz=640,
        conf=0.25,
        task="segment",
        retina_masks=True
    )

    r = results[0]
    r.save(filename=str(OUT / f"{tag}_overlay.jpg"))

    print("boxes:", 0 if r.boxes is None else len(r.boxes))
    print("masks is None:", r.masks is None)

    if r.masks is None:
        continue

    masks = r.masks.data.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()

    print("mask shape:", masks.shape)

    for i, mask in enumerate(masks):
        mask_bin = (mask > 0.5).astype(np.uint8)
        area = int(mask_bin.sum())
        area_ratio = area / mask_bin.size
        max_val = float(mask.max())
        min_val = float(mask.min())

        print(
            f"obj {i}: class={classes[i]}, score={scores[i]:.3f}, "
            f"mask_area={area}, area_ratio={area_ratio:.6f}, "
            f"mask_min={min_val:.4f}, mask_max={max_val:.4f}"
        )

        cv2.imwrite(str(OUT / f"{tag}_mask_{i}.png"), mask_bin * 255)
        np.save(OUT / f"{tag}_mask_{i}.npy", mask_bin)
