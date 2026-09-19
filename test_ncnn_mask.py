from ultralytics import YOLO

model = YOLO("croprow_coco_s_ncnn_model")

results = model(
    "test.png",
    imgsz=320,
    conf=0.25,
    task="segment"
)

r = results[0]
r.save(filename="ncnn_pc_croprow_result.jpg")

print("Boxes:", len(r.boxes))
print("Mask shape:", r.masks.data.shape if r.masks is not None else None)
print("Names:", r.names)
