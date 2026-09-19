from ultralytics import YOLO

MODEL = "croprow_coco_s.pt"
IMG = "test.png"

model = YOLO(MODEL)

print("model.task =", model.task)
print("last layer =", type(model.model.model[-1]).__name__)
print("names =", model.names)

last = model.model.model[-1]
print("nc =", getattr(last, "nc", None))
print("nm =", getattr(last, "nm", None))

results = model(
    IMG,
    imgsz=320,
    conf=0.25,
    task="segment"
)

r = results[0]
r.save(filename="pc_croprow_result.jpg")

print("boxes =", None if r.boxes is None else len(r.boxes))
print("masks =", r.masks)
print("mask shape =", None if r.masks is None else r.masks.data.shape)

if r.masks is not None:
    print("OK: segmentation mask exists")
else:
    print("ERROR: no mask")