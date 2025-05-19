from transformers import pipeline
import numpy as np
from PIL import Image
import cv2

seg_parser = pipeline("image-segmentation", model="jonathandinu/face-parsing")

def crop_face(img: Image.Image) -> Image.Image:
    masks = seg_parser(img)
    skin_mask = next((m["mask"] for m in masks if m["label"].lower() == "skin"), None)

    if skin_mask is not None:
        mask_np = np.array(skin_mask, dtype=bool)
        ys, xs = np.where(mask_np)
        if len(xs) == 0 or len(ys) == 0:
            return _fallback_opencv(img)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        return img.crop((x0, y0, x1, y1))

    return _fallback_opencv(img)

def _fallback_opencv(img: Image.Image) -> Image.Image:
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        raise RuntimeError("Не удалось найти лицо ни через skin-маску, ни через OpenCV")
    x, y, w, h = faces[0]
    return img.crop((x, y, x + w, y + h))
