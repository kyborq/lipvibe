from PIL import Image, ImageDraw
import numpy as np
from transformers import pipeline

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def apply_lipstick(image: Image.Image, hex_color: str, alpha: float = 0.3) -> Image.Image:
    """
    Наносит цвет на губы с заданной прозрачностью.
    :param image: PIL.Image.Image — исходное изображение
    :param hex_color: str — цвет в HEX (#RRGGBB)
    :param alpha: float — степень окрашивания (0.0–1.0)
    :return: PIL.Image.Image — изображение с окрашенными губами
    """
    # Пробуем первую модель
    try:
        seg_model = pipeline("image-segmentation", model="jonathandinu/face-parsing")
        masks = seg_model(image)
        
        # Выводим все найденные метки для диагностики
        print("Найденные сегменты:", [m["label"] for m in masks])
        
        # Ищем маски верхней и нижней губ
        upper_lip = next((m for m in masks if m["label"].lower() == "u_lip"), None)
        lower_lip = next((m for m in masks if m["label"].lower() == "l_lip"), None)
        
        if upper_lip is None and lower_lip is None:
            print("[!] Маски губ не найдены в первой модели, пробуем альтернативную...")
            # Пробуем альтернативную модель
            alt_model = pipeline("image-segmentation", model="mattmdjaga/segformer_b2_clothes")
            alt_masks = alt_model(image)
            print("Альтернативные сегменты:", [m["label"] for m in alt_masks])
            
            lip_mask_data = next((m for m in alt_masks if m["label"].lower() in {"lips", "lip", "mouth"}), None)
            
            if lip_mask_data is None:
                print("[!] Маска губ не найдена ни в одной модели.")
                return image
        else:
            # Объединяем маски верхней и нижней губ
            if upper_lip and lower_lip:
                # Если есть обе маски, объединяем их
                upper_mask = np.array(upper_lip["mask"].convert("L")) / 255.0
                lower_mask = np.array(lower_lip["mask"].convert("L")) / 255.0
                combined_mask = np.maximum(upper_mask, lower_mask)
                lip_mask_data = {"mask": Image.fromarray((combined_mask * 255).astype(np.uint8))}
            else:
                # Если есть только одна маска, используем её
                lip_mask_data = upper_lip or lower_lip
    except Exception as e:
        print(f"[!] Ошибка при сегментации: {str(e)}")
        return image

    mask_img = lip_mask_data["mask"].convert("L")  # в градациях серого
    lip_mask_np = np.array(mask_img) / 255.0       # в диапазоне [0, 1]

    rgb = np.array(image).astype(np.float32)
    lip_color = np.array(hex_to_rgb(hex_color), dtype=np.float32)

    # Создать маску в формате (H, W, 3)
    mask_3d = np.stack([lip_mask_np]*3, axis=-1)

    # Альфа смешивание
    blended = rgb * (1 - mask_3d * alpha) + lip_color * (mask_3d * alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)
