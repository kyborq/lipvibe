# analysis/recommender.py

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

from analysis.embedding import get_image_embedding
from analysis.palette import get_all_shades

SWATCH_SIZE = (128, 128)
TOP_K       = 3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

def recommend_lipsticks(face_image: Image.Image):
    """
    Принимает изображение лица, возвращает TOP‑K оттенков помады из базы
    """
    all_shades = get_all_shades()

    swatch_images = [
        Image.new("RGB", SWATCH_SIZE, shade["hex"]) for shade in all_shades
    ]

    # Эмбеддинги
    em_face = get_image_embedding(face_image, processor, model)
    em_sw   = get_image_embedding(swatch_images, processor, model)

    # Сходство
    sims = torch.nn.functional.cosine_similarity(em_face, em_sw)
    
    # Получаем уникальные топ-K результатов
    unique_sims = []
    unique_idxs = []
    seen_idxs = set()
    
    for score, idx in zip(sims.tolist(), range(len(sims))):
        if idx not in seen_idxs:
            unique_sims.append(score)
            unique_idxs.append(idx)
            seen_idxs.add(idx)
            if len(unique_sims) >= TOP_K:
                break
    
    # Результат
    recommendations = []
    for score, idx in zip(unique_sims, unique_idxs):
        shade = all_shades[idx].copy()  # Создаем копию, чтобы не модифицировать оригинал
        shade["score"] = round(score, 3)
        recommendations.append(shade)
    return recommendations
