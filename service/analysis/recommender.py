# analysis/recommender.py

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

from analysis.embedding import get_image_embedding
from analysis.palette import get_all_shades

SWATCH_SIZE = (128, 128)
TOP_K       = 3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)

def get_dominant_colors(image: Image.Image, n_colors=3):
    """Извлекает доминирующие цвета из изображения"""
    # Преобразуем изображение в массив
    img_array = np.array(image)
    
    # Изменяем форму для кластеризации
    pixels = img_array.reshape(-1, 3)
    
    # Применяем K-means для получения доминирующих цветов
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    # Получаем центры кластеров (доминирующие цвета)
    colors = kmeans.cluster_centers_.astype(int)
    
    # Сортируем цвета по яркости
    brightness = np.mean(colors, axis=1)
    sorted_indices = np.argsort(brightness)
    return colors[sorted_indices]

def analyze_features(face_image: Image.Image):
    """Анализирует характеристики лица"""
    # Получаем доминирующие цвета
    colors = get_dominant_colors(face_image, n_colors=5)
    
    # Определяем цвет кожи (самый светлый цвет)
    skin_color = colors[0]
    
    # Определяем цвет волос (самый темный цвет)
    hair_color = colors[-1]
    
    # Анализируем оттенки кожи
    skin_r, skin_g, skin_b = skin_color
    
    # Определяем теплоту оттенка
    warmth = (skin_r - skin_b) / 255.0  # Положительное значение = теплый оттенок
    
    # Определяем контрастность
    contrast = np.std(skin_color) / 255.0
    
    # Определяем яркость
    brightness = np.mean(skin_color) / 255.0
    
    # Определяем сезонный цветотип
    if warmth > 0.1:  # Теплые оттенки
        if contrast > 0.15 and brightness < 0.7:
            season = "Autumn"  # Теплый, контрастный, не очень светлый
        else:
            season = "Spring"  # Теплый, менее контрастный, светлый
    else:  # Холодные оттенки
        if contrast > 0.15 and brightness < 0.7:
            season = "Winter"  # Холодный, контрастный, не очень светлый
        else:
            season = "Summer"  # Холодный, менее контрастный, светлый
    
    return {
        "skin_color": skin_color.tolist(),
        "hair_color": hair_color.tolist(),
        "season": season,
        "analysis": {
            "warmth": float(warmth),
            "contrast": float(contrast),
            "brightness": float(brightness)
        }
    }

def recommend_lipsticks(face_image: Image.Image):
    """
    Принимает изображение лица, возвращает TOP‑K оттенков помады из базы
    """
    # Анализируем характеристики лица
    features = analyze_features(face_image)
    
    # Получаем все оттенки
    all_shades = get_all_shades()
    
    # Фильтруем оттенки по сезону
    seasonal_shades = [s for s in all_shades if s["season"] == features["season"]]
    if not seasonal_shades:
        seasonal_shades = all_shades  # Если нет оттенков для сезона, используем все
    
    # Создаем патчи для сравнения
    swatch_images = [
        Image.new("RGB", SWATCH_SIZE, shade["hex"]) for shade in seasonal_shades
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
        shade = seasonal_shades[idx].copy()
        shade["score"] = round(score, 3)
        recommendations.append(shade)
    
    return {
        "recommendations": recommendations,
        "analysis": features
    }
