from PIL import Image
from analysis.face_crop import crop_face
from analysis.recommender import recommend_lipsticks
from database.schema import initialize_database
from analysis.lip_filter import apply_lipstick

# 0) Инициализация БД
initialize_database()

# 1) Извлечение лица
img_path = "подопытный_2.jpg"
img      = Image.open(img_path).convert("RGB")
face_img = crop_face(img)

# 2) Рекомендации
results = recommend_lipsticks(face_img)

# 3) Вывод
print("🎯 Топ‑3 подходящих оттенков:")
for r in results:
    print(f"💄 {r['name']} ({r['hex']}) — {r['brand']} | score: {r['score']}")

# Применяем топ-1 оттенок
top_shade = results[0]  # Берем первый (лучший) результат из списка рекомендаций
colored_img = apply_lipstick(img, hex_color=top_shade["hex"])

# Показать результат
colored_img.show()
