from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
from typing import List, Dict
import uuid

from analysis.face_crop import crop_face
from analysis.recommender import recommend_lipsticks
from analysis.lip_filter import apply_lipstick
from database.schema import initialize_database

app = FastAPI(title="LipVibe API", description="API for lipstick recommendations and virtual try-on")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321"],  # Astro dev server default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создаем директорию для временных файлов
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске сервера"""
    initialize_database()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Анализирует загруженное изображение и возвращает рекомендации помад
    """
    try:
        # Читаем загруженное изображение
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Генерируем уникальные имена файлов
        base_name = str(uuid.uuid4())
        face_path = os.path.join(TEMP_DIR, f"{base_name}_face.jpg")
        result_path = os.path.join(TEMP_DIR, f"{base_name}_result.jpg")
        
        # Извлекаем лицо
        face_img = crop_face(img)
        face_img.save(face_path)
        
        # Получаем рекомендации и анализ
        result = recommend_lipsticks(face_img)
        recommendations = result["recommendations"]
        analysis = result["analysis"]
        
        # Применяем лучший оттенок
        top_shade = recommendations[0]
        colored_img = apply_lipstick(img, hex_color=top_shade["hex"])
        colored_img.save(result_path)
        
        # Формируем ответ
        response = {
            "recommendations": recommendations,
            "analysis": {
                "color_type": analysis["season"],
                "skin_color": analysis["skin_color"],
                "hair_color": analysis["hair_color"],
                "metrics": analysis["analysis"]
            },
            "images": {
                "face": f"/images/{os.path.basename(face_path)}",
                "result": f"/images/{os.path.basename(result_path)}"
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/images/{filename}")
async def get_image(filename: str):
    """Получение изображения по имени файла"""
    file_path = os.path.join(TEMP_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(
        status_code=404,
        content={"error": "Image not found"}
    )

# Запуск сервера: uvicorn server:app --reload 