from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
from typing import List, Dict, Optional
import uuid
import sqlite3

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

def get_db_connection():
    conn = sqlite3.connect('database/lipsticks.db')
    conn.row_factory = sqlite3.Row
    return conn

def check_and_initialize_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if we have any data
    cursor.execute("SELECT COUNT(*) FROM lipsticks")
    count = cursor.fetchone()[0]
    
    if count == 0:
        # Initialize database with default data
        initialize_database()
    
    conn.close()

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске сервера"""
    check_and_initialize_data()

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

@app.get("/lipsticks")
async def get_lipsticks(color: Optional[str] = Query(None, description="Filter by hex color")):
    """Получение списка помад с возможностью фильтрации по цвету"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if color:
            # Convert hex to RGB for comparison
            color = color.lstrip('#')
            r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            
            # Query with color similarity
            cursor.execute("""
                SELECT name, brand, hex, description
                FROM lipsticks
                WHERE ABS(CAST(substr(hex, 2, 2) AS INTEGER) - ?) <= 20
                AND ABS(CAST(substr(hex, 4, 2) AS INTEGER) - ?) <= 20
                AND ABS(CAST(substr(hex, 6, 2) AS INTEGER) - ?) <= 20
                ORDER BY (
                    ABS(CAST(substr(hex, 2, 2) AS INTEGER) - ?) +
                    ABS(CAST(substr(hex, 4, 2) AS INTEGER) - ?) +
                    ABS(CAST(substr(hex, 6, 2) AS INTEGER) - ?)
                )
                LIMIT 10
            """, (r, g, b, r, g, b))
        else:
            cursor.execute("SELECT name, brand, hex, description FROM lipsticks LIMIT 10")
        
        lipsticks = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return JSONResponse(content={"lipsticks": lipsticks})
        
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