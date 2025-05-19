import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "lipsticks.db")

def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lipsticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season TEXT,
            name TEXT,
            hex TEXT,
            brand TEXT
        )
    ''')

    lipsticks_data = [
        ("Winter", "Фуксия", "#D3167A", "MAC Flat Out Fabulous"),
        ("Winter", "Винный", "#600028", "Maybelline Divine Wine"),
        ("Winter", "Холодный красный", "#B80028", "MAC Ruby Woo"),
        ("Winter", "Ягодный", "#9C2B6D", "Revlon Berry Haute"),
        ("Spring", "Коралловый", "#F88379", "L'Oréal Color Riche C402"),
        ("Spring", "Абрикосовый нюд", "#E6A57E", "Maybelline Peach Buff"),
        ("Spring", "Тёплый розовый", "#F4A6B0", "NYX Pink the Town"),
        ("Spring", "Оранжево-красный", "#E34234", "L'Oréal Pure Fire"),
        ("Summer", "Пыльная роза", "#C48793", "Maybelline Touch of Spice"),
        ("Summer", "Приглушённая малина", "#B04E6F", "NYX Soft Spoken"),
        ("Summer", "Розово-коричневый", "#B97A57", "MAC Twig"),
        ("Summer", "Нежно-розовый", "#E8B2C8", "Revlon Primrose"),
        ("Autumn", "Терракота", "#CC4E30", "MAC Chili"),
        ("Autumn", "Кирпично-красный", "#B33B24", "NARS Mona"),
        ("Autumn", "Медный", "#B87333", "L'Oréal Bronze Coin"),
        ("Autumn", "Коричнево-рыжий", "#8B4000", "NYX Cold Brew")
    ]

    cursor.executemany('''
        INSERT INTO lipsticks (season, name, hex, brand)
        VALUES (?, ?, ?, ?)
    ''', lipsticks_data)

    conn.commit()
    conn.close()

def get_lipsticks_by_season(season: str):
    """
    Возвращает список помад для заданного сезона
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT name, hex, brand, season
        FROM lipsticks
        WHERE season = ?
    ''', (season,))
    
    results = cursor.fetchall()
    conn.close()
    
    return [{"name": name, "hex": hex, "brand": brand, "season": season} 
            for name, hex, brand, season in results]
