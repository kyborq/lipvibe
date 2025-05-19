import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "lipsticks.db")

def get_lipsticks_by_season(season):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT name, hex, brand FROM lipsticks WHERE season = ?', (season,))
    results = cursor.fetchall()
    conn.close()

    return [
        {"name": name, "hex": hex_code, "example": brand}
        for name, hex_code, brand in results
    ]
