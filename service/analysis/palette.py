from database.schema import get_lipsticks_by_season

def get_palette(season: str):
    """
    Возвращает список оттенков для заданного сезона
    """
    return get_lipsticks_by_season(season)

def get_all_shades():
    """
    Возвращает все оттенки из всех сезонов (для эмбеддинга и сравнения)
    """
    all_seasons = ["Winter", "Spring", "Summer", "Autumn"]
    shades = []
    for season in all_seasons:
        shades.extend(get_palette(season))
    return shades
