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
        season_shades = get_palette(season)
        # Убедимся, что у каждого оттенка есть информация о сезоне
        for shade in season_shades:
            if "season" not in shade:
                shade["season"] = season
        shades.extend(season_shades)
    return shades
