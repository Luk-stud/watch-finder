def should_include_watch(watch, filter_preferences):
    """Check if a watch should be included based on filter preferences."""
    if not filter_preferences:
        return True
    
    # Brand filter
    if filter_preferences.get('brands') and watch.get('brand') not in filter_preferences['brands']:
        return False
    
    # Price filter
    price_range = filter_preferences.get('priceRange')
    if price_range and watch.get('price'):
        if watch['price'] < price_range[0] or watch['price'] > price_range[1]:
            return False
    
    # Case material filter
    case_materials = filter_preferences.get('caseMaterials')
    if case_materials and watch.get('case_material') not in case_materials:
        return False
    
    # Movement filter
    movements = filter_preferences.get('movements')
    if movements and watch.get('movement') not in movements:
        return False
    
    # Dial color filter
    dial_colors = filter_preferences.get('dialColors')
    if dial_colors and watch.get('dial_color') not in dial_colors:
        return False
    
    # Watch type filter
    watch_types = filter_preferences.get('watchTypes')
    if watch_types and watch.get('watch_type') not in watch_types:
        return False
    
    # Size filters
    min_diameter = filter_preferences.get('minDiameter')
    max_diameter = filter_preferences.get('maxDiameter')
    if min_diameter and watch.get('diameter') and watch['diameter'] < min_diameter:
        return False
    if max_diameter and watch.get('diameter') and watch['diameter'] > max_diameter:
        return False
    
    min_thickness = filter_preferences.get('minThickness')
    max_thickness = filter_preferences.get('maxThickness')
    if min_thickness and watch.get('thickness') and watch['thickness'] < min_thickness:
        return False
    if max_thickness and watch.get('thickness') and watch['thickness'] > max_thickness:
        return False
    
    # Water resistance filter
    min_water_resistance = filter_preferences.get('waterResistance')
    if min_water_resistance and watch.get('water_resistance', 0) < min_water_resistance:
        return False
    
    # Limited edition filter
    limited_edition = filter_preferences.get('limitedEdition')
    if limited_edition is True and not watch.get('limited_edition', False):
        return False
    
    # Vintage filter
    vintage = filter_preferences.get('vintage')
    if vintage is True and not watch.get('vintage', False):
        return False
    
    # In stock filter
    in_stock = filter_preferences.get('inStock')
    if in_stock and not watch.get('in_stock', True):
        return False
    
    return True 