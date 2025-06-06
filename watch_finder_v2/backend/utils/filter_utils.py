def should_include_watch(watch, filter_preferences):
    """Check if a watch should be included based on filter preferences."""
    if not filter_preferences:
        return True
    
    # Get specs once for reuse
    specs = watch.get('specs', {})
    
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
    if case_materials:
        watch_case_material = specs.get('case_material')
        if watch_case_material not in case_materials:
            return False
    
    # Movement filter
    movements = filter_preferences.get('movements')
    if movements:
        watch_movement = specs.get('movement') or specs.get('winding')
        if watch_movement not in movements:
            return False
    
    # Dial color filter
    dial_colors = filter_preferences.get('dialColors')
    if dial_colors:
        watch_dial_color = specs.get('dial_color')
        if watch_dial_color not in dial_colors:
            return False
    
    # Watch type filter
    watch_types = filter_preferences.get('watchTypes')
    if watch_types:
        watch_type = specs.get('watch_type') or specs.get('second_watch_type')
        if watch_type not in watch_types:
            return False
    
    # Size filters
    min_diameter = filter_preferences.get('minDiameter')
    max_diameter = filter_preferences.get('maxDiameter')
    
    # Get diameter from specs (it's stored as diameter_mm)
    diameter = specs.get('diameter_mm')
    if diameter and str(diameter).strip() not in ['-', '', 'N/A', 'None']:
        try:
            diameter = float(diameter)
            if min_diameter and diameter < min_diameter:
                return False
            if max_diameter and diameter > max_diameter:
                return False
        except (ValueError, TypeError):
            # If diameter can't be converted to float, don't filter based on it
            # This allows watches with missing/invalid diameter data to pass through
            pass
    elif diameter is None or str(diameter).strip() in ['-', '', 'N/A', 'None']:
        # If user has diameter preferences but watch has no diameter data,
        # exclude it to avoid showing irrelevant results
        if min_diameter or max_diameter:
            return False
    
    min_thickness = filter_preferences.get('minThickness')
    max_thickness = filter_preferences.get('maxThickness')
    
    # Get thickness from specs (could be thickness_with_crystal_mm or thickness_without_crystal_mm)
    thickness = specs.get('thickness_with_crystal_mm') or specs.get('thickness_without_crystal_mm')
    if thickness:
        try:
            thickness = float(thickness)
            if min_thickness and thickness < min_thickness:
                return False
            if max_thickness and thickness > max_thickness:
                return False
        except (ValueError, TypeError):
            pass  # Skip if thickness can't be converted to float
    
    # Water resistance filter
    min_water_resistance = filter_preferences.get('waterResistance')
    if min_water_resistance:
        water_resistance = specs.get('waterproofing_meters')
        if water_resistance:
            try:
                water_resistance = float(water_resistance)
                if water_resistance < min_water_resistance:
                    return False
            except (ValueError, TypeError):
                pass
    
    # Limited edition filter
    limited_edition = filter_preferences.get('limitedEdition')
    if limited_edition is True and not specs.get('limited_edition_name'):
        return False
    
    # Vintage filter - this might need custom logic based on launch_date
    vintage = filter_preferences.get('vintage')
    if vintage is True:
        launch_date = specs.get('launch_date')
        # Consider vintage if launched before 2000 or no launch date
        if launch_date:
            try:
                year = int(launch_date.split('-')[0]) if '-' in launch_date else int(launch_date)
                if year >= 2000:
                    return False
            except (ValueError, TypeError):
                pass
    
    # In stock filter
    in_stock = filter_preferences.get('inStock')
    if in_stock and specs.get('availability') == 'Sold Out':
        return False
    
    return True 