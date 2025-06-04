// Import the modern watch interface from our API
import { ModernWatch } from '../lib/api';

// Re-export the ModernWatch interface as Watch for compatibility
export type Watch = ModernWatch;

// Remove the static mock data - we'll get real data from the backend
export const watchData: Watch[] = [];

// Helper functions for the frontend
export const formatPrice = (price?: number | string): string => {
  if (!price) return 'Contact for price';
  
  if (typeof price === 'string') {
    // Handle string prices like "$1,000" or "1000"
    const numPrice = parseFloat(price.replace(/[^0-9.]/g, ''));
    if (isNaN(numPrice)) return price;
    price = numPrice;
  }
  
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(price);
};

export const getWatchImageUrl = (watch: Watch): string => {
  // Priority order: local image path > main image > image_url > placeholder
  if (watch.local_image_path) {
    return `/${watch.local_image_path}`;
  }
  
  if (watch.main_image) {
    return watch.main_image;
  }
  
  if (watch.image_url) {
    return watch.image_url;
  }
  
  // Return a placeholder
  return `data:image/svg+xml;base64,${btoa(`
    <svg width="400" height="400" viewBox="0 0 400 400" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect width="400" height="400" fill="#F3F4F6"/>
      <path d="M200 100C144.477 100 100 144.477 100 200S144.477 300 200 300S299.999 255.523 299.999 200S255.522 100 200 100ZM200 120C244.183 120 280 155.817 280 200S244.183 280 200 280S120 244.183 120 200S155.817 120 200 120Z" fill="#9CA3AF"/>
      <text x="200" y="340" text-anchor="middle" fill="#6B7280" font-family="Arial" font-size="16">${watch.brand || 'Watch'}</text>
    </svg>
  `)}`;
};

export const getWatchComplications = (watch: Watch): string[] => {
  if (!watch.specs) return [];
  
  const complications = [];
  if (watch.specs.complication_date && watch.specs.complication_date !== '-') complications.push('Date');
  if (watch.specs.complication_chronograph && watch.specs.complication_chronograph !== '-') complications.push('Chronograph');
  if (watch.specs.complication_gmt && watch.specs.complication_gmt !== '-') complications.push('GMT');
  if (watch.specs.complication_dual_time && watch.specs.complication_dual_time !== '-') complications.push('Dual Time');
  if (watch.specs.complication_power_reserve && watch.specs.complication_power_reserve !== '-') complications.push('Power Reserve');
  if (watch.specs.complication_moonphase && watch.specs.complication_moonphase !== '-') complications.push('Moonphase');
  if (watch.specs.complication_world_time_zone && watch.specs.complication_world_time_zone !== '-') complications.push('World Time');
  if (watch.specs.complication_alarm && watch.specs.complication_alarm !== '-') complications.push('Alarm');
  
  return complications;
};

export const getWatchMovement = (watch: Watch): string => {
  return watch.specs?.movement || watch.movement || 'N/A';
};

export const getWatchCaseMaterial = (watch: Watch): string => {
  return watch.specs?.case_material || watch.case_material || 'N/A';
};

export const getWatchDiameter = (watch: Watch): string => {
  if (watch.specs?.diameter_mm) {
    return `${watch.specs.diameter_mm}mm`;
  }
  if (watch.case_diameter) {
    return watch.case_diameter;
  }
  return 'N/A';
};

export const getWatchWaterResistance = (watch: Watch): string => {
  if (watch.specs?.waterproofing_meters) {
    return `${watch.specs.waterproofing_meters}m`;
  }
  if (watch.water_resistance) {
    return watch.water_resistance;
  }
  return 'N/A';
};

export const getWatchPrice = (watch: Watch): number => {
  // Try to extract a numeric price from various fields
  if (watch.price && typeof watch.price === 'number') {
    return watch.price;
  }
  
  if (watch.specs?.price_usd) {
    const price = parseFloat(watch.specs.price_usd.replace(/[^0-9.]/g, ''));
    if (!isNaN(price)) return price;
  }
  
  if (watch.specs?.msrp_eur) {
    const price = parseFloat(watch.specs.msrp_eur.replace(/[^0-9.]/g, ''));
    if (!isNaN(price)) return price * 1.1; // Rough EUR to USD conversion
  }
  
  if (watch.specs?.launch_price_eur) {
    const price = parseFloat(watch.specs.launch_price_eur.replace(/[^0-9.]/g, ''));
    if (!isNaN(price)) return price * 1.1; // Rough EUR to USD conversion
  }
  
  return 0; // Unknown price
};
