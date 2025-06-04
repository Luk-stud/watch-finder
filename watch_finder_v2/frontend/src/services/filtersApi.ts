import { ENV_CONFIG } from '@/config/environment';

export interface FilterOptions {
  brands: string[];
  caseMaterials: string[];
  movements: string[];
  dialColors: string[];
  watchTypes: string[];
  complications: string[];
  priceRange: [number, number];
  diameterRange: [number, number];
  thicknessRange: [number, number];
  waterResistanceOptions: number[];
}

class FiltersApiService {
  private baseUrl: string;
  private fallbackUrls: string[];

  constructor() {
    this.baseUrl = ENV_CONFIG.API_BASE_URL;
    this.fallbackUrls = ENV_CONFIG.FALLBACK_URLS;
  }

  /**
   * Fetch available filter options from the backend
   */
  async getFilterOptions(): Promise<FilterOptions> {
    const urls = [this.baseUrl, ...this.fallbackUrls];
    
    for (const url of urls) {
      try {
        console.log(`🔍 Fetching filter options from: ${url}/filter-options`);
        
        const response = await fetch(`${url}/filter-options`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          signal: AbortSignal.timeout(10000), // 10 second timeout
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        
        if (data.error) {
          throw new Error(data.error);
        }

        console.log('✅ Filter options fetched successfully:', {
          brands: data.brands?.length || 0,
          caseMaterials: data.caseMaterials?.length || 0,
          movements: data.movements?.length || 0,
          dialColors: data.dialColors?.length || 0,
          watchTypes: data.watchTypes?.length || 0,
          complications: data.complications?.length || 0,
        });

        return {
          brands: data.brands || [],
          caseMaterials: data.caseMaterials || [],
          movements: data.movements || [],
          dialColors: data.dialColors || [],
          watchTypes: data.watchTypes || [],
          complications: data.complications || [],
          priceRange: data.priceRange || [0, 50000],
          diameterRange: data.diameterRange || [30, 50],
          thicknessRange: data.thicknessRange || [5, 20],
          waterResistanceOptions: data.waterResistanceOptions || [],
        };

      } catch (error) {
        console.warn(`❌ Failed to fetch from ${url}:`, error);
        
        // If this is the last URL, throw the error
        if (url === urls[urls.length - 1]) {
          console.error('❌ All API endpoints failed, using fallback data');
          return this.getFallbackFilterOptions();
        }
      }
    }

    // This should never be reached, but TypeScript requires it
    return this.getFallbackFilterOptions();
  }

  /**
   * Get fallback filter options when API is unavailable
   */
  private getFallbackFilterOptions(): FilterOptions {
    return {
      brands: [
        'Rolex', 'Omega', 'Patek Philippe', 'Audemars Piguet', 'Tudor', 'Seiko',
        'Casio', 'Tissot', 'TAG Heuer', 'Breitling', 'IWC', 'Cartier',
        'Melbourne Watch Co', 'Citizen', 'Orient', 'Hamilton', 'Longines'
      ],
      caseMaterials: [
        'Stainless Steel', '316L Stainless Steel', 'Titanium', 'Gold',
        'Rose Gold', 'White Gold', 'Platinum', 'Ceramic', 'Carbon Fiber',
        'Bronze', 'Aluminum'
      ],
      movements: [
        'Automatic', 'Manual', 'Quartz', 'Solar', 'Kinetic', 'Spring Drive',
        'Miyota 9015', 'ETA 2824', 'Seiko NH35', 'Swiss Made'
      ],
      dialColors: [
        'Black', 'White', 'Blue', 'Green', 'Silver', 'Grey', 'Brown',
        'Gold', 'Red', 'Orange', 'Purple', 'Champagne', 'Salmon'
      ],
      watchTypes: [
        'Dress', 'Sport', 'Diver', 'GMT', 'Chronograph', 'Field',
        'Pilot', 'Racing', 'Military', 'Casual'
      ],
      complications: [
        'Date', 'Day-Date', 'GMT', 'Chronograph', 'Moon Phase',
        'Power Reserve', 'Tourbillon', 'Perpetual Calendar', 'Minute Repeater'
      ],
      priceRange: [0, 50000],
      diameterRange: [30, 50],
      thicknessRange: [5, 20],
      waterResistanceOptions: [30, 50, 100, 200, 300, 500, 1000],
    };
  }

  /**
   * Test API connectivity
   */
  async testConnection(): Promise<boolean> {
    const urls = [this.baseUrl, ...this.fallbackUrls];
    
    for (const url of urls) {
      try {
        const response = await fetch(`${url}/health`, {
          method: 'GET',
          signal: AbortSignal.timeout(5000), // 5 second timeout
        });

        if (response.ok) {
          console.log(`✅ API connection successful: ${url}`);
          return true;
        }
      } catch (error) {
        console.warn(`❌ Connection failed: ${url}`, error);
      }
    }

    console.error('❌ All API endpoints are unreachable');
    return false;
  }
}

// Export singleton instance
export const filtersApi = new FiltersApiService();
export default filtersApi; 