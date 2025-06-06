import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface FilterPreferences {
  // Watch filters
  brands: string[];
  priceRange: [number, number];
  caseMaterials: string[];
  movements: string[];
  dialColors: string[];
  watchTypes: string[];
  complications: string[];
  
  // Size preferences
  minDiameter: number;
  maxDiameter: number;
  minThickness: number;
  maxThickness: number;
  
  // Additional filters
  waterResistance: number;
  limitedEdition: boolean | null; // null = no preference
  vintage: boolean | null;
  inStock: boolean;
}

const DEFAULT_FILTERS: FilterPreferences = {
  brands: [],
  priceRange: [0, 50000],
  caseMaterials: [],
  movements: [],
  dialColors: [],
  watchTypes: [],
  complications: [],
  minDiameter: 30,
  maxDiameter: 50,
  minThickness: 5,
  maxThickness: 20,
  waterResistance: 0,
  limitedEdition: null,
  vintage: null,
  inStock: false,
};

interface FiltersContextType {
  filters: FilterPreferences;
  updateFilter: <K extends keyof FilterPreferences>(key: K, value: FilterPreferences[K]) => void;
  resetFilters: () => void;
  hasActiveFilters: boolean;
  saveFilters: () => void;
  loadFilters: () => void;
}

const FiltersContext = createContext<FiltersContextType | undefined>(undefined);

interface FiltersProviderProps {
  children: ReactNode;
}

export const FiltersProvider: React.FC<FiltersProviderProps> = ({ children }) => {
  const [filters, setFilters] = useState<FilterPreferences>(() => {
    // Load from localStorage on initialization
    const saved = localStorage.getItem('watchFinderFilters');
    return saved ? { ...DEFAULT_FILTERS, ...JSON.parse(saved) } : DEFAULT_FILTERS;
  });

  // Save to localStorage whenever filters change
  useEffect(() => {
    localStorage.setItem('watchFinderFilters', JSON.stringify(filters));
  }, [filters]);

  const updateFilter = <K extends keyof FilterPreferences>(
    key: K, 
    value: FilterPreferences[K]
  ) => {
    setFilters(prev => ({ ...prev, [key]: value }));
  };

  const resetFilters = () => {
    setFilters(DEFAULT_FILTERS);
  };

  const saveFilters = () => {
    localStorage.setItem('watchFinderFilters', JSON.stringify(filters));
  };

  const loadFilters = () => {
    const saved = localStorage.getItem('watchFinderFilters');
    if (saved) {
      setFilters({ ...DEFAULT_FILTERS, ...JSON.parse(saved) });
    }
  };

  // Check if any filters are active (different from default)
  const hasActiveFilters = 
    filters.brands.length > 0 ||
    filters.caseMaterials.length > 0 ||
    filters.movements.length > 0 ||
    filters.dialColors.length > 0 ||
    filters.watchTypes.length > 0 ||
    filters.complications.length > 0 ||
    filters.priceRange[0] !== DEFAULT_FILTERS.priceRange[0] ||
    filters.priceRange[1] !== DEFAULT_FILTERS.priceRange[1] ||
    filters.minDiameter !== DEFAULT_FILTERS.minDiameter ||
    filters.maxDiameter !== DEFAULT_FILTERS.maxDiameter ||
    filters.minThickness !== DEFAULT_FILTERS.minThickness ||
    filters.maxThickness !== DEFAULT_FILTERS.maxThickness ||
    filters.waterResistance !== DEFAULT_FILTERS.waterResistance ||
    filters.limitedEdition !== null ||
    filters.vintage !== null ||
    filters.inStock !== DEFAULT_FILTERS.inStock;

  const value = {
    filters,
    updateFilter,
    resetFilters,
    hasActiveFilters,
    saveFilters,
    loadFilters,
  };

  return (
    <FiltersContext.Provider value={value}>
      {children}
    </FiltersContext.Provider>
  );
};

export const useFilters = (): FiltersContextType => {
  const context = useContext(FiltersContext);
  if (context === undefined) {
    throw new Error('useFilters must be used within a FiltersProvider');
  }
  return context;
};

export { DEFAULT_FILTERS }; 