import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';
import { useNavigate } from 'react-router-dom';
import { useToast } from '@/hooks/use-toast';
import { useFilters } from '@/context/FiltersContext';
import { filtersApi, FilterOptions } from '@/services/filtersApi';
import { Settings, Filter, Sliders, Eye, Brain, Palette, Search, Loader2, AlertTriangle } from 'lucide-react';

export interface FilterPreferences {
  // Similarity preferences
  clipSimilarityWeight: number; // 0-100, how much visual similarity matters
  textSimilarityWeight: number; // 0-100, how much description/vibe similarity matters
  
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
  clipSimilarityWeight: 50,
  textSimilarityWeight: 50,
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

const Filters: React.FC = () => {
  const { filters, updateFilter, resetFilters, hasActiveFilters } = useFilters();
  
  const [availableOptions, setAvailableOptions] = useState<FilterOptions>({
    brands: [],
    caseMaterials: [],
    movements: [],
    dialColors: [],
    watchTypes: [],
    complications: [],
    priceRange: [0, 50000],
    diameterRange: [30, 50],
    thicknessRange: [5, 20],
    waterResistanceOptions: [],
  });
  
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const navigate = useNavigate();
  const { toast } = useToast();
  
  // Load available filter options from API
  useEffect(() => {
    const loadFilterOptions = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        console.log('🔍 Loading filter options from API...');
        const options = await filtersApi.getFilterOptions();
        setAvailableOptions(options);
        
        // Update price and diameter ranges in filters if they're at default values
        if (filters.priceRange[0] === 0 && filters.priceRange[1] === 50000) {
          updateFilter('priceRange', options.priceRange);
        }
        if (filters.minDiameter === 30 && filters.maxDiameter === 50) {
          updateFilter('minDiameter', options.diameterRange[0]);
          updateFilter('maxDiameter', options.diameterRange[1]);
        }
        if (filters.minThickness === 5 && filters.maxThickness === 20) {
          updateFilter('minThickness', options.thicknessRange[0]);
          updateFilter('maxThickness', options.thicknessRange[1]);
        }
        
        console.log('✅ Filter options loaded successfully');
      } catch (error) {
        console.error('❌ Failed to load filter options:', error);
        setError('Failed to load filter options. Using default values.');
        
        toast({
          title: "Warning",
          description: "Could not load latest filter options from server. Using defaults.",
          variant: "destructive"
        });
      } finally {
        setIsLoading(false);
      }
    };
    
    loadFilterOptions();
  }, []);
  
  const toggleArrayItem = (array: string[], item: string) => {
    return array.includes(item) 
      ? array.filter(i => i !== item)
      : [...array, item];
  };
  
  const handleSaveAndContinue = () => {
    // Validate similarity weights
    const totalWeight = filters.clipSimilarityWeight + filters.textSimilarityWeight;
    if (totalWeight === 0) {
      toast({
        title: "Invalid Similarity Settings",
        description: "At least one similarity type must have weight > 0",
        variant: "destructive"
      });
      return;
    }
    
    toast({
      title: "Filters Saved! 🎯",
      description: "Your preferences have been saved and will be applied to recommendations.",
    });
    
    // Navigate to the main swipe interface
    navigate('/swipe');
  };
  
  const handleResetFilters = () => {
    resetFilters();
    toast({
      title: "Filters Reset",
      description: "All filters have been reset to default values.",
    });
  };
  
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 p-4 flex items-center justify-center">
        <div className="text-center space-y-4">
          <Loader2 className="h-8 w-8 animate-spin mx-auto text-blue-600" />
          <h2 className="text-xl font-semibold text-gray-900">Loading Filter Options</h2>
          <p className="text-gray-600">Fetching the latest watch specifications...</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      <div className="h-screen overflow-y-auto overflow-x-hidden scroll-smooth">
        <div className="max-w-4xl mx-auto space-y-6 p-4 pb-20 pt-safe-area-inset-top">
          {/* Header */}
          <div className="text-center space-y-2 pt-4">
            <div className="flex items-center justify-center gap-2">
              <Settings className="h-8 w-8 text-blue-600" />
              <h1 className="text-3xl font-bold text-gray-900">Watch Preferences</h1>
            </div>
            <p className="text-gray-600 max-w-2xl mx-auto">
              Customize your watch discovery experience. Set your similarity preferences and filter criteria 
              to get personalized recommendations that match your style and preferences.
            </p>
            
            {error && (
              <div className="flex items-center justify-center gap-2 text-amber-600 bg-amber-50 p-2 rounded-lg max-w-md mx-auto">
                <AlertTriangle className="h-4 w-4" />
                <span className="text-sm">{error}</span>
              </div>
            )}
          </div>
          
          {/* Similarity Preferences - Most Important */}
          <Card className="border-2 border-blue-200 bg-blue-50/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-blue-600" />
                Recommendation Algorithm Preferences
              </CardTitle>
              <CardDescription>
                Balance between visual similarity (how watches look) and vibe similarity (style descriptions, mood, occasion)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Visual Similarity Weight */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="flex items-center gap-2">
                    <Eye className="h-4 w-4 text-purple-600" />
                    Visual Similarity Weight: {filters.clipSimilarityWeight}%
                  </Label>
                  <Badge variant="secondary">{filters.clipSimilarityWeight}%</Badge>
                </div>
                <Slider
                  value={[filters.clipSimilarityWeight]}
                  onValueChange={([value]) => updateFilter('clipSimilarityWeight', value)}
                  max={100}
                  step={5}
                  className="w-full"
                />
                <p className="text-sm text-gray-600">
                  How much visual appearance matters (case shape, dial layout, hands, markers, colors, materials)
                </p>
              </div>
              
              {/* Text/Vibe Similarity Weight */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="flex items-center gap-2">
                    <Palette className="h-4 w-4 text-green-600" />
                    Vibe Similarity Weight: {filters.textSimilarityWeight}%
                  </Label>
                  <Badge variant="secondary">{filters.textSimilarityWeight}%</Badge>
                </div>
                <Slider
                  value={[filters.textSimilarityWeight]}
                  onValueChange={([value]) => updateFilter('textSimilarityWeight', value)}
                  max={100}
                  step={5}
                  className="w-full"
                />
                <p className="text-sm text-gray-600">
                  How much style and mood matters (elegance, sportiness, occasion, personality, feeling)
                </p>
              </div>
              
              {/* Total weight indicator */}
              <div className="bg-white p-3 rounded-lg border">
                <div className="flex justify-between items-center">
                  <span className="text-sm font-medium">Total Weight:</span>
                  <span className="font-bold text-lg">
                    {filters.clipSimilarityWeight + filters.textSimilarityWeight}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div 
                    className="bg-gradient-to-r from-purple-500 to-green-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${Math.min(100, filters.clipSimilarityWeight + filters.textSimilarityWeight)}%` }}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
          
          {/* Watch Filters */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Filter className="h-5 w-5 text-green-600" />
                Watch Filters
              </CardTitle>
              <CardDescription>
                Filter watches by specific characteristics and specifications
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Accordion type="multiple" className="w-full">
                
                {/* Basic Filters */}
                <AccordionItem value="basic">
                  <AccordionTrigger>Basic Filters</AccordionTrigger>
                  <AccordionContent className="space-y-4">
                    
                    {/* Price Range */}
                    <div className="space-y-3">
                      <Label>Price Range: ${filters.priceRange[0].toLocaleString()} - ${filters.priceRange[1].toLocaleString()}</Label>
                      <div className="px-3">
                        <Slider
                          value={filters.priceRange}
                          onValueChange={(value) => updateFilter('priceRange', value as [number, number])}
                          max={availableOptions.priceRange[1]}
                          min={availableOptions.priceRange[0]}
                          step={100}
                          className="w-full"
                        />
                      </div>
                    </div>
                    
                    {/* Brands */}
                    {availableOptions.brands.length > 0 && (
                      <div className="space-y-3">
                        <Label>Brands ({filters.brands.length} selected)</Label>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-40 overflow-y-auto">
                          {availableOptions.brands.map(brand => (
                            <div key={brand} className="flex items-center space-x-2">
                              <Checkbox
                                id={`brand-${brand}`}
                                checked={filters.brands.includes(brand)}
                                onCheckedChange={() => 
                                  updateFilter('brands', toggleArrayItem(filters.brands, brand))
                                }
                              />
                              <Label htmlFor={`brand-${brand}`} className="text-sm">{brand}</Label>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Watch Types */}
                    {availableOptions.watchTypes.length > 0 && (
                      <div className="space-y-3">
                        <Label>Watch Types ({filters.watchTypes.length} selected)</Label>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                          {availableOptions.watchTypes.map(type => (
                            <div key={type} className="flex items-center space-x-2">
                              <Checkbox
                                id={`type-${type}`}
                                checked={filters.watchTypes.includes(type)}
                                onCheckedChange={() => 
                                  updateFilter('watchTypes', toggleArrayItem(filters.watchTypes, type))
                                }
                              />
                              <Label htmlFor={`type-${type}`} className="text-sm">{type}</Label>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                  </AccordionContent>
                </AccordionItem>
                
                {/* Technical Specifications */}
                <AccordionItem value="technical">
                  <AccordionTrigger>Technical Specifications</AccordionTrigger>
                  <AccordionContent className="space-y-4">
                    
                    {/* Case Materials */}
                    {availableOptions.caseMaterials.length > 0 && (
                      <div className="space-y-3">
                        <Label>Case Materials ({filters.caseMaterials.length} selected)</Label>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 max-h-32 overflow-y-auto">
                          {availableOptions.caseMaterials.map(material => (
                            <div key={material} className="flex items-center space-x-2">
                              <Checkbox
                                id={`material-${material}`}
                                checked={filters.caseMaterials.includes(material)}
                                onCheckedChange={() => 
                                  updateFilter('caseMaterials', toggleArrayItem(filters.caseMaterials, material))
                                }
                              />
                              <Label htmlFor={`material-${material}`} className="text-sm">{material}</Label>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Movements */}
                    {availableOptions.movements.length > 0 && (
                      <div className="space-y-3">
                        <Label>Movements ({filters.movements.length} selected)</Label>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                          {availableOptions.movements.map(movement => (
                            <div key={movement} className="flex items-center space-x-2">
                              <Checkbox
                                id={`movement-${movement}`}
                                checked={filters.movements.includes(movement)}
                                onCheckedChange={() => 
                                  updateFilter('movements', toggleArrayItem(filters.movements, movement))
                                }
                              />
                              <Label htmlFor={`movement-${movement}`} className="text-sm">{movement}</Label>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {/* Complications */}
                    {availableOptions.complications.length > 0 && (
                      <div className="space-y-3">
                        <Label>Complications ({filters.complications.length} selected)</Label>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                          {availableOptions.complications.map(complication => (
                            <div key={complication} className="flex items-center space-x-2">
                              <Checkbox
                                id={`comp-${complication}`}
                                checked={filters.complications.includes(complication)}
                                onCheckedChange={() => 
                                  updateFilter('complications', toggleArrayItem(filters.complications, complication))
                                }
                              />
                              <Label htmlFor={`comp-${complication}`} className="text-sm">{complication}</Label>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                  </AccordionContent>
                </AccordionItem>
                
                {/* Physical Dimensions */}
                <AccordionItem value="dimensions">
                  <AccordionTrigger>Physical Dimensions</AccordionTrigger>
                  <AccordionContent className="space-y-4">
                    
                    {/* Case Diameter */}
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label htmlFor="minDiameter">Min Diameter (mm)</Label>
                        <Input
                          id="minDiameter"
                          type="number"
                          value={filters.minDiameter}
                          onChange={(e) => updateFilter('minDiameter', Number(e.target.value))}
                          min={availableOptions.diameterRange[0]}
                          max={availableOptions.diameterRange[1]}
                        />
                      </div>
                      <div>
                        <Label htmlFor="maxDiameter">Max Diameter (mm)</Label>
                        <Input
                          id="maxDiameter"
                          type="number"
                          value={filters.maxDiameter}
                          onChange={(e) => updateFilter('maxDiameter', Number(e.target.value))}
                          min={availableOptions.diameterRange[0]}
                          max={availableOptions.diameterRange[1]}
                        />
                      </div>
                    </div>
                    
                    {/* Case Thickness */}
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label htmlFor="minThickness">Min Thickness (mm)</Label>
                        <Input
                          id="minThickness"
                          type="number"
                          value={filters.minThickness}
                          onChange={(e) => updateFilter('minThickness', Number(e.target.value))}
                          min={availableOptions.thicknessRange[0]}
                          max={availableOptions.thicknessRange[1]}
                          step="0.1"
                        />
                      </div>
                      <div>
                        <Label htmlFor="maxThickness">Max Thickness (mm)</Label>
                        <Input
                          id="maxThickness"
                          type="number"
                          value={filters.maxThickness}
                          onChange={(e) => updateFilter('maxThickness', Number(e.target.value))}
                          min={availableOptions.thicknessRange[0]}
                          max={availableOptions.thicknessRange[1]}
                          step="0.1"
                        />
                      </div>
                    </div>
                    
                    {/* Water Resistance */}
                    <div className="space-y-2">
                      <Label>Minimum Water Resistance: {filters.waterResistance}m</Label>
                      <Slider
                        value={[filters.waterResistance]}
                        onValueChange={([value]) => updateFilter('waterResistance', value)}
                        max={Math.max(1000, ...availableOptions.waterResistanceOptions)}
                        step={50}
                        className="w-full"
                      />
                      {availableOptions.waterResistanceOptions.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-2">
                          {availableOptions.waterResistanceOptions.slice(0, 10).map(wr => (
                            <Badge
                              key={wr}
                              variant="outline"
                              className="cursor-pointer hover:bg-gray-100"
                              onClick={() => updateFilter('waterResistance', wr)}
                            >
                              {wr}m
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>
                    
                  </AccordionContent>
                </AccordionItem>
                
                {/* Style & Aesthetics */}
                <AccordionItem value="style">
                  <AccordionTrigger>Style & Aesthetics</AccordionTrigger>
                  <AccordionContent className="space-y-4">
                    
                    {/* Dial Colors */}
                    {availableOptions.dialColors.length > 0 && (
                      <div className="space-y-3">
                        <Label>Dial Colors ({filters.dialColors.length} selected)</Label>
                        <div className="grid grid-cols-3 md:grid-cols-4 gap-2">
                          {availableOptions.dialColors.map(color => (
                            <div key={color} className="flex items-center space-x-2">
                              <Checkbox
                                id={`color-${color}`}
                                checked={filters.dialColors.includes(color)}
                                onCheckedChange={() => 
                                  updateFilter('dialColors', toggleArrayItem(filters.dialColors, color))
                                }
                              />
                              <Label htmlFor={`color-${color}`} className="text-sm">{color}</Label>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                  </AccordionContent>
                </AccordionItem>
                
                {/* Special Features */}
                <AccordionItem value="special">
                  <AccordionTrigger>Special Features & Availability</AccordionTrigger>
                  <AccordionContent className="space-y-4">
                    
                    {/* Limited Edition */}
                    <div className="flex items-center justify-between">
                      <Label>Limited Edition Only</Label>
                      <Switch
                        checked={filters.limitedEdition === true}
                        onCheckedChange={(checked) => 
                          updateFilter('limitedEdition', checked ? true : null)
                        }
                      />
                    </div>
                    
                    {/* Vintage */}
                    <div className="flex items-center justify-between">
                      <Label>Vintage Watches Only</Label>
                      <Switch
                        checked={filters.vintage === true}
                        onCheckedChange={(checked) => 
                          updateFilter('vintage', checked ? true : null)
                        }
                      />
                    </div>
                    
                    {/* In Stock */}
                    <div className="flex items-center justify-between">
                      <Label>In Stock Only</Label>
                      <Switch
                        checked={filters.inStock}
                        onCheckedChange={(checked) => updateFilter('inStock', checked)}
                      />
                    </div>
                    
                  </AccordionContent>
                </AccordionItem>
                
              </Accordion>
            </CardContent>
          </Card>
          
          {/* Action Buttons */}
          <div className="flex gap-4 justify-center">
            <Button variant="outline" onClick={handleResetFilters}>
              Reset All
            </Button>
            <Button onClick={handleSaveAndContinue} size="lg" className="px-8">
              <Search className="h-4 w-4 mr-2" />
              Save & Start Discovering
            </Button>
          </div>
          
          {/* Active Filters Summary */}
          {hasActiveFilters && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Active Filters Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {filters.brands.length > 0 && (
                    <div>
                      <span className="font-medium">Brands: </span>
                      {filters.brands.map(brand => (
                        <Badge key={brand} variant="secondary" className="mr-1">
                          {brand}
                        </Badge>
                      ))}
                    </div>
                  )}
                  {filters.watchTypes.length > 0 && (
                    <div>
                      <span className="font-medium">Types: </span>
                      {filters.watchTypes.map(type => (
                        <Badge key={type} variant="secondary" className="mr-1">
                          {type}
                        </Badge>
                      ))}
                    </div>
                  )}
                  <div>
                    <span className="font-medium">Price: </span>
                    <Badge variant="outline">
                      ${filters.priceRange[0].toLocaleString()} - ${filters.priceRange[1].toLocaleString()}
                    </Badge>
                  </div>
                  <div>
                    <span className="font-medium">Algorithm: </span>
                    <Badge variant="outline">
                      {filters.clipSimilarityWeight}% Visual + {filters.textSimilarityWeight}% Vibe
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
          
        </div>
        
        {/* Sticky Action Bar for Mobile */}
        <div className="sticky bottom-0 left-0 right-0 bg-white/95 backdrop-blur-sm border-t border-gray-200 p-4 md:hidden">
          <div className="flex gap-3 max-w-sm mx-auto">
            <Button variant="outline" onClick={handleResetFilters} className="flex-1">
              Reset All
            </Button>
            <Button onClick={handleSaveAndContinue} className="flex-1">
              <Search className="h-4 w-4 mr-2" />
              Save & Discover
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Filters; 