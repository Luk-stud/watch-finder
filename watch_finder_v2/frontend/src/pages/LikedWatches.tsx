import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Watch, getWatchImageUrl, formatPrice, getWatchPrice } from '../data/watchData';
import { ArrowLeft, Heart, ExternalLink, Info, Loader2, AlertTriangle, Settings } from 'lucide-react';
import { useViewportHeight } from '../hooks/useViewportHeight';
import { apiService } from '../lib/api';

const LikedWatches = () => {
  const [likedWatches, setLikedWatches] = useState<Watch[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Use the viewport height hook for mobile browser compatibility
  useViewportHeight();

  useEffect(() => {
    loadLikedWatches();
  }, []);

  const loadLikedWatches = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await apiService.getLikedWatches();
      if (response.status === 'success') {
        setLikedWatches(response.liked_watches);
      } else {
        throw new Error(response.message || 'Failed to load liked watches');
      }
    } catch (err) {
      console.error('Error loading liked watches:', err);
      setError(err instanceof Error ? err.message : 'Failed to load liked watches');
    } finally {
      setIsLoading(false);
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="flex-viewport bg-background">
        <div className="flex items-center justify-center h-full">
        <div className="text-center">
            <Loader2 className="w-8 h-8 mx-auto mb-4 animate-spin text-primary" />
            <h2 className="text-lg font-medium text-foreground">Loading your collection...</h2>
            <p className="text-sm text-muted-foreground mt-2">Gathering your liked watches</p>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex-viewport bg-background">
        <div className="flex items-center justify-center h-full p-6">
        <div className="text-center max-w-md">
            <div className="p-4 rounded-full bg-destructive/10 w-fit mx-auto mb-6">
              <AlertTriangle className="w-8 h-8 text-destructive" />
            </div>
            <h2 className="text-2xl font-semibold mb-3 text-foreground">Unable to load watches</h2>
            <p className="text-muted-foreground mb-6 leading-relaxed">{error}</p>
          <button
            onClick={loadLikedWatches}
              className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2"
          >
            Try Again
          </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-viewport bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="p-4 max-w-6xl mx-auto">
          <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
        <Link
                to="/"
                className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-9 w-9"
        >
                <ArrowLeft className="w-4 h-4" />
        </Link>
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                  <Heart className="w-4 h-4 text-primary-foreground" />
                </div>
        <div>
                  <h1 className="text-2xl font-semibold text-foreground">Your Collection</h1>
                  <p className="text-sm text-muted-foreground">
                    {likedWatches.length} watch{likedWatches.length !== 1 ? 'es' : ''} saved
              </p>
            </div>
          </div>
            </div>
            <div className="flex items-center gap-2 text-red-500">
              <Heart className="w-5 h-5" />
              <span className="text-xl font-semibold">{likedWatches.length}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="flex-1 overflow-auto p-6">
        <div className="max-w-6xl mx-auto">
        {likedWatches.length === 0 ? (
            <div className="text-center py-20">
              <div className="p-4 rounded-full bg-muted/50 w-fit mx-auto mb-6">
                <Heart className="w-12 h-12 text-muted-foreground" />
              </div>
              <h2 className="text-2xl font-semibold text-foreground mb-3">No watches saved yet</h2>
              <p className="text-muted-foreground mb-8 max-w-md mx-auto leading-relaxed">
                Start discovering and liking watches to build your personal collection
              </p>
            <Link
                to="/"
                className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-6 py-2"
            >
                Start Discovering
            </Link>
          </div>
        ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
              {likedWatches.map((watch, index) => (
                <LikedWatchCard key={watch.index || index} watch={watch} />
              ))}
            </div>
          )}
        </div>
      </main>
                  </div>
  );
};

interface LikedWatchCardProps {
  watch: Watch;
}

const LikedWatchCard: React.FC<LikedWatchCardProps> = ({ watch }) => {
  const price = getWatchPrice(watch);
  const displayPrice = formatPrice(price);
  const [imageError, setImageError] = useState(false);

  return (
    <div className="bg-card rounded-xl overflow-hidden shadow-sm border border-border hover:shadow-md transition-all hover:scale-[1.02] group">
      {/* Watch Image */}
      <div className="aspect-square bg-muted/30 flex items-center justify-center p-6 relative overflow-hidden">
        {!imageError ? (
        <img
          src={getWatchImageUrl(watch)}
          alt={`${watch.brand} ${watch.model}`}
            className="w-full h-full object-contain transition-transform group-hover:scale-105"
            onError={() => setImageError(true)}
        />
        ) : (
          <div className="flex flex-col items-center justify-center text-muted-foreground">
            <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-3">
              <Settings className="w-8 h-8" />
                  </div>
            <p className="text-sm text-center">{watch.brand}</p>
                </div>
        )}

        {/* Algorithm Badge */}
        {watch.algorithm && (
          <div className="absolute top-3 right-3 bg-card/95 backdrop-blur-sm text-card-foreground px-2 py-1 rounded-md text-xs font-mono border border-border shadow-sm">
            {watch.algorithm}
            {watch.confidence && (
              <span className="ml-1 text-green-600 dark:text-green-400">
                {watch.confidence.toFixed(1)}
              </span>
            )}
          </div>
        )}

        {/* Quick specs overlay */}
        <div className="absolute bottom-3 left-3 right-3">
          <div className="flex flex-wrap gap-1">
            {watch.specs?.diameter_mm && watch.specs.diameter_mm !== '-' && (
              <span className="bg-card/95 backdrop-blur-sm text-card-foreground text-xs px-2 py-1 rounded-full border border-border">
                {watch.specs.diameter_mm}mm
              </span>
            )}
            {watch.specs?.case_material && watch.specs.case_material !== '-' && (
              <span className="bg-card/95 backdrop-blur-sm text-card-foreground text-xs px-2 py-1 rounded-full border border-border">
                {watch.specs.case_material}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Watch Info */}
      <div className="p-4 space-y-3">
        <div>
          <h3 className="font-semibold text-foreground leading-tight">
            {watch.brand}
          </h3>
          <p className="text-sm text-muted-foreground">
            {watch.model}
          </p>
        </div>

        {/* Specifications */}
        <div className="space-y-2">
          {watch.specs?.movement && watch.specs.movement !== '-' && (
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">Movement</span>
              <span className="text-foreground font-medium">{watch.specs.movement}</span>
            </div>
          )}
          {watch.specs?.waterproofing_meters && watch.specs.waterproofing_meters !== '-' && (
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">Water Resistance</span>
              <span className="text-foreground font-medium">{watch.specs.waterproofing_meters}m</span>
            </div>
          )}
        </div>

        {/* Price */}
        {displayPrice && displayPrice !== 'Contact for price' && (
          <div className="pt-3 border-t border-border">
            <p className="text-lg font-semibold text-foreground">
              {displayPrice}
            </p>
          </div>
        )}

        {/* Links */}
        <div className="space-y-2 pt-2">
          {watch.product_url && (
            <a
              href={watch.product_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center text-xs text-primary hover:text-primary/80 transition-colors"
            >
              <ExternalLink className="w-3 h-3 mr-1" />
              View Product Details
            </a>
          )}
          
          {(watch.brand_website || watch.specs?.brand_website) && 
           (watch.brand_website !== '-' && watch.specs?.brand_website !== '-') && (
            <a
              href={watch.brand_website || watch.specs?.brand_website}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <ExternalLink className="w-3 h-3 mr-1" />
              Visit {watch.brand} Website
            </a>
          )}
        </div>
      </div>
    </div>
  );
};

export default LikedWatches;
