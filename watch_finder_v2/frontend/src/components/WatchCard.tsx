import React, { useState, useRef } from 'react';
import { Watch, getWatchImageUrl, formatPrice, getWatchPrice } from '../data/watchData';
import { Heart, X, Info, Layers, Copy } from 'lucide-react';

export interface WatchCardProps {
  watch: Watch;
  onSwipe: (direction: 'left' | 'right') => void;
  onSpecsClick: (watch: Watch) => void;
  onVariantsClick?: (watch: Watch) => void;
}

const WatchCard: React.FC<WatchCardProps> = ({
  watch,
  onSwipe,
  onSpecsClick,
  onVariantsClick
}) => {
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [isExiting, setIsExiting] = useState(false);
  const [exitDirection, setExitDirection] = useState<'left' | 'right' | null>(null);
  const [imageError, setImageError] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (isExiting) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || isExiting) return;
    
    const deltaX = e.clientX - dragStart.x;
    const deltaY = e.clientY - dragStart.y;
    setDragOffset({ x: deltaX, y: deltaY });
  };

  const handleMouseUp = () => {
    if (!isDragging || isExiting) return;
    
    setIsDragging(false);
    
    if (Math.abs(dragOffset.x) > 100) {
      // Prevent double-swipes by checking if already exiting
      if (isExiting) return;
      
      // Trigger exit animation
      const direction = dragOffset.x > 0 ? 'right' : 'left';
      setIsExiting(true);
      setExitDirection(direction);
      
      // Continue the swipe motion for smooth exit with faster timing
      const exitDistance = direction === 'right' ? 450 : -450;
      const exitY = dragOffset.y * 0.3 + (direction === 'right' ? -20 : 20); // Slight arc
      setDragOffset({ x: exitDistance, y: exitY });
      
      // Call onSwipe much faster to prevent gap
      setTimeout(() => {
        onSwipe(direction);
      }, 50); // Reduced from 120ms to 50ms
    } else {
      // Return to center if swipe wasn't strong enough
      setDragOffset({ x: 0, y: 0 });
    }
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    if (isExiting) return;
    setIsDragging(true);
    const touch = e.touches[0];
    setDragStart({ x: touch.clientX, y: touch.clientY });
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (!isDragging || isExiting) return;
    
    const touch = e.touches[0];
    const deltaX = touch.clientX - dragStart.x;
    const deltaY = touch.clientY - dragStart.y;
    setDragOffset({ x: deltaX, y: deltaY });
  };

  const handleTouchEnd = () => {
    if (!isDragging || isExiting) return;
    
    setIsDragging(false);
    
    if (Math.abs(dragOffset.x) > 100) {
      // Prevent double-swipes by checking if already exiting
      if (isExiting) return;
      
      // Trigger exit animation
      const direction = dragOffset.x > 0 ? 'right' : 'left';
      setIsExiting(true);
      setExitDirection(direction);
      
      // Continue the swipe motion for smooth exit with faster timing
      const exitDistance = direction === 'right' ? 450 : -450;
      const exitY = dragOffset.y * 0.3 + (direction === 'right' ? -20 : 20); // Slight arc
      setDragOffset({ x: exitDistance, y: exitY });
      
      // Call onSwipe much faster to prevent gap
      setTimeout(() => {
        onSwipe(direction);
      }, 50); // Reduced from 120ms to 50ms
    } else {
      // Return to center if swipe wasn't strong enough
      setDragOffset({ x: 0, y: 0 });
    }
  };

  const rotation = dragOffset.x * 0.1;
  const opacity = isExiting ? 0 : Math.max(0.7, 1 - Math.abs(dragOffset.x) * 0.003);

  // Enhanced transition timing with faster exit for seamless flow
  const getTransition = () => {
    if (isDragging) return 'none';
    if (isExiting) return 'all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94)'; // Faster, smoother exit
    return 'transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.3s ease-out'; // Smooth return
  };

  // Extract key info from our modern backend data
  const price = getWatchPrice(watch);
  const displayPrice = formatPrice(price);
  const diameter = watch.specs?.diameter_mm ? `${watch.specs.diameter_mm}mm` : '';
  const movement = watch.specs?.movement || watch.movement || '';

  return (
    <div
      ref={cardRef}
      className="relative w-full h-full cursor-grab active:cursor-grabbing select-none"
      style={{
        transform: `translateX(${dragOffset.x}px) translateY(${dragOffset.y * 0.1}px) rotate(${rotation}deg)`,
        opacity,
        transition: getTransition(),
        pointerEvents: isExiting ? 'none' : 'auto', // Disable interactions during exit
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      {/* Algorithm Badge */}
      {watch.algorithm && (
        <div className="absolute top-3 left-3 z-20 bg-card/95 backdrop-blur-sm text-card-foreground px-2 py-1 rounded-md text-xs font-mono border border-border shadow-sm">
          {watch.algorithm}
          {watch.confidence && (
            <span className="ml-1 text-green-600 dark:text-green-400">
              {watch.confidence.toFixed(1)}
            </span>
          )}
        </div>
      )}

      {/* Swipe Indicators */}
      {(Math.abs(dragOffset.x) > 50 || isExiting) && (
        <>
          <div
            className={`absolute inset-0 rounded-3xl z-10 flex items-center justify-center transition-all duration-300 ${
              (dragOffset.x > 0 || exitDirection === 'right') 
                ? 'bg-green-500/10 backdrop-blur-sm' 
                : 'bg-red-500/10 backdrop-blur-sm'
            }`}
            style={{
              opacity: isExiting ? 1 : Math.min(1, Math.abs(dragOffset.x) / 150)
            }}
          >
            <div
              className={`p-4 rounded-full border-4 transition-all duration-200 backdrop-blur-sm ${
                (dragOffset.x > 0 || exitDirection === 'right')
                  ? 'border-green-500 text-green-500 bg-green-500/10'
                  : 'border-red-500 text-red-500 bg-red-500/10'
              }`}
              style={{
                transform: isExiting 
                  ? 'scale(1.2)' 
                  : `scale(${Math.min(1.2, 0.8 + Math.abs(dragOffset.x) / 250)})`,
                filter: isExiting ? 'drop-shadow(0 0 20px currentColor)' : 'none'
              }}
            >
              {(dragOffset.x > 0 || exitDirection === 'right') ? (
                <Heart className="w-12 h-12" />
              ) : (
                <X className="w-12 h-12" />
              )}
            </div>
          </div>
        </>
      )}

      {/* Main Card */}
      <div className="bg-card rounded-3xl h-full overflow-hidden shadow-lg border border-border flex flex-col">
        {/* Watch Image */}
        <div className="relative flex-1 min-h-[60%] bg-muted/30 flex items-center justify-center overflow-hidden">
          {!imageError ? (
            <img
              src={getWatchImageUrl(watch)}
              alt={`${watch.brand} ${watch.model}`}
              className="w-full h-full object-contain p-4"
              onError={() => setImageError(true)}
            />
          ) : (
            <div className="flex flex-col items-center justify-center text-muted-foreground p-8">
              <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-3">
                <Info className="w-8 h-8" />
              </div>
              <p className="text-sm text-center">Image not available</p>
            </div>
          )}
          
          {/* Action Buttons */}
          <div className="absolute top-3 right-3 flex gap-2">
            {/* Variants Button */}
            {onVariantsClick && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onVariantsClick(watch);
                }}
                className="w-10 h-10 rounded-full bg-card/95 backdrop-blur-sm border border-border flex items-center justify-center hover:bg-accent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                title="View variants"
              >
                <Layers className="w-4 h-4 text-muted-foreground" />
              </button>
            )}
            
            {/* Info Button */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                onSpecsClick(watch);
              }}
              className="w-10 h-10 rounded-full bg-card/95 backdrop-blur-sm border border-border flex items-center justify-center hover:bg-accent transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              title="View specifications"
            >
              <Info className="w-4 h-4 text-muted-foreground" />
            </button>
          </div>
        </div>

        {/* Watch Details */}
        <div className="p-6 space-y-4 bg-card">
          {/* Brand and Model */}
          <div>
            <h3 className="text-xl font-semibold text-foreground leading-tight">
              {watch.brand}
            </h3>
            <p className="text-lg text-muted-foreground">
              {watch.model}
            </p>
          </div>

          {/* Specs Row */}
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-4">
              {diameter && (
                <span className="text-muted-foreground">
                  {diameter}
                </span>
              )}
              {movement && (
                <span className="text-muted-foreground">
                  {movement}
                </span>
              )}
            </div>
          </div>

          {/* Price */}
          {displayPrice && (
            <div className="pt-2 border-t border-border">
              <p className="text-2xl font-bold text-foreground">
                {displayPrice}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default WatchCard;
