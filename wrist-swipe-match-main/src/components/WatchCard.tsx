import React, { useState, useRef } from 'react';
import { Watch, getWatchImageUrl, formatPrice, getWatchPrice } from '../data/watchData';
import { Heart, X, Info, Layers, Cpu } from 'lucide-react';

interface WatchCardProps {
  watch: Watch;
  onSwipe: (direction: 'left' | 'right') => void;
  onSpecsClick: (watch: Watch) => void;
  onSeriesClick: (watch: Watch) => void;
}

const WatchCard: React.FC<WatchCardProps> = ({
  watch,
  onSwipe,
  onSpecsClick,
  onSeriesClick,
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
  const series = watch.specs?.serie || watch.brand;
  const hasValidSeries = series && series !== '-' && series !== 'All';
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
      {/* Modern Backend Debug Badge */}
      {watch.algorithm && (
        <div className="absolute top-2 left-2 z-20 bg-black/70 text-white px-2 py-1 rounded-md text-xs font-mono">
          {watch.algorithm}
          {watch.confidence && (
            <span className="ml-1 text-green-300">
              {watch.confidence.toFixed(1)}
            </span>
          )}
          {watch.is_seed && (
            <span className="ml-1 text-blue-300">SEED</span>
          )}
        </div>
      )}

      {/* Swipe Indicators */}
      {(Math.abs(dragOffset.x) > 50 || isExiting) && (
        <>
          <div
            className={`absolute inset-0 rounded-3xl z-10 flex items-center justify-center transition-all duration-300 ${
              (dragOffset.x > 0 || exitDirection === 'right') ? 'bg-green-500/20' : 'bg-red-500/20'
            }`}
            style={{
              opacity: isExiting ? 1 : Math.min(1, Math.abs(dragOffset.x) / 150)
            }}
          >
            <div
              className={`p-4 rounded-full border-4 transition-all duration-200 ${
                (dragOffset.x > 0 || exitDirection === 'right')
                  ? 'border-green-500 text-green-500'
                  : 'border-red-500 text-red-500'
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
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-3xl h-full overflow-hidden shadow-2xl border border-gray-700 flex flex-col">
        {/* Watch Image */}
        <div className="relative flex-1 min-h-[60%] bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center overflow-hidden">
          {!imageError ? (
            <img
              src={getWatchImageUrl(watch)}
              alt={`${watch.brand} ${watch.model}`}
              className="w-full h-full object-contain p-4"
              onError={() => setImageError(true)}
              style={{ 
                filter: 'drop-shadow(0 10px 20px rgba(0,0,0,0.3))',
                maxWidth: '90%',
                maxHeight: '90%'
              }}
            />
          ) : (
            // Fallback when image fails to load
            <div className="w-48 h-48 bg-gradient-to-br from-gray-300 to-gray-500 rounded-full shadow-2xl flex items-center justify-center">
              <div className="w-40 h-40 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-full flex items-center justify-center">
                <div className="text-black font-bold text-lg text-center">
                  {watch.brand}
                  <div className="text-sm font-normal">{watch.model}</div>
                </div>
              </div>
            </div>
          )}
          
          {/* Action Buttons */}
          <div className="absolute top-4 right-4 flex flex-col gap-2">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onSpecsClick(watch);
              }}
              className="w-10 h-10 bg-black/50 backdrop-blur-sm rounded-full flex items-center justify-center text-white hover:bg-black/70 transition-colors"
            >
              <Info className="w-5 h-5" />
            </button>
            {hasValidSeries && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onSeriesClick(watch);
                }}
                className="w-10 h-10 bg-black/50 backdrop-blur-sm rounded-full flex items-center justify-center text-white hover:bg-black/70 transition-colors"
              >
                <Layers className="w-5 h-5" />
              </button>
            )}
          </div>

          {/* Watch specifications overlay */}
          <div className="absolute bottom-4 left-4 right-4">
            <div className="flex flex-wrap gap-2">
              {diameter && (
                <span className="bg-black/50 backdrop-blur-sm text-white text-xs px-2 py-1 rounded-full">
                  {diameter}
                </span>
              )}
              {watch.specs?.case_material && watch.specs.case_material !== '-' && (
                <span className="bg-black/50 backdrop-blur-sm text-white text-xs px-2 py-1 rounded-full">
                  {watch.specs.case_material}
                </span>
              )}
              {watch.specs?.waterproofing_meters && watch.specs.waterproofing_meters !== '-' && (
                <span className="bg-black/50 backdrop-blur-sm text-white text-xs px-2 py-1 rounded-full">
                  {watch.specs.waterproofing_meters}m WR
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Watch Info */}
        <div className="p-4 md:p-6 flex-shrink-0 flex flex-col justify-between min-h-[40%] max-h-[40%] overflow-hidden">
          <div className="flex-1 min-h-0">
            <h2 className="text-xl md:text-2xl font-bold text-white mb-1 truncate">{watch.model}</h2>
            <p className="text-yellow-400 font-semibold mb-1 truncate">{watch.brand}</p>
            {hasValidSeries && (
              <p className="text-gray-400 text-sm mb-2 truncate">{series} Series</p>
            )}
            {movement && movement !== '-' && (
              <p className="text-gray-400 text-xs truncate mb-2">{movement}</p>
            )}
          </div>
          
          <div className="flex justify-between items-end mt-auto">
            <span className="text-xl md:text-2xl font-bold text-white truncate">{displayPrice}</span>
            {watch.specs?.launch_date && (
              <span className="text-gray-400 text-sm flex-shrink-0 ml-2">{watch.specs.launch_date}</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default WatchCard;
