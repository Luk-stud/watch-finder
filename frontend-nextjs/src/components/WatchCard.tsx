'use client';

import React from 'react';
import { useState, useEffect } from 'react';
import { motion, PanInfo, AnimatePresence } from 'framer-motion';
import Image from 'next/image';
import { Heart, X, Droplets, Clock, Settings, Diamond, Info, ChevronDown } from 'lucide-react';
import type { Watch, ViewType } from '@/types';
import { formatPrice, getImageUrl } from '@/lib/utils';
import Navigation from './Navigation';

interface WatchCardProps {
  watch: Watch;
  onLike: () => void;
  onPass: () => void;
  isActive?: boolean;
  onShowSeries?: (watchIndex: number) => void;
  currentView: ViewType;
  likedCount: number;
  onViewChange: (view: ViewType) => void;
  onMenuClose?: () => void;
}

export default function WatchCard({ 
  watch, 
  onLike, 
  onPass, 
  isActive = true, 
  onShowSeries, 
  currentView,
  likedCount,
  onViewChange,
  onMenuClose 
}: WatchCardProps) {
  const [dragX, setDragX] = useState(0);
  const [imageError, setImageError] = useState(false);
  const [useNextImage, setUseNextImage] = useState(true);
  const [isMobile, setIsMobile] = useState(false);
  const [showDetailsOverlay, setShowDetailsOverlay] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const handleDrag = (_: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    setDragX(info.offset.x);
  };

  const handleDragEnd = (_: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    const threshold = 100;
    
    if (info.offset.x > threshold) {
      onLike();
    } else if (info.offset.x < -threshold) {
      onPass();
    }
    
    setDragX(0);
  };

  const getWatchImageUrl = (watch: Watch) => {
    // Use the new local image function
    return getImageUrl(watch);
  };

  const handleImageError = () => {
    console.log('Image error for:', getWatchImageUrl(watch));
    if (useNextImage) {
      setUseNextImage(false);
    } else {
      setImageError(true);
    }
  };

  const getSwipeIndicator = () => {
    const intensity = Math.abs(dragX) / 100;
    
    if (dragX > 50) {
      return (
        <motion.div
          className="absolute top-6 right-6 bg-green-500 text-white px-4 py-2 rounded-xl text-lg font-bold z-20 border-4 border-green-300"
          initial={{ scale: 0, rotate: -12 }}
          animate={{ scale: 1 + intensity * 0.2, rotate: -12 }}
          style={{ 
            boxShadow: '0 8px 32px rgba(34, 197, 94, 0.4)',
          }}
        >
          LIKE
        </motion.div>
      );
    } else if (dragX < -50) {
      return (
        <motion.div
          className="absolute top-6 left-6 bg-red-500 text-white px-4 py-2 rounded-xl text-lg font-bold z-20 border-4 border-red-300"
          initial={{ scale: 0, rotate: 12 }}
          animate={{ scale: 1 + intensity * 0.2, rotate: 12 }}
          style={{ 
            boxShadow: '0 8px 32px rgba(239, 68, 68, 0.4)',
          }}
        >
          PASS
        </motion.div>
      );
    }
    return null;
  };

  // Helper function to format price from new structure
  const getFormattedPrice = () => {
    if (watch.specs?.price_usd && watch.specs.price_usd !== '-') {
      return `$${watch.specs.price_usd}`;
    }
    if (watch.specs?.msrp_eur && watch.specs.msrp_eur !== '-') {
      return `€${watch.specs.msrp_eur}`;
    }
    if (watch.price) {
      return formatPrice(watch.price);
    }
    return 'Contact for price';
  };

  // Helper function to get brand and model
  const getBrandModel = () => {
    const brand = watch.specs?.brand || watch.brand;
    const model = watch.specs?.model || watch.model;
    return { brand, model };
  };

  // Helper function to get complications
  const getComplications = () => {
    if (!watch.specs) return [];
    
    const complications = [];
    if (watch.specs.complication_date && watch.specs.complication_date !== '-') complications.push('Date');
    if (watch.specs.complication_chronograph && watch.specs.complication_chronograph !== '-') complications.push('Chronograph');
    if (watch.specs.complication_gmt && watch.specs.complication_gmt !== '-') complications.push('GMT');
    if (watch.specs.complication_dual_time && watch.specs.complication_dual_time !== '-') complications.push('Dual Time');
    if (watch.specs.complication_power_reserve && watch.specs.complication_power_reserve !== '-') complications.push('Power Reserve');
    if (watch.specs.complication_moonphase && watch.specs.complication_moonphase !== '-') complications.push('Moonphase');
    
    return complications;
  };

  // Helper function to check if series has multiple watches (could be enhanced with actual count)
  const hasSeriesWatches = () => {
    const result = watch.specs?.serie && watch.specs.serie !== '-' && watch.specs.serie !== 'All';
    return result;
  };

  const { brand, model } = getBrandModel();
  const complications = getComplications();

  return (
    <>
      <motion.div
        className="bg-white rounded-3xl shadow-2xl overflow-hidden select-none w-full mx-auto flex flex-col relative"
        style={{ 
          touchAction: 'pan-y',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(255, 255, 255, 0.05)',
          marginBottom: '2rem',
          height: 'calc(100dvh - env(safe-area-inset-top) - env(safe-area-inset-bottom) - 4rem)',
          maxHeight: 'calc(-webkit-fill-available - env(safe-area-inset-top) - env(safe-area-inset-bottom) - 4rem)',
          position: 'absolute',
          left: 0,
          right: 0,
          top: 0
        }}
        drag={isActive ? "x" : false}
        dragConstraints={{ left: 0, right: 0 }}
        dragElastic={0.7}
        onDrag={handleDrag}
        onDragEnd={handleDragEnd}
        animate={{
          x: dragX,
          rotate: dragX * 0.05,
          scale: 1 - Math.abs(dragX) * 0.0002,
          opacity: Math.max(0.7, 1 - Math.abs(dragX) / 300),
        }}
        initial={{ 
          scale: 0.8,
          opacity: 0,
          y: 50,
        }}
        exit={{ 
          scale: 0.8,
          opacity: 0,
          y: dragX > 50 ? -100 : dragX < -50 ? 100 : 0,
          x: dragX > 50 ? 100 : dragX < -50 ? -100 : 0,
          transition: { duration: 0.2, ease: "easeOut" }
        }}
        transition={{ 
          type: "spring",
          stiffness: 400,
          damping: 40,
          mass: 0.8,
        }}
        layout
      >
        {/* Swipe Indicators */}
        {getSwipeIndicator()}

        {/* Full Image with Overlaid Info */}
        <motion.div 
          className="w-full h-full overflow-hidden relative flex flex-col"
          style={{ backgroundColor: '#f1f2f0' }}
          animate={{
            scale: 1 + Math.abs(dragX) * 0.0001,
          }}
          transition={{ duration: 0.2 }}
        >
          {/* Image Container */}
          <div className="relative flex-1">
            {useNextImage && !imageError ? (
              <motion.div
                className="w-full h-full"
                animate={{
                  rotate: dragX * 0.02,
                }}
                transition={{ duration: 0.2 }}
              >
                <Image
                  src={getWatchImageUrl(watch)}
                  alt={`${brand} ${model}`}
                  width={400}
                  height={400}
                  className="object-cover w-full h-full"
                  sizes="(max-width: 640px) 280px, 400px"
                  onError={handleImageError}
                  priority={isActive}
                  style={{ 
                    objectFit: 'cover'
                  }}
                />
              </motion.div>
            ) : (
              <motion.img
                src={getWatchImageUrl(watch)}
                alt={`${brand} ${model}`}
                className="w-full h-full object-cover"
                onError={handleImageError}
                style={{ 
                  objectFit: 'cover'
                }}
                animate={{
                  rotate: dragX * 0.02,
                }}
                transition={{ duration: 0.2 }}
              />
            )}

            {/* Enhanced gradient overlay for text visibility */}
            <div 
              className="absolute inset-0 pointer-events-none"
              style={{
                background: 'linear-gradient(to top, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.4) 30%, transparent 60%)'
              }}
            />

            {/* Overlaid Essential Info - Text content area (bottom left) */}
            <div className="absolute bottom-0 left-0 z-20 p-4 text-left">
              {/* Brand & Model */}
              <div className="mb-1">
                <h2 className="text-2xl font-bold mb-0.5 truncate text-white">{brand}</h2>
                <p className="text-lg truncate text-white/90">{model}</p>
                {watch.specs?.serie && watch.specs.serie !== '-' && (
                  <p className="text-sm font-medium px-2 py-1 rounded-full inline-block mt-1 text-xs text-white" style={{ backgroundColor: '#558a86' }}>
                    {watch.specs.serie}
                  </p>
                )}
              </div>

              {/* Diameter Text Display */}
              {(watch.specs?.diameter_mm || watch.case_diameter) && (
                <div className="mt-1 text-xs text-white/80">
                  <span>Diameter: {watch.specs?.diameter_mm || watch.case_diameter}mm</span>
                </div>
              )}

              {/* Price */}
              <div className="mt-1.5 text-xl font-bold text-white">
                {getFormattedPrice()}
              </div>
            </div>
          </div>

          {/* Buttons below image */}
          <div className="flex gap-2 p-4 bg-white/5 backdrop-blur-sm">
            {/* Specs Button */}
            <button
              onClick={() => setShowDetailsOverlay(true)}
              className="flex-1 flex items-center justify-center gap-2 text-white px-4 py-3 rounded-xl text-sm font-medium hover:opacity-80 transition-all duration-200 shadow-md"
              style={{ backgroundColor: '#558a86' }}
            >
              <Info className="w-4 h-4" /> 
              <span>SPECS</span>
            </button>

            {/* Series Button */}
            {hasSeriesWatches() && onShowSeries && (
              <button
                onClick={() => onShowSeries?.(watch.index)}
                className="flex-1 flex items-center justify-center gap-2 text-white px-4 py-3 rounded-xl text-sm font-medium hover:opacity-80 transition-all duration-200 shadow-md"
                style={{ backgroundColor: '#a63e14' }}
              >
                <Settings className="w-4 h-4" />
                <span>SERIES</span>
              </button>
            )}
          </div>
        </motion.div>
      </motion.div>

      {/* Details Overlay Modal */}
      <AnimatePresence>
        {showDetailsOverlay && (
          <motion.div
            className="fixed inset-0 z-50 flex items-end justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {/* Backdrop */}
            <div 
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
              onClick={() => setShowDetailsOverlay(false)}
            />
            
            {/* Modal Content */}
            <motion.div
              className="relative rounded-t-3xl w-full max-w-sm mx-auto max-h-[70vh] overflow-hidden"
              style={{ backgroundColor: '#f1f2f0' }}
              initial={{ y: "100%" }}
              animate={{ y: 0 }}
              exit={{ y: "100%" }}
              transition={{ type: "spring", damping: 25, stiffness: 200 }}
            >
              {/* Modal Header */}
              <div className="text-white p-6 sticky top-0 z-10" style={{ backgroundColor: '#558a86' }}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold">Watch Details</h3>
                  <button
                    onClick={() => setShowDetailsOverlay(false)}
                    className="text-white hover:text-white/80 transition-colors p-2 rounded-full hover:bg-white/20"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
                <div className="flex items-center gap-2 text-sm text-white/80">
                  <ChevronDown className="w-4 h-4" />
                  <span>Swipe down to close</span>
                </div>
              </div>

              {/* Scrollable Content */}
              <div className="p-6 space-y-6 overflow-y-auto max-h-[calc(70vh-120px)] overscroll-behavior-y-contain">
                {/* Availability */}
                {watch.specs?.availability && (
                  <div className="flex items-center justify-between p-4 rounded-2xl border" style={{ backgroundColor: '#f1f2f0', borderColor: '#558a86' }}>
                    <span className="font-medium" style={{ color: '#35342f' }}>Availability</span>
                    <span className={`px-3 py-2 rounded-xl text-sm font-semibold shadow-sm text-white`} 
                      style={{ 
                        backgroundColor: watch.specs.availability === 'Yes' 
                          ? '#558a86'
                          : watch.specs.availability === 'Out of Stock'
                          ? '#a63e14'
                          : '#bfbabe'
                      }}>
                      {watch.specs.availability === 'Yes' ? 'Available' : watch.specs.availability}
                    </span>
                  </div>
                )}

                {/* Key Specifications */}
                <div className="p-4 rounded-2xl shadow-sm border" style={{ backgroundColor: '#f1f2f0', borderColor: '#bfbabe' }}>
                  <h4 className="text-lg font-semibold mb-3" style={{ color: '#35342f' }}>Specifications</h4>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    {/* Water Resistance */}
                    {(watch.specs?.waterproofing_meters || watch.water_resistance) && (
                      <div className="flex items-center gap-2">
                        <Droplets className="w-4 h-4" style={{ color: '#558a86' }} />
                        <div>
                          <span style={{ color: '#35342f' }}>WR:</span>
                          <span className="ml-1 font-medium" style={{ color: '#35342f' }}>
                            {watch.specs?.waterproofing_meters || watch.water_resistance}m
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Movement */}
                    {(watch.specs?.movement || watch.movement) && (
                      <div className="flex items-center gap-2 col-span-2">
                        <Clock className="w-4 h-4" style={{ color: '#558a86' }} />
                        <div>
                          <span style={{ color: '#35342f' }}>Movement:</span>
                          <span className="ml-1 font-medium text-xs" style={{ color: '#35342f' }}>
                            {watch.specs?.movement || watch.movement}
                          </span>
                        </div>
                      </div>
                    )}

                    {/* Case Material */}
                    {(watch.specs?.case_material || watch.case_material) && (
                      <div className="flex items-center gap-2 col-span-2">
                        <Diamond className="w-4 h-4" style={{ color: '#558a86' }} />
                        <div>
                          <span style={{ color: '#35342f' }}>Material:</span>
                          <span className="ml-1 font-medium text-xs" style={{ color: '#35342f' }}>
                            {watch.specs?.case_material || watch.case_material}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Additional Details */}
                {watch.specs && (
                  <div className="p-4 rounded-2xl shadow-sm border" style={{ backgroundColor: '#f1f2f0', borderColor: '#e1e0dd' }}>
                    <h4 className="text-lg font-semibold mb-3" style={{ color: '#35342f' }}>Details</h4>
                    
                    {/* Crystal & Dial Info */}
                    <div className="grid grid-cols-2 gap-3 text-xs mb-4">
                      {watch.specs.crystal_material && watch.specs.crystal_material !== '-' && (
                        <div className="p-2 rounded-lg border" style={{ backgroundColor: '#f1f2f0', borderColor: '#e1e0dd' }}>
                          <span className="block text-xs" style={{ color: '#35342f' }}>Crystal:</span>
                          <span className="font-medium" style={{ color: '#35342f' }}>{watch.specs.crystal_material}</span>
                        </div>
                      )}
                      {watch.specs.dial_color && watch.specs.dial_color !== '-' && (
                        <div className="p-2 rounded-lg border" style={{ backgroundColor: '#f1f2f0', borderColor: '#e1e0dd' }}>
                          <span className="block text-xs" style={{ color: '#35342f' }}>Dial:</span>
                          <span className="font-medium" style={{ color: '#35342f' }}>{watch.specs.dial_color}</span>
                        </div>
                      )}
                      {watch.specs.case_finishing && watch.specs.case_finishing !== '-' && (
                        <div className="p-2 rounded-lg border" style={{ backgroundColor: '#f1f2f0', borderColor: '#e1e0dd' }}>
                          <span className="block text-xs" style={{ color: '#35342f' }}>Finish:</span>
                          <span className="font-medium" style={{ color: '#35342f' }}>{watch.specs.case_finishing}</span>
                        </div>
                      )}
                      {watch.specs.power_reserve_hour && watch.specs.power_reserve_hour !== '-' && (
                        <div className="p-2 rounded-lg border" style={{ backgroundColor: '#f1f2f0', borderColor: '#e1e0dd' }}>
                          <span className="block text-xs" style={{ color: '#35342f' }}>Power:</span>
                          <span className="font-medium" style={{ color: '#35342f' }}>{watch.specs.power_reserve_hour}h</span>
                        </div>
                      )}
                    </div>

                    {/* Complications */}
                    {complications.length > 0 && (
                      <div className="mb-4">
                        <span className="text-sm font-medium block mb-2" style={{ color: '#35342f' }}>Complications:</span>
                        <div className="flex flex-wrap gap-2">
                          {complications.map((comp, index) => (
                            <span
                              key={index}
                              className="px-3 py-1 text-white rounded-full text-xs font-medium shadow-lg"
                              style={{ backgroundColor: '#558a86' }}
                            >
                              {comp}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Watch Type */}
                    {watch.specs.watch_type && watch.specs.watch_type !== '-' && (
                      <div className="p-3 rounded-lg border" style={{ backgroundColor: '#f1f2f0', borderColor: '#e1e0dd' }}>
                        <span className="text-xs block" style={{ color: '#35342f' }}>Type:</span>
                        <span className="font-medium" style={{ color: '#35342f' }}>{watch.specs.watch_type}</span>
                      </div>
                    )}
                  </div>
                )}
                
                {/* Bottom padding */}
                <div className="h-4"></div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
} 