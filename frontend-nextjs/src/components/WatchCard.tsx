'use client';

import React from 'react';
import { useState } from 'react';
import { motion, PanInfo } from 'framer-motion';
import Image from 'next/image';
import { Heart, X, Droplets, Clock, Settings, Diamond } from 'lucide-react';
import type { Watch } from '@/types';
import { formatPrice, getImageUrl } from '@/lib/utils';

interface WatchCardProps {
  watch: Watch;
  onLike: () => void;
  onPass: () => void;
  isActive?: boolean;
  onShowSeries?: (watchIndex: number) => void;
}

export default function WatchCard({ watch, onLike, onPass, isActive = true, onShowSeries }: WatchCardProps) {
  const [dragX, setDragX] = useState(0);
  const [imageError, setImageError] = useState(false);
  const [useNextImage, setUseNextImage] = useState(true);

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
    if (dragX > 50) {
      return (
        <motion.div
          className="absolute top-4 right-4 bg-green-500 text-white px-3 py-1 rounded-full text-sm font-medium z-10"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
        >
          <Heart className="w-4 h-4" />
        </motion.div>
      );
    } else if (dragX < -50) {
      return (
        <motion.div
          className="absolute top-4 left-4 bg-red-500 text-white px-3 py-1 rounded-full text-sm font-medium z-10"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
        >
          <X className="w-4 h-4" />
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
    <motion.div
      className="bg-white rounded-2xl shadow-lg overflow-hidden select-none"
      drag={isActive ? "x" : false}
      dragConstraints={{ left: 0, right: 0 }}
      dragElastic={0.7}
      onDrag={handleDrag}
      onDragEnd={handleDragEnd}
      animate={{
        x: dragX,
        rotate: dragX * 0.1,
        opacity: Math.max(0.5, 1 - Math.abs(dragX) / 200),
      }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      style={{ touchAction: 'pan-y' }}
    >
      {/* Swipe Indicators */}
      {getSwipeIndicator()}

      {/* Image Section */}
      <div className="flex items-center justify-center w-full h-80 bg-gradient-to-br from-gray-50 to-gray-100 overflow-hidden" style={{ height: '320px' }}>
        {useNextImage && !imageError ? (
          <Image
            src={getWatchImageUrl(watch)}
            alt={`${brand} ${model}`}
            width={400}
            height={320}
            className="object-contain"
            sizes="(max-width: 768px) 100vw, 400px"
            onError={handleImageError}
            priority={isActive}
            style={{ objectFit: 'contain' }}
          />
        ) : (
          <img
            src={getWatchImageUrl(watch)}
            alt={`${brand} ${model}`}
            className="w-full h-full object-contain p-8"
            onError={handleImageError}
            style={{ objectFit: 'contain' }}
          />
        )}
      </div>

      {/* Content Section */}
      <div className="p-6 space-y-4">
        {/* Brand & Model */}
        <div>
          <h2 className="text-xl font-bold text-gray-900 mb-1">{brand}</h2>
          <p className="text-lg text-gray-600">{model}</p>
          {watch.specs?.serie && watch.specs.serie !== '-' && (
            <p className="text-sm text-gray-500">{watch.specs.serie}</p>
          )}
        </div>

        {/* Price & Availability */}
        <div className="flex items-center justify-between">
          <div className="text-2xl font-bold text-green-600">
            {getFormattedPrice()}
          </div>
          {watch.specs?.availability && (
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
              watch.specs.availability === 'Yes' 
                ? 'bg-green-100 text-green-800' 
                : watch.specs.availability === 'Out of Stock'
                ? 'bg-red-100 text-red-800'
                : 'bg-gray-100 text-gray-800'
            }`}>
              {watch.specs.availability === 'Yes' ? 'Available' : watch.specs.availability}
            </span>
          )}
        </div>

        {/* Key Specifications Grid */}
        <div className="grid grid-cols-2 gap-3 text-sm">
          {/* Case Size */}
          {(watch.specs?.diameter_mm || watch.case_diameter) && (
            <div className="flex items-center gap-2">
              <Settings className="w-4 h-4 text-gray-400" />
              <div>
                <span className="text-gray-500">Size:</span>
                <span className="ml-1 font-medium">
                  {watch.specs?.diameter_mm || watch.case_diameter}mm
                </span>
              </div>
            </div>
          )}

          {/* Water Resistance */}
          {(watch.specs?.waterproofing_meters || watch.water_resistance) && (
            <div className="flex items-center gap-2">
              <Droplets className="w-4 h-4 text-blue-400" />
              <div>
                <span className="text-gray-500">WR:</span>
                <span className="ml-1 font-medium">
                  {watch.specs?.waterproofing_meters || watch.water_resistance}m
                </span>
              </div>
            </div>
          )}

          {/* Movement */}
          {(watch.specs?.movement || watch.movement) && (
            <div className="flex items-center gap-2 col-span-2">
              <Clock className="w-4 h-4 text-orange-400" />
              <div>
                <span className="text-gray-500">Movement:</span>
                <span className="ml-1 font-medium text-xs">
                  {watch.specs?.movement || watch.movement}
                </span>
              </div>
            </div>
          )}

          {/* Case Material */}
          {(watch.specs?.case_material || watch.case_material) && (
            <div className="flex items-center gap-2 col-span-2">
              <Diamond className="w-4 h-4 text-purple-400" />
              <div>
                <span className="text-gray-500">Material:</span>
                <span className="ml-1 font-medium text-xs">
                  {watch.specs?.case_material || watch.case_material}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Additional Details from new structure */}
        {watch.specs && (
          <div className="space-y-3 pt-2 border-t border-gray-100">
            
            {/* Crystal & Dial Info */}
            <div className="grid grid-cols-2 gap-3 text-xs">
              {watch.specs.crystal_material && watch.specs.crystal_material !== '-' && (
                <div>
                  <span className="text-gray-500">Crystal:</span>
                  <span className="ml-1 font-medium">{watch.specs.crystal_material}</span>
                </div>
              )}
              {watch.specs.dial_color && watch.specs.dial_color !== '-' && (
                <div>
                  <span className="text-gray-500">Dial:</span>
                  <span className="ml-1 font-medium">{watch.specs.dial_color}</span>
                </div>
              )}
              {watch.specs.case_finishing && watch.specs.case_finishing !== '-' && (
                <div>
                  <span className="text-gray-500">Finish:</span>
                  <span className="ml-1 font-medium">{watch.specs.case_finishing}</span>
                </div>
              )}
              {watch.specs.power_reserve_hour && watch.specs.power_reserve_hour !== '-' && (
                <div>
                  <span className="text-gray-500">Power:</span>
                  <span className="ml-1 font-medium">{watch.specs.power_reserve_hour}h</span>
                </div>
              )}
            </div>

            {/* Complications */}
            {complications.length > 0 && (
              <div>
                <span className="text-gray-500 text-xs">Complications:</span>
                <div className="flex flex-wrap gap-1 mt-1">
                  {complications.map((comp, index) => (
                    <span
                      key={index}
                      className="px-2 py-1 bg-blue-50 text-blue-700 rounded-full text-xs font-medium"
                    >
                      {comp}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Watch Type */}
            {watch.specs.watch_type && watch.specs.watch_type !== '-' && (
              <div className="text-xs">
                <span className="text-gray-500">Type:</span>
                <span className="ml-1 font-medium text-gray-700">{watch.specs.watch_type}</span>
              </div>
            )}
          </div>
        )}

        {/* Swipe Instructions */}
        <div className="text-center text-xs text-gray-400 mt-4 pt-2 border-t border-gray-100">
          Swipe right to like • Swipe left to pass
        </div>

        {/* Series Button */}
        {hasSeriesWatches() && onShowSeries && (
          <button
            onClick={() => onShowSeries?.(watch.index)}
            className="w-full mt-3 px-4 py-2 bg-gradient-to-r from-blue-50 to-purple-50 hover:from-blue-100 hover:to-purple-100 text-blue-700 rounded-lg text-sm font-medium transition-all duration-200 flex items-center justify-center gap-2"
          >
            <Settings className="w-4 h-4" />
            <span>View Series ({watch.specs?.serie})</span>
            <span className="text-xs bg-blue-200 text-blue-800 px-2 py-0.5 rounded-full">More</span>
          </button>
        )}
      </div>
    </motion.div>
  );
} 