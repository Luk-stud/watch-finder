'use client';

import React from 'react';
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Image from 'next/image';
import { X, Settings, Droplets, Clock, Diamond, Heart, Eye } from 'lucide-react';
import type { Watch } from '@/types';
import { formatPrice, getImageUrl } from '@/lib/utils';

interface SeriesModalProps {
  isOpen: boolean;
  onClose: () => void;
  seriesWatches: Watch[];
  currentWatchIndex: number;
  seriesName: string;
  onSelectWatch?: (watch: Watch) => void;
  onLikeWatch?: (watch: Watch) => void;
}

export default function SeriesModal({
  isOpen,
  onClose,
  seriesWatches,
  currentWatchIndex,
  seriesName,
  onSelectWatch,
  onLikeWatch
}: SeriesModalProps) {
  const [selectedWatch, setSelectedWatch] = useState<Watch | null>(null);

  useEffect(() => {
    if (isOpen && seriesWatches.length > 0) {
      const current = seriesWatches.find(w => w.index === currentWatchIndex);
      setSelectedWatch(current || seriesWatches[0]);
    }
  }, [isOpen, seriesWatches, currentWatchIndex]);

  if (!isOpen) return null;

  const getWatchImageUrl = (watch: Watch) => {
    return getImageUrl(watch);
  };

  const getFormattedPrice = (watch: Watch) => {
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

  const getAvailabilityColor = (availability: string) => {
    switch (availability) {
      case 'Yes':
        return 'bg-green-100 text-green-800';
      case 'Out of Stock':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4"
        style={{ zIndex: 999999 }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      >
        <motion.div
          className="bg-white rounded-2xl max-w-6xl max-h-[90vh] overflow-hidden shadow-2xl"
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold">Series: {seriesName}</h2>
                <p className="text-blue-100">{seriesWatches.length} watches in this series</p>
              </div>
              <button
                onClick={onClose}
                className="p-2 hover:bg-white/10 rounded-full transition-colors"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
          </div>

          <div className="flex flex-col lg:flex-row max-h-[75vh]">
            {/* Watch Grid */}
            <div className="lg:w-1/2 p-6 overflow-y-auto">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {seriesWatches.map((watch) => (
                  <motion.div
                    key={watch.index}
                    className={`border rounded-xl p-4 cursor-pointer transition-all ${
                      selectedWatch?.index === watch.index
                        ? 'border-blue-500 bg-blue-50'
                        : watch.index === currentWatchIndex
                        ? 'border-green-500 bg-green-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setSelectedWatch(watch)}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    {/* Current watch indicator */}
                    {watch.index === currentWatchIndex && (
                      <div className="flex items-center gap-1 text-green-600 text-xs font-medium mb-2">
                        <Eye className="w-3 h-3" />
                        Currently viewing
                      </div>
                    )}

                    {/* Watch Image */}
                    <div className="flex items-center justify-center w-full h-32 bg-gray-50 rounded-lg mb-3 overflow-hidden">
                      <Image
                        src={getWatchImageUrl(watch)}
                        alt={`${watch.specs?.brand} ${watch.specs?.model}`}
                        width={200}
                        height={128}
                        className="object-contain max-w-full max-h-full w-full h-full"
                        sizes="(max-width: 640px) 150px, 200px"
                      />
                    </div>

                    {/* Watch Info */}
                    <div>
                      <h4 className="font-medium text-gray-900 text-sm mb-1">
                        {watch.specs?.model || watch.model}
                      </h4>
                      <p className="text-xs text-gray-500 mb-2">
                        {watch.specs?.reference && watch.specs.reference !== '-' && (
                          `Ref: ${watch.specs.reference}`
                        )}
                      </p>
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-semibold text-green-600">
                          {getFormattedPrice(watch)}
                        </span>
                        {watch.specs?.availability && (
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getAvailabilityColor(watch.specs.availability)}`}>
                            {watch.specs.availability === 'Yes' ? 'Available' : watch.specs.availability}
                          </span>
                        )}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Selected Watch Details */}
            {selectedWatch && (
              <div className="lg:w-1/2 bg-gray-50 p-6 overflow-y-auto">
                <div className="bg-white rounded-xl p-6 shadow-sm">
                  {/* Large Image */}
                  <div className="flex items-center justify-center w-full h-64 bg-gray-100 rounded-lg mb-6 overflow-hidden">
                    <Image
                      src={getWatchImageUrl(selectedWatch)}
                      alt={`${selectedWatch.specs?.brand} ${selectedWatch.specs?.model}`}
                      width={400}
                      height={256}
                      className="object-contain"
                      sizes="400px"
                    />
                  </div>

                  {/* Watch Details */}
                  <div className="space-y-4">
                    <div>
                      <h3 className="text-xl font-bold text-gray-900">
                        {selectedWatch.specs?.brand}
                      </h3>
                      <p className="text-lg text-gray-600">
                        {selectedWatch.specs?.model || selectedWatch.model}
                      </p>
                      {selectedWatch.specs?.reference && selectedWatch.specs.reference !== '-' && (
                        <p className="text-sm text-gray-500">Ref: {selectedWatch.specs.reference}</p>
                      )}
                    </div>

                    <div className="text-2xl font-bold text-green-600">
                      {getFormattedPrice(selectedWatch)}
                    </div>

                    {/* Specifications Grid */}
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      {selectedWatch.specs?.diameter_mm && (
                        <div className="flex items-center gap-2">
                          <Settings className="w-4 h-4 text-gray-400" />
                          <div>
                            <span className="text-gray-500">Size:</span>
                            <span className="ml-1 font-medium">{selectedWatch.specs.diameter_mm}mm</span>
                          </div>
                        </div>
                      )}

                      {selectedWatch.specs?.waterproofing_meters && selectedWatch.specs.waterproofing_meters !== '-' && (
                        <div className="flex items-center gap-2">
                          <Droplets className="w-4 h-4 text-blue-400" />
                          <div>
                            <span className="text-gray-500">WR:</span>
                            <span className="ml-1 font-medium">{selectedWatch.specs.waterproofing_meters}m</span>
                          </div>
                        </div>
                      )}

                      {selectedWatch.specs?.movement && (
                        <div className="flex items-center gap-2 col-span-2">
                          <Clock className="w-4 h-4 text-orange-400" />
                          <div>
                            <span className="text-gray-500">Movement:</span>
                            <span className="ml-1 font-medium">{selectedWatch.specs.movement}</span>
                          </div>
                        </div>
                      )}

                      {selectedWatch.specs?.case_material && selectedWatch.specs.case_material !== '-' && (
                        <div className="flex items-center gap-2 col-span-2">
                          <Diamond className="w-4 h-4 text-purple-400" />
                          <div>
                            <span className="text-gray-500">Material:</span>
                            <span className="ml-1 font-medium">{selectedWatch.specs.case_material}</span>
                          </div>
                        </div>
                      )}

                      {selectedWatch.specs?.dial_color && selectedWatch.specs.dial_color !== '-' && (
                        <div>
                          <span className="text-gray-500">Dial:</span>
                          <span className="ml-1 font-medium">{selectedWatch.specs.dial_color}</span>
                        </div>
                      )}

                      {selectedWatch.specs?.crystal_material && selectedWatch.specs.crystal_material !== '-' && (
                        <div>
                          <span className="text-gray-500">Crystal:</span>
                          <span className="ml-1 font-medium">{selectedWatch.specs.crystal_material}</span>
                        </div>
                      )}
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-3 pt-4 border-t border-gray-200">
                      {onLikeWatch && (
                        <button
                          onClick={() => onLikeWatch(selectedWatch)}
                          className="flex-1 bg-gradient-to-r from-pink-500 to-red-500 text-white py-3 px-4 rounded-lg font-medium hover:from-pink-600 hover:to-red-600 transition-all flex items-center justify-center gap-2"
                        >
                          <Heart className="w-4 h-4" />
                          Like This Watch
                        </button>
                      )}
                      {onSelectWatch && selectedWatch.index !== currentWatchIndex && (
                        <button
                          onClick={() => onSelectWatch(selectedWatch)}
                          className="flex-1 bg-gradient-to-r from-blue-500 to-purple-500 text-white py-3 px-4 rounded-lg font-medium hover:from-blue-600 hover:to-purple-600 transition-all"
                        >
                          View This Watch
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
} 