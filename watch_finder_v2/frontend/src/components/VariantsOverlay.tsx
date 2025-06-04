import React from 'react';
import { X } from 'lucide-react';
import { ModernWatch } from '../lib/api';
import { formatPrice, getWatchImageUrl } from '../data/watchData';

interface VariantsOverlayProps {
  watch: ModernWatch;
  variants: ModernWatch[];
  onClose: () => void;
}

const VariantsOverlay: React.FC<VariantsOverlayProps> = ({ watch, variants, onClose }) => {
  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden shadow-2xl border border-gray-700">
        {/* Header */}
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white">
              {watch.brand} {watch.model} Variants
            </h2>
            <p className="text-gray-400 text-sm">
              {variants.length} variants available
            </p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-full bg-gray-700/50 hover:bg-gray-700 flex items-center justify-center text-gray-300"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Variants Grid */}
        <div className="p-4 overflow-y-auto max-h-[calc(90vh-4rem)]">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {variants.map((variant) => (
              <div
                key={variant.index}
                className={`bg-gray-800/50 rounded-xl p-4 border ${
                  variant.index === watch.index
                    ? 'border-yellow-500'
                    : 'border-gray-700'
                }`}
              >
                {/* Watch Image */}
                <div className="aspect-square bg-gray-900 rounded-lg mb-4 overflow-hidden">
                  <img
                    src={getWatchImageUrl(variant)}
                    alt={`${variant.brand} ${variant.model}`}
                    className="w-full h-full object-contain p-2"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.src = '/placeholder.png';
                    }}
                  />
                </div>

                {/* Watch Info */}
                <div>
                  <h3 className="text-white font-semibold mb-1 truncate">
                    {variant.model}
                  </h3>
                  <p className="text-yellow-400 text-sm mb-2">{variant.brand}</p>

                  {/* Key Differences */}
                  <div className="space-y-1">
                    {variant.specs?.case_material && (
                      <p className="text-gray-400 text-sm truncate">
                        Case: {variant.specs.case_material}
                      </p>
                    )}
                    {variant.specs?.dial_color && (
                      <p className="text-gray-400 text-sm truncate">
                        Dial: {variant.specs.dial_color}
                      </p>
                    )}
                    {variant.specs?.diameter_mm && (
                      <p className="text-gray-400 text-sm truncate">
                        Size: {variant.specs.diameter_mm}mm
                      </p>
                    )}
                    {variant.price && (
                      <p className="text-white font-semibold mt-2">
                        {formatPrice(variant.price)}
                      </p>
                    )}
                  </div>

                  {/* Current Watch Indicator */}
                  {variant.index === watch.index && (
                    <div className="mt-2 text-yellow-500 text-sm font-medium">
                      Current Watch
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VariantsOverlay; 