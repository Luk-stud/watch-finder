import React from 'react';
import { Watch, getWatchComplications, getWatchMovement, getWatchCaseMaterial, getWatchDiameter, getWatchWaterResistance, formatPrice, getWatchPrice } from '../data/watchData';
import { X, Info, Calendar, Cog, Droplets, Ruler, Settings } from 'lucide-react';

interface SpecsOverlayProps {
  watch: Watch;
  onClose: () => void;
}

const SpecsOverlay: React.FC<SpecsOverlayProps> = ({ watch, onClose }) => {
  const price = getWatchPrice(watch);
  const complications = getWatchComplications(watch);

  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl max-w-md w-full max-h-[80vh] overflow-y-auto border border-gray-700">
        {/* Header */}
        <div className="p-6 border-b border-gray-700">
          <div className="flex justify-between items-start">
            <div>
              <h2 className="text-2xl font-bold text-white mb-1">{watch.model}</h2>
              <p className="text-yellow-400 font-semibold">{watch.brand}</p>
              {watch.specs?.serie && watch.specs.serie !== '-' && (
                <p className="text-gray-400 text-sm">{watch.specs.serie} Series</p>
              )}
            </div>
          <button
            onClick={onClose}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-lg transition-colors"
          >
              <X className="w-6 h-6" />
          </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Price and Basic Info */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-700/50 p-4 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Settings className="w-4 h-4 text-yellow-400" />
                <span className="text-gray-300 text-sm">Price</span>
              </div>
              <p className="text-white font-semibold">{formatPrice(price)}</p>
            </div>
            
            {watch.specs?.reference && watch.specs.reference !== '-' && (
              <div className="bg-gray-700/50 p-4 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Info className="w-4 h-4 text-blue-400" />
                  <span className="text-gray-300 text-sm">Reference</span>
                </div>
                <p className="text-white font-semibold text-sm">{watch.specs.reference}</p>
              </div>
            )}
          </div>

          {/* Technical Specifications */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Technical Specifications</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center py-2 border-b border-gray-700">
                <span className="text-gray-300">Movement</span>
                <span className="text-white font-medium">{getWatchMovement(watch)}</span>
              </div>
              
              <div className="flex justify-between items-center py-2 border-b border-gray-700">
                <span className="text-gray-300">Case Material</span>
                <span className="text-white font-medium">{getWatchCaseMaterial(watch)}</span>
              </div>
              
              <div className="flex justify-between items-center py-2 border-b border-gray-700">
                <span className="text-gray-300">Case Diameter</span>
                <span className="text-white font-medium">{getWatchDiameter(watch)}</span>
              </div>
              
              <div className="flex justify-between items-center py-2 border-b border-gray-700">
                <span className="text-gray-300">Water Resistance</span>
                <span className="text-white font-medium">{getWatchWaterResistance(watch)}</span>
              </div>

              {watch.specs?.dial_color && watch.specs.dial_color !== '-' && (
                <div className="flex justify-between items-center py-2 border-b border-gray-700">
                  <span className="text-gray-300">Dial Color</span>
                  <span className="text-white font-medium">{watch.specs.dial_color}</span>
                </div>
              )}

              {watch.specs?.strap_1_material && watch.specs.strap_1_material !== '-' && (
                <div className="flex justify-between items-center py-2 border-b border-gray-700">
                  <span className="text-gray-300">Strap/Bracelet</span>
                  <span className="text-white font-medium">{watch.specs.strap_1_material}</span>
                </div>
              )}

              {watch.specs?.power_reserve_hour && watch.specs.power_reserve_hour !== '-' && (
                <div className="flex justify-between items-center py-2 border-b border-gray-700">
                  <span className="text-gray-300">Power Reserve</span>
                  <span className="text-white font-medium">{watch.specs.power_reserve_hour}h</span>
                </div>
              )}
            </div>
          </div>

          {/* Complications */}
          {complications.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Complications</h3>
              <div className="flex flex-wrap gap-2">
                {complications.map((complication) => (
                  <span
                    key={complication}
                    className="bg-blue-500/20 text-blue-300 px-3 py-1 rounded-full text-sm border border-blue-500/30"
                  >
                    {complication}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Additional Information */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Additional Information</h3>
            <div className="space-y-3">
              {watch.specs?.availability && watch.specs.availability !== '-' && (
                <div className="flex justify-between items-center py-2 border-b border-gray-700">
                  <span className="text-gray-300">Availability</span>
                  <span className="text-white font-medium">{watch.specs.availability}</span>
                </div>
              )}

              {watch.specs?.launch_date && watch.specs.launch_date !== '-' && (
                <div className="flex justify-between items-center py-2 border-b border-gray-700">
                  <span className="text-gray-300">Launch Date</span>
                  <span className="text-white font-medium">{watch.specs.launch_date}</span>
                </div>
              )}

              {watch.specs?.made_in && watch.specs.made_in !== '-' && (
                <div className="flex justify-between items-center py-2 border-b border-gray-700">
                  <span className="text-gray-300">Made In</span>
                  <span className="text-white font-medium">{watch.specs.made_in}</span>
                </div>
              )}

              {watch.specs?.warranty_year && watch.specs.warranty_year !== '-' && (
                <div className="flex justify-between items-center py-2 border-b border-gray-700">
                  <span className="text-gray-300">Warranty</span>
                  <span className="text-white font-medium">{watch.specs.warranty_year} years</span>
                </div>
              )}
            </div>
          </div>

          {/* Modern Backend Debug Info */}
          {(watch.algorithm || watch.confidence || watch.is_seed) && (
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">AI Recommendation Info</h3>
              <div className="space-y-2">
                {watch.algorithm && (
                  <div className="flex justify-between items-center py-2 border-b border-gray-700">
                    <span className="text-gray-300">Algorithm</span>
                    <span className="text-green-400 font-mono text-sm">{watch.algorithm}</span>
                  </div>
                )}
                
                {watch.confidence && (
                  <div className="flex justify-between items-center py-2 border-b border-gray-700">
                    <span className="text-gray-300">Confidence</span>
                    <span className="text-green-400 font-mono text-sm">{watch.confidence.toFixed(3)}</span>
                  </div>
                )}
                
                {watch.is_seed && (
                  <div className="flex justify-between items-center py-2 border-b border-gray-700">
                    <span className="text-gray-300">Type</span>
                    <span className="text-blue-400 font-mono text-sm">SEED</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Brand Website Link */}
          {watch.specs?.brand_website && watch.specs.brand_website !== '-' && (
            <div className="pt-4">
              <a
                href={watch.specs.brand_website}
                target="_blank"
                rel="noopener noreferrer"
                className="w-full bg-gradient-to-r from-blue-500 to-blue-700 text-white py-3 px-4 rounded-lg font-semibold hover:from-blue-600 hover:to-blue-800 transition-colors flex items-center justify-center gap-2"
              >
                <Info className="w-5 h-5" />
                Visit Brand Website
              </a>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SpecsOverlay;
