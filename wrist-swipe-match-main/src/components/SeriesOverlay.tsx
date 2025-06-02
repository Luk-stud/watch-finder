import React from 'react';
import { Watch, getWatchImageUrl, formatPrice, getWatchPrice } from '../data/watchData';
import { X, Heart, ArrowRight } from 'lucide-react';

interface SeriesOverlayProps {
  series: string;
  watches?: Watch[];
  onClose: () => void;
}

const SeriesOverlay: React.FC<SeriesOverlayProps> = ({ series, watches = [], onClose }) => {
  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl max-w-4xl w-full max-h-[80vh] overflow-hidden border border-gray-700">
        {/* Header */}
        <div className="p-6 border-b border-gray-700">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-bold text-white mb-1">{series} Series</h2>
              <p className="text-gray-400">{watches.length} watches in this series</p>
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
        <div className="p-6 overflow-y-auto max-h-[calc(80vh-120px)]">
          {watches.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-gray-400 text-lg">No watches found in this series</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {watches.map((watch, index) => (
                <WatchSeriesCard key={watch.index} watch={watch} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

interface WatchSeriesCardProps {
  watch: Watch;
}

const WatchSeriesCard: React.FC<WatchSeriesCardProps> = ({ watch }) => {
  const price = getWatchPrice(watch);
  
  return (
    <div className="bg-gray-700/50 rounded-xl p-4 hover:bg-gray-700/70 transition-colors">
      {/* Watch Image */}
      <div className="aspect-square bg-gray-600 rounded-lg mb-4 overflow-hidden flex items-center justify-center">
        <img
          src={getWatchImageUrl(watch)}
          alt={`${watch.brand} ${watch.model}`}
          className="w-full h-full object-contain p-2"
          style={{
            filter: 'drop-shadow(0 5px 10px rgba(0,0,0,0.3))',
          }}
          onError={(e) => {
            // Fallback to placeholder on error
            const target = e.target as HTMLImageElement;
            target.style.display = 'none';
            const fallback = target.nextElementSibling as HTMLElement;
            if (fallback) fallback.style.display = 'flex';
          }}
        />
        <div
          className="w-full h-full bg-gradient-to-br from-gray-300 to-gray-500 rounded-lg items-center justify-center"
          style={{ display: 'none' }}
        >
          <div className="text-center">
            <div className="text-black font-bold text-sm">{watch.brand}</div>
            <div className="text-black text-xs">{watch.model}</div>
          </div>
        </div>
      </div>

      {/* Watch Info */}
      <div className="space-y-2">
        <h3 className="text-white font-semibold text-lg truncate">{watch.model}</h3>
        <p className="text-yellow-400 font-medium">{watch.brand}</p>
        
        {/* Key specifications */}
        <div className="space-y-1">
          {watch.specs?.diameter_mm && watch.specs.diameter_mm !== '-' && (
            <p className="text-gray-300 text-sm">
              <span className="text-gray-400">Size:</span> {watch.specs.diameter_mm}mm
            </p>
          )}
          
          {watch.specs?.case_material && watch.specs.case_material !== '-' && (
            <p className="text-gray-300 text-sm">
              <span className="text-gray-400">Material:</span> {watch.specs.case_material}
            </p>
          )}
          
          {watch.specs?.movement && watch.specs.movement !== '-' && (
            <p className="text-gray-300 text-sm truncate">
              <span className="text-gray-400">Movement:</span> {watch.specs.movement}
            </p>
          )}
        </div>

        {/* Price */}
        <div className="pt-2">
          <p className="text-white font-bold text-xl">{formatPrice(price)}</p>
          {watch.specs?.availability && watch.specs.availability !== '-' && (
            <p className="text-gray-400 text-xs">{watch.specs.availability}</p>
          )}
        </div>

        {/* Additional info badges */}
        <div className="flex flex-wrap gap-1 pt-2">
          {watch.specs?.waterproofing_meters && watch.specs.waterproofing_meters !== '-' && (
            <span className="bg-blue-500/20 text-blue-300 px-2 py-1 rounded text-xs">
              {watch.specs.waterproofing_meters}m WR
            </span>
          )}
          
          {watch.specs?.complication_chronograph && watch.specs.complication_chronograph !== '-' && (
            <span className="bg-purple-500/20 text-purple-300 px-2 py-1 rounded text-xs">
              Chronograph
            </span>
          )}
          
          {watch.specs?.complication_gmt && watch.specs.complication_gmt !== '-' && (
            <span className="bg-green-500/20 text-green-300 px-2 py-1 rounded text-xs">
              GMT
            </span>
          )}

          {watch.is_seed && (
            <span className="bg-blue-500/30 text-blue-200 px-2 py-1 rounded text-xs font-bold">
              SEED
            </span>
          )}
        </div>

        {/* Brand website link */}
        {watch.specs?.brand_website && watch.specs.brand_website !== '-' && (
          <div className="pt-3">
            <a
              href={watch.specs.brand_website}
              target="_blank"
              rel="noopener noreferrer"
              className="w-full bg-gray-600 hover:bg-gray-500 text-white py-2 px-3 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
            >
              View Details <ArrowRight className="w-4 h-4" />
            </a>
          </div>
        )}
      </div>
    </div>
  );
};

export default SeriesOverlay;
