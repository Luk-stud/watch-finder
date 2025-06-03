import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import { Watch, getWatchImageUrl, formatPrice, getWatchPrice } from '../data/watchData';
import { ArrowLeft, Heart, ExternalLink, Info } from 'lucide-react';
import { useViewportHeight } from '../hooks/useViewportHeight';

const LikedWatches = () => {
  const location = useLocation();
  const likedWatches = (location.state?.likedWatches as Watch[]) || [];

  // Use the viewport height hook for mobile browser compatibility
  useViewportHeight();

  return (
    <div className="flex-viewport bg-gradient-to-br from-slate-900 via-gray-900 to-black">
      {/* Header */}
      <div className="p-6 border-b border-gray-800 flex-shrink-0">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
        <Link
          to="/swipe"
              className="w-10 h-10 bg-gray-700 hover:bg-gray-600 rounded-full flex items-center justify-center text-white transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
        </Link>
        <div>
              <h1 className="text-3xl font-bold text-white">Your Liked Watches</h1>
              <p className="text-gray-400">
                {likedWatches.length} watch{likedWatches.length !== 1 ? 'es' : ''} selected
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-red-400">
            <Heart className="w-6 h-6" />
            <span className="text-2xl font-bold">{likedWatches.length}</span>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 p-6 overflow-auto min-h-0">
        <div className="max-w-6xl mx-auto">
        {likedWatches.length === 0 ? (
            <div className="text-center py-20">
              <Heart className="w-24 h-24 mx-auto mb-6 text-gray-600" />
              <h2 className="text-2xl font-bold text-white mb-4">No Liked Watches Yet</h2>
              <p className="text-gray-400 mb-8">
                Start swiping to discover watches you love!
              </p>
            <Link
              to="/swipe"
                className="bg-gradient-to-r from-yellow-400 to-yellow-600 text-black px-8 py-3 rounded-full font-semibold hover:scale-105 transition-transform"
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
                    </div>
                  </div>
  );
};

interface LikedWatchCardProps {
  watch: Watch;
}

const LikedWatchCard: React.FC<LikedWatchCardProps> = ({ watch }) => {
  const price = getWatchPrice(watch);

  return (
    <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl overflow-hidden shadow-xl border border-gray-700 hover:border-yellow-500/50 transition-all hover:scale-105">
      {/* Watch Image */}
      <div className="aspect-square bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center p-6 relative overflow-hidden">
        <img
          src={getWatchImageUrl(watch)}
          alt={`${watch.brand} ${watch.model}`}
          className="w-full h-full object-contain"
          style={{
            filter: 'drop-shadow(0 10px 20px rgba(0,0,0,0.3))',
            maxWidth: '90%',
            maxHeight: '90%'
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
            <div className="text-black font-bold text-lg">{watch.brand}</div>
            <div className="text-black text-sm">{watch.model}</div>
                  </div>
                </div>

        {/* AI Badge */}
        {watch.algorithm && (
          <div className="absolute top-3 right-3 bg-black/70 text-white px-2 py-1 rounded-md text-xs font-mono">
            {watch.algorithm}
            {watch.confidence && (
              <span className="ml-1 text-green-300">
                {watch.confidence.toFixed(1)}
              </span>
            )}
                </div>
        )}

        {/* Seed Badge */}
        {watch.is_seed && (
          <div className="absolute top-3 left-3 bg-blue-500/80 text-white px-2 py-1 rounded-md text-xs font-bold">
            SEED
          </div>
        )}

        {/* Quick specs overlay */}
        <div className="absolute bottom-3 left-3 right-3">
          <div className="flex flex-wrap gap-1">
            {watch.specs?.diameter_mm && watch.specs.diameter_mm !== '-' && (
              <span className="bg-black/70 text-white text-xs px-2 py-1 rounded-full">
                {watch.specs.diameter_mm}mm
              </span>
            )}
            {watch.specs?.case_material && watch.specs.case_material !== '-' && (
              <span className="bg-black/70 text-white text-xs px-2 py-1 rounded-full">
                {watch.specs.case_material}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Watch Info */}
      <div className="p-6">
        <h3 className="text-xl font-bold text-white mb-1 truncate">{watch.model}</h3>
        <p className="text-yellow-400 font-semibold mb-2">{watch.brand}</p>
        
        {watch.specs?.serie && watch.specs.serie !== '-' && (
          <p className="text-gray-400 text-sm mb-3">{watch.specs.serie} Series</p>
        )}

        {/* Key Specifications */}
        <div className="space-y-2 mb-4">
          {watch.specs?.movement && watch.specs.movement !== '-' && (
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Movement</span>
              <span className="text-white truncate ml-2">{watch.specs.movement}</span>
            </div>
          )}
          
          {watch.specs?.waterproofing_meters && watch.specs.waterproofing_meters !== '-' && (
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Water Resistance</span>
              <span className="text-white">{watch.specs.waterproofing_meters}m</span>
            </div>
          )}

          {watch.specs?.power_reserve_hour && watch.specs.power_reserve_hour !== '-' && (
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Power Reserve</span>
              <span className="text-white">{watch.specs.power_reserve_hour}h</span>
            </div>
          )}
        </div>

        {/* Complications */}
        <div className="flex flex-wrap gap-1 mb-4">
          {watch.specs?.complication_date && watch.specs.complication_date !== '-' && (
            <span className="bg-blue-500/20 text-blue-300 px-2 py-1 rounded text-xs">Date</span>
          )}
          {watch.specs?.complication_chronograph && watch.specs.complication_chronograph !== '-' && (
            <span className="bg-purple-500/20 text-purple-300 px-2 py-1 rounded text-xs">Chronograph</span>
          )}
          {watch.specs?.complication_gmt && watch.specs.complication_gmt !== '-' && (
            <span className="bg-green-500/20 text-green-300 px-2 py-1 rounded text-xs">GMT</span>
          )}
        </div>

        {/* Price */}
        <div className="flex justify-between items-end">
          <div>
            <p className="text-2xl font-bold text-white">{formatPrice(price)}</p>
            {watch.specs?.availability && watch.specs.availability !== '-' && (
              <p className="text-gray-400 text-xs">{watch.specs.availability}</p>
            )}
          </div>
          {watch.specs?.launch_date && (
            <p className="text-gray-400 text-sm">{watch.specs.launch_date}</p>
          )}
        </div>

        {/* Action Links */}
        <div className="flex gap-2 mt-4">
          {watch.specs?.brand_website && watch.specs.brand_website !== '-' && (
            <a
              href={watch.specs.brand_website}
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 bg-gray-700 hover:bg-gray-600 text-white py-2 px-3 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
            >
              <ExternalLink className="w-4 h-4" />
              Brand Site
            </a>
          )}
          
          {watch.product_url && (
            <a
              href={watch.product_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 bg-yellow-600 hover:bg-yellow-500 text-black py-2 px-3 rounded-lg text-sm font-medium transition-colors flex items-center justify-center gap-2"
            >
              <Info className="w-4 h-4" />
              Details
            </a>
          )}
        </div>
      </div>
    </div>
  );
};

export default LikedWatches;
