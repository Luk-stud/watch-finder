import React from 'react';
import { Link } from 'react-router-dom';
import { Heart, X, Zap } from 'lucide-react';
import { useViewportHeight } from '../hooks/useViewportHeight';

const Landing = () => {
  // Use the viewport height hook for mobile browser compatibility
  useViewportHeight();

  return (
    <div className="flex-viewport bg-gradient-to-br from-slate-900 via-gray-900 to-black">
      {/* Header */}
      <div className="p-4 flex-shrink-0">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-yellow-400 to-yellow-600 bg-clip-text text-transparent text-center">
          WatchSwipe
        </h1>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col items-center justify-center p-6 text-center min-h-0">
        {/* Hero Section */}
        <div className="max-w-md mx-auto mb-8">
          <div className="w-32 h-32 bg-gradient-to-br from-gray-300 to-gray-500 rounded-full shadow-xl mx-auto mb-6 flex items-center justify-center">
            <div className="w-24 h-24 bg-gradient-to-br from-yellow-400 to-yellow-600 rounded-full flex items-center justify-center">
              <Zap className="w-10 h-10 text-black" />
            </div>
          </div>
          
          <h2 className="text-3xl font-bold text-white mb-4">
            Find Your Perfect Watch
          </h2>
          
          <p className="text-gray-300 text-lg mb-8 leading-relaxed">
            Swipe through luxury watches like never before. Discover timepieces that match your style, explore detailed specs, and find watches from your favorite series.
          </p>
        </div>

        {/* Features */}
        <div className="grid grid-cols-1 gap-4 max-w-sm mx-auto mb-8">
          <div className="flex items-center gap-3 text-left">
            <div className="w-10 h-10 bg-green-500/20 border border-green-500 rounded-full flex items-center justify-center flex-shrink-0">
              <Heart className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <h3 className="text-white font-semibold">Like & Dislike</h3>
              <p className="text-gray-400 text-sm">Swipe right to like, left to pass</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3 text-left">
            <div className="w-10 h-10 bg-blue-500/20 border border-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
              <Zap className="w-5 h-5 text-blue-500" />
            </div>
            <div>
              <h3 className="text-white font-semibold">Detailed Specs</h3>
              <p className="text-gray-400 text-sm">Tap to see complete specifications</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3 text-left">
            <div className="w-10 h-10 bg-purple-500/20 border border-purple-500 rounded-full flex items-center justify-center flex-shrink-0">
              <X className="w-5 h-5 text-purple-500" />
            </div>
            <div>
              <h3 className="text-white font-semibold">Series Explorer</h3>
              <p className="text-gray-400 text-sm">Discover all watches in a series</p>
            </div>
          </div>
        </div>

        {/* CTA Button */}
        <Link
          to="/swipe"
          className="bg-gradient-to-r from-yellow-400 to-yellow-600 text-black px-8 py-4 rounded-full font-bold text-lg hover:scale-105 transition-transform shadow-lg"
        >
          Start Swiping
        </Link>
        
        <p className="text-gray-500 text-sm mt-4">
          Discover your next timepiece
        </p>
      </div>
    </div>
  );
};

export default Landing;
