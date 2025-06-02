import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import WatchCard from '../components/WatchCard';
import SpecsOverlay from '../components/SpecsOverlay';
import SeriesOverlay from '../components/SeriesOverlay';
import { Watch, formatPrice } from '../data/watchData';
import { apiService } from '../lib/api';
import { useViewportHeight } from '../hooks/useViewportHeight';
import { Heart, X, Loader2, AlertTriangle } from 'lucide-react';

const Index = () => {
  const [currentWatches, setCurrentWatches] = useState<Watch[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [showSpecs, setShowSpecs] = useState(false);
  const [showSeries, setShowSeries] = useState(false);
  const [selectedWatch, setSelectedWatch] = useState<Watch | null>(null);
  const [seriesWatches, setSeriesWatches] = useState<Watch[]>([]);
  const [likedWatches, setLikedWatches] = useState<Watch[]>([]);
  const [dislikedWatches, setDislikedWatches] = useState<Watch[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState(0);
  const [isTransitioning, setIsTransitioning] = useState(false);

  // Use the viewport height hook for mobile browser compatibility
  useViewportHeight();

  const currentWatch = currentWatches[currentIndex];

  // Debug logging for card state
  useEffect(() => {
    console.log(`📱 Card State: index=${currentIndex}, total=${currentWatches.length}, hasCurrentWatch=${!!currentWatch}, isTransitioning=${isTransitioning}, isLoading=${isLoading}`);
  }, [currentIndex, currentWatches.length, currentWatch, isTransitioning, isLoading]);

  // Initialize session and get first watches
  useEffect(() => {
    initializeSession();
  }, []);

  const initializeSession = async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Check backend health first
      await apiService.checkHealth();
      
      // Start a new session with 7 seeds
      const response = await apiService.startSession(7);
      
      if (response.status === 'success') {
        setCurrentWatches(response.seeds);
        setCurrentIndex(0);
        setStep(0);
        
        // Log some debug info about the algorithm
        console.log(`🚀 Session started with ${response.seeds.length} seeds`);
        if (response.algorithm_used) {
          console.log(`🤖 Algorithm: ${response.algorithm_used}`);
        }
      } else {
        throw new Error('Failed to start session');
      }
    } catch (err) {
      console.error('Failed to initialize session:', err);
      setError(err instanceof Error ? err.message : 'Failed to connect to backend');
    } finally {
      setIsLoading(false);
    }
  };

  const getMoreRecommendations = async () => {
    try {
      setIsLoading(true);
      
      const likedIndices = likedWatches.map(w => w.index);
      const dislikedIndices = dislikedWatches.map(w => w.index);
      const currentCandidates = currentWatches.map(w => w.index);

      const response = await apiService.getRecommendations(
        likedIndices,
        dislikedIndices,
        currentCandidates,
        step + 1,
        7 // Get 7 new recommendations
      );
      
      if (response.status === 'success' && response.recommendations.length > 0) {
        // Log modern backend insights
        if (response.algorithm_used) {
          console.log(`🤖 Algorithm used: ${response.algorithm_used}`);
        }
        if (response.diversity_score !== undefined) {
          console.log(`🎯 Diversity score: ${response.diversity_score.toFixed(2)}`);
        }
        if (response.user_profile_summary) {
          console.log(`👤 User engagement: ${response.user_profile_summary.engagement_level}`);
        }
        if (response.next_exploration_suggestions) {
          console.log(`💡 Suggestions:`, response.next_exploration_suggestions);
        }
        
        // Update state atomically to prevent gaps
        setCurrentWatches(response.recommendations);
        setCurrentIndex(0);
        setStep(response.step);
      } else {
        throw new Error('No more recommendations available');
      }
    } catch (err) {
      console.error('Error getting recommendations:', err);
      setError(err instanceof Error ? err.message : 'Failed to get recommendations');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSwipe = async (direction: 'left' | 'right') => {
    if (!currentWatch || isTransitioning) return;

    try {
      setIsTransitioning(true);
      
      // Add feedback to backend
      await apiService.addFeedback(
        currentWatch.index,
        direction === 'right' ? 'like' : 'dislike',
        1.0
      );

      // Update local state
      if (direction === 'right') {
        setLikedWatches(prev => [...prev, currentWatch]);
      } else {
        setDislikedWatches(prev => [...prev, currentWatch]);
      }

      // Move to next watch or get more recommendations
      if (currentIndex < currentWatches.length - 1) {
        // Still have cards in current batch - move to next immediately
        setTimeout(() => {
          setCurrentIndex(prev => prev + 1);
          setIsTransitioning(false);
        }, 100);
      } else {
        // Last card in batch - get more recommendations
        setTimeout(async () => {
          try {
            await getMoreRecommendations();
          } catch (err) {
            console.error('Failed to get more recommendations:', err);
          } finally {
            setIsTransitioning(false);
          }
        }, 100);
      }

    } catch (err) {
      console.error('Error processing swipe:', err);
      // Still move to next watch even if feedback fails
      setTimeout(() => {
        if (currentIndex < currentWatches.length - 1) {
          setCurrentIndex(prev => prev + 1);
        }
        setIsTransitioning(false);
      }, 100);
    }
  };

  const handleSpecsClick = (watch: Watch) => {
    setSelectedWatch(watch);
    setShowSpecs(true);
  };

  const handleSeriesClick = async (watch: Watch) => {
    try {
      setIsLoading(true);
      const response = await apiService.getSeries(watch.index);
      
      if (response.status === 'success' && response.series_watches.length > 1) {
        setSelectedWatch(watch);
        setSeriesWatches(response.series_watches);
        setShowSeries(true);
      } else {
        console.log('No series found or only one watch in series');
      }
    } catch (err) {
      console.error('Error getting series:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const resetCards = () => {
    setCurrentIndex(0);
    setLikedWatches([]);
    setDislikedWatches([]);
    setStep(0);
    initializeSession();
  };

  // Error state
  if (error) {
    return (
      <div className="flex-viewport bg-gradient-to-br from-slate-900 via-gray-900 to-black flex items-center justify-center p-4">
        <div className="text-center text-white max-w-md">
          <AlertTriangle className="w-16 h-16 mx-auto mb-4 text-red-400" />
          <h2 className="text-2xl font-bold mb-4">Connection Error</h2>
          <p className="text-gray-300 mb-6">{error}</p>
          <button
            onClick={initializeSession}
            className="bg-gradient-to-r from-blue-500 to-blue-700 text-white px-6 py-3 rounded-full font-semibold hover:scale-105 transition-transform"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  // Loading state
  if (isLoading && currentWatches.length === 0) {
    return (
      <div className="flex-viewport bg-gradient-to-br from-slate-900 via-gray-900 to-black flex items-center justify-center p-4">
        <div className="text-center text-white">
          <Loader2 className="w-16 h-16 mx-auto mb-4 animate-spin text-blue-400" />
          <h2 className="text-2xl font-bold mb-4">Starting AI Engine...</h2>
          <p className="text-gray-300">Connecting to recommendation backend</p>
        </div>
      </div>
    );
  }

  // No more watches state
  if (currentIndex >= currentWatches.length && currentWatches.length > 0) {
    return (
      <div className="flex-viewport bg-gradient-to-br from-slate-900 via-gray-900 to-black flex items-center justify-center p-4">
        <div className="text-center text-white">
          <h2 className="text-3xl font-bold mb-4">You've seen all recommendations!</h2>
          <p className="text-gray-300 mb-2">Liked {likedWatches.length} watches</p>
          <p className="text-gray-400 text-sm mb-6">
            AI analyzed your preferences across {step} recommendation rounds
          </p>
          <div className="flex flex-col gap-4">
            <Link
              to="/liked"
              state={{ likedWatches }}
              className="bg-gradient-to-r from-green-400 to-green-600 text-black px-6 py-3 rounded-full font-semibold hover:scale-105 transition-transform"
            >
              View Liked Watches
            </Link>
            <button
              onClick={resetCards}
              className="bg-gradient-to-r from-yellow-400 to-yellow-600 text-black px-6 py-3 rounded-full font-semibold hover:scale-105 transition-transform"
            >
              Start New Session
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-viewport bg-gradient-to-br from-slate-900 via-gray-900 to-black overflow-hidden">
      {/* Header */}
      <div className="flex justify-between items-center p-4 text-white flex-shrink-0">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-yellow-400 to-yellow-600 bg-clip-text text-transparent">
          WatchSwipe AI
        </h1>
        <div className="flex items-center gap-4">
          <Link
            to="/liked"
            state={{ likedWatches }}
            className="bg-gradient-to-r from-green-400/20 to-green-600/20 border border-green-500 text-green-400 px-4 py-2 rounded-full text-sm font-medium hover:bg-green-500/30 transition-colors flex items-center gap-2"
          >
            <Heart className="w-4 h-4" />
            {likedWatches.length}
          </Link>
          <div className="text-sm text-gray-400">
            Round {step + 1} • {currentIndex + 1} / {currentWatches.length}
          </div>
        </div>
      </div>

      {/* Card Stack - Uses flexible container that scales with available space */}
      <div className="card-container p-4">
        <div className="watch-card-wrapper">
          {/* Only render cards if we have data and valid index */}
          {currentWatches.length > 0 && currentIndex < currentWatches.length && (
            <>
              {/* Preview cards (behind current card) with real watch data */}
              {[2, 1].map((offset) => {
                const previewIndex = currentIndex + offset;
                const previewWatch = currentWatches[previewIndex];
                if (!previewWatch) return null;
                
                return (
                  <div
                    key={`preview-${previewIndex}-${previewWatch.index}`}
                    className="absolute inset-0 w-full pointer-events-none"
                    style={{
                      transform: `scale(${1 - offset * 0.05}) translateY(${offset * 8}px)`,
                      zIndex: -offset,
                      opacity: 1 - offset * 0.3
                    }}
                  >
                    <WatchCard
                      watch={previewWatch}
                      onSwipe={() => {}} // Disabled for preview cards
                      onSpecsClick={() => {}} // Disabled for preview cards
                      onSeriesClick={() => {}} // Disabled for preview cards
                    />
                  </div>
                );
              })}

              {/* Current card */}
              {currentWatch && (
                <div key={`current-${currentIndex}-${currentWatch.index}`} className="relative z-10 h-full">
                  <WatchCard
                    watch={currentWatch}
                    onSwipe={handleSwipe}
                    onSpecsClick={handleSpecsClick}
                    onSeriesClick={handleSeriesClick}
                  />
                </div>
              )}
            </>
          )}
          
          {/* Show loading state in the card area when getting new recommendations */}
          {isLoading && currentWatches.length === 0 && (
            <div className="flex items-center justify-center h-full text-white">
              <div className="text-center">
                <Loader2 className="w-12 h-12 mx-auto mb-4 animate-spin text-blue-400" />
                <p className="text-lg">Getting recommendations...</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Action Buttons - Stay at bottom with safe area support */}
      <div className="action-buttons flex justify-center items-center gap-8 p-4">
        <button
          onClick={() => handleSwipe('left')}
          className="w-16 h-16 bg-red-500/20 hover:bg-red-500/30 border-2 border-red-500 rounded-full flex items-center justify-center transition-all hover:scale-110"
          disabled={isLoading || isTransitioning}
        >
          <X className="w-8 h-8 text-red-500" />
        </button>
        
        <button
          onClick={() => handleSwipe('right')}
          className="w-16 h-16 bg-green-500/20 hover:bg-green-500/30 border-2 border-green-500 rounded-full flex items-center justify-center transition-all hover:scale-110"
          disabled={isLoading || isTransitioning}
        >
          <Heart className="w-8 h-8 text-green-500" />
        </button>
      </div>

      {/* Loading overlay */}
      {isLoading && currentWatches.length > 0 && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-6 rounded-xl text-white text-center">
            <Loader2 className="w-8 h-8 mx-auto mb-2 animate-spin" />
            <p>Getting recommendations...</p>
          </div>
        </div>
      )}

      {/* Overlays */}
      {showSpecs && selectedWatch && (
        <SpecsOverlay
          watch={selectedWatch}
          onClose={() => setShowSpecs(false)}
        />
      )}

      {showSeries && selectedWatch && (
        <SeriesOverlay
          series={selectedWatch.specs?.serie || selectedWatch.brand}
          watches={seriesWatches}
          onClose={() => setShowSeries(false)}
        />
      )}
    </div>
  );
};

export default Index;
