'use client';

import { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Compass, AlertTriangle, Heart, X, History } from 'lucide-react';
import Image from 'next/image';
import { useApp } from '@/contexts/AppContext';
import { apiService } from '@/lib/api';
import { CONFIG, getPlaceholderImage, formatPrice } from '@/lib/utils';
import LoadingScreen from './LoadingScreen';
import WelcomeScreen from './WelcomeScreen';
import WatchCard from './WatchCard';
import Navigation from './Navigation';

type AppScreen = 'loading' | 'welcome' | 'main' | 'error';

export default function WatchFinderApp() {
  const {
    currentWatches,
    currentIndex,
    likedWatches,
    dislikedWatches,
    step,
    isLoading,
    currentView,
    setCurrentWatches,
    setCurrentIndex,
    addLikedWatch,
    addDislikedWatch,
    setStep,
    setSessionId,
    setIsLoading,
    setCurrentView,
    resetState,
    getCurrentWatch,
    getTotalLiked,
  } = useApp();

  const [currentScreen, setCurrentScreen] = useState<AppScreen>('loading');
  const [error, setError] = useState<string | null>(null);

  const [seriesView, setSeriesView] = useState<{
    isVisible: boolean;
    watches: typeof likedWatches;
    currentWatchIndex: number;
    seriesName: string;
  }>({
    isVisible: false,
    watches: [],
    currentWatchIndex: -1,
    seriesName: '',
  });

  const checkHealthWithRetry = useCallback(async (retries = CONFIG.MAX_RETRIES): Promise<void> => {
    for (let i = 0; i < retries; i++) {
      try {
        await apiService.checkHealth();
        return;
      } catch (error) {
        console.warn(`Health check attempt ${i + 1} failed:`, error);
        if (i === retries - 1) throw error;
        await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY));
      }
    }
  }, []);

  const initializeApp = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Check API health with retries
      await checkHealthWithRetry();
      
      // Show welcome screen
      setCurrentScreen('welcome');
    } catch (error) {
      console.error('Failed to initialize app:', error);
      setError(error instanceof Error ? error.message : 'Failed to initialize application');
      setCurrentScreen('error');
    } finally {
      setIsLoading(false);
    }
  }, [checkHealthWithRetry, setIsLoading]);

  // Initialize the app
  useEffect(() => {
    initializeApp();
  }, [initializeApp]);

  const getMoreRecommendations = useCallback(async () => {
    try {
      setIsLoading(true);
      
      const likedIndices = likedWatches.map(w => w.index);
      const dislikedIndices = dislikedWatches.map(w => w.index);
      const currentCandidates = currentWatches.map(w => w.index);
      
      const response = await apiService.getRecommendations(
        likedIndices,
        dislikedIndices,
        currentCandidates,
        step + 1
      );
      
      if (response.status === 'success' && response.recommendations.length > 0) {
        setCurrentWatches(response.recommendations);
        setCurrentIndex(0);
        setStep(response.step);
      } else {
        throw new Error('No more recommendations available');
      }
    } catch (error) {
      console.error('Error getting recommendations:', error);
      setError(error instanceof Error ? error.message : 'Failed to get recommendations');
    } finally {
      setIsLoading(false);
    }
  }, [likedWatches, dislikedWatches, currentWatches, step, setIsLoading, setCurrentWatches, setCurrentIndex, setStep]);

  const moveToNext = useCallback(async () => {
    if (currentIndex < currentWatches.length - 1) {
      setCurrentIndex(currentIndex + 1);
    } else {
      await getMoreRecommendations();
    }
  }, [currentIndex, currentWatches.length, setCurrentIndex, getMoreRecommendations]);

  const startSession = async () => {
    try {
      setIsLoading(true);
      setError(null);
      resetState();

      const response = await apiService.startSession(3);
      
      if (response.status === 'success') {
        setCurrentWatches(response.seeds);
        setSessionId(response.session_id);
        setCurrentIndex(0);
        setCurrentScreen('main');
        setCurrentView('discover');
      } else {
        throw new Error('Failed to start session');
      }
    } catch (error) {
      console.error('Failed to start session:', error);
      setError(error instanceof Error ? error.message : 'Failed to start session');
      setCurrentScreen('error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLike = useCallback(async () => {
    const watch = getCurrentWatch();
    if (!watch) return;

    try {
      addLikedWatch(watch);
      await moveToNext();
    } catch (error) {
      console.error('Error liking watch:', error);
    }
  }, [getCurrentWatch, addLikedWatch, moveToNext]);

  const handlePass = useCallback(async () => {
    const watch = getCurrentWatch();
    if (!watch) return;

    try {
      addDislikedWatch(watch);
      await moveToNext();
    } catch (error) {
      console.error('Error passing watch:', error);
    }
  }, [getCurrentWatch, addDislikedWatch, moveToNext]);

  const handleViewChange = (view: typeof currentView) => {
    setCurrentView(view);
  };

  const handleShowSeries = async (watchIndex: number) => {
    try {
      console.log('handleShowSeries called with watchIndex:', watchIndex);
      setIsLoading(true);
      const response = await apiService.getSeries(watchIndex);
      
      console.log('Series response:', response);
      
      if (response.status === 'success' && response.series_watches.length > 1) {
        const currentWatch = response.series_watches.find(w => w.index === watchIndex);
        const seriesName = currentWatch?.specs?.serie || 'Unknown Series';
        
        console.log('Setting series view state:', {
          watchCount: response.series_watches.length,
          seriesName,
          currentWatchIndex: watchIndex
        });
        
        setSeriesView({
          isVisible: true,
          watches: response.series_watches,
          currentWatchIndex: watchIndex,
          seriesName,
        });
      } else {
        console.log('Not enough watches in series:', response.series_watches?.length || 0);
      }
    } catch (error) {
      console.error('Error getting series watches:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCloseSeriesView = () => {
    setSeriesView({
      isVisible: false,
      watches: [],
      currentWatchIndex: -1,
      seriesName: '',
    });
  };

  const handleLikeFromSeries = async (watch: typeof likedWatches[0]) => {
    try {
      addLikedWatch(watch);
      // Close modal after liking
      handleCloseSeriesView();
    } catch (error) {
      console.error('Error liking watch from series:', error);
    }
  };

  const renderLikedView = () => (
    <div className="p-6">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-gray-900 mb-2">Liked Watches</h2>
        <p className="text-gray-600">Your favorite timepieces</p>
      </div>
      
      {likedWatches.length === 0 ? (
        <div className="text-center py-16">
          <Heart className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-600 mb-2">No liked watches yet</h3>
          <p className="text-gray-500">Start discovering to build your collection</p>
        </div>
      ) : (
        <div className="space-y-4">
          {likedWatches.map((watch) => (
            <motion.div
              key={watch.index}
              className="bg-white rounded-xl p-4 shadow-sm border border-gray-100"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="flex gap-4">
                <div className="w-16 h-16 bg-gray-50 rounded-lg flex items-center justify-center flex-shrink-0 overflow-hidden">
                  <Image
                    src={watch.image_url || getPlaceholderImage()}
                    alt={`${watch.brand} ${watch.model}`}
                    width={64}
                    height={64}
                    className="object-contain w-full h-full"
                    sizes="64px"
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-gray-900 truncate">{watch.brand}</h3>
                  <p className="text-gray-600 truncate">{watch.model}</p>
                  {watch.price && (
                    <p className="text-green-600 font-medium mt-1">
                      {formatPrice(watch.price)}
                    </p>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );

  const renderHistoryView = () => {
    const allWatches = [...likedWatches, ...dislikedWatches].sort((a, b) => b.index - a.index);
    
    return (
      <div className="p-6">
        <div className="mb-6">
          <h2 className="text-xl font-bold text-gray-900 mb-2">Watch History</h2>
          <p className="text-gray-600">All watches you&apos;ve reviewed</p>
        </div>
        
        {allWatches.length === 0 ? (
          <div className="text-center py-16">
            <History className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-600 mb-2">No history yet</h3>
            <p className="text-gray-500">Start discovering to see your history</p>
          </div>
        ) : (
          <div className="space-y-3">
            {allWatches.map((watch) => {
              const isLiked = likedWatches.some(w => w.index === watch.index);
              return (
                <motion.div
                  key={watch.index}
                  className="flex items-center gap-3 p-3 bg-white rounded-lg shadow-sm border border-gray-100"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                >
                  <div className="w-12 h-12 bg-gray-50 rounded-lg flex items-center justify-center flex-shrink-0 overflow-hidden">
                    <Image
                      src={watch.image_url || getPlaceholderImage()}
                      alt={`${watch.brand} ${watch.model}`}
                      width={48}
                      height={48}
                      className="object-contain w-full h-full"
                      sizes="48px"
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-gray-900 truncate">{watch.brand}</h3>
                    <p className="text-sm text-gray-600 truncate">{watch.model}</p>
                  </div>
                  <div className={`p-2 rounded-full ${isLiked ? 'bg-green-100' : 'bg-red-100'}`}>
                    {isLiked ? (
                      <Heart className="w-4 h-4 text-green-600" />
                    ) : (
                      <X className="w-4 h-4 text-red-600" />
                    )}
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  if (currentScreen === 'loading') {
    return <LoadingScreen />;
  }

  if (currentScreen === 'welcome') {
    return <WelcomeScreen onStart={startSession} />;
  }

  if (currentScreen === 'error') {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
        <div className="text-center max-w-md">
          <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Connection Error</h2>
          <p className="text-gray-600 mb-6">{error}</p>
          <button
            onClick={initializeApp}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Simple Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-sm mx-auto flex items-center justify-center">
          <div className="flex items-center gap-3">
            <Compass className="w-6 h-6 text-blue-600" />
            <h1 className="text-lg font-bold text-gray-900">Watch Finder</h1>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <div className="bg-white border-b border-gray-100 px-6 py-3">
        <div className="max-w-sm mx-auto">
          <Navigation
            currentView={currentView}
            likedCount={getTotalLiked()}
            onViewChange={handleViewChange}
          />
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-sm mx-auto h-[calc(100vh-120px)]">
        <AnimatePresence mode="wait">
          {currentView === 'discover' && (
            <div className="flex-1 flex flex-col overflow-y-auto">
              {/* Discovery Section */}
              <div className="flex-shrink-0">
                <motion.div className="p-4 h-screen relative bg-gradient-to-br from-gray-900 via-gray-800 to-black">
                  {/* Header */}
                  <div className="flex justify-between items-center mb-6">
                    <div>
                      <h1 className="text-2xl font-bold text-white">Discover Watches</h1>
                      <p className="text-gray-400">Swipe right to like, left to pass</p>
                    </div>
                    <button
                      onClick={() => setCurrentView('liked')}
                      className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
                    >
                      <Heart className="w-4 h-4" />
                      Liked ({likedWatches.length})
                    </button>
                  </div>

                  {currentWatches.length > 0 && (
                    <div className="flex justify-center items-center h-[70vh] relative">
                      <AnimatePresence mode="wait">
                        <WatchCard
                          key={currentWatches[currentIndex].index}
                          watch={currentWatches[currentIndex]}
                          onLike={handleLike}
                          onPass={handlePass}
                          onShowSeries={handleShowSeries}
                        />
                      </AnimatePresence>
                    </div>
                  )}

                  {isLoading && (
                    <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
                      <div className="bg-white p-6 rounded-lg shadow-lg flex items-center gap-3">
                        <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
                        <span className="text-gray-800">Loading...</span>
                      </div>
                    </div>
                  )}
                </motion.div>
              </div>

              {/* Series View Section */}
              {seriesView.isVisible && (
                <div className="flex-shrink-0 bg-gray-900 border-t border-gray-700">
                  <div className="p-6">
                    <div className="flex justify-between items-center mb-4">
                      <h2 className="text-xl font-bold text-white">
                        {seriesView.seriesName} Series ({seriesView.watches.length} watches)
                      </h2>
                      <button
                        onClick={handleCloseSeriesView}
                        className="text-gray-400 hover:text-white transition-colors"
                      >
                        <X className="w-6 h-6" />
                      </button>
                    </div>
                    
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 max-h-96 overflow-y-auto">
                      {seriesView.watches.map((watch) => (
                        <div
                          key={watch.index}
                          className={`bg-gray-800 rounded-lg p-4 border-2 transition-colors ${
                            watch.index === seriesView.currentWatchIndex
                              ? 'border-green-500'
                              : 'border-gray-700 hover:border-gray-600'
                          }`}
                        >
                          <div className="aspect-square bg-gray-700 rounded-lg mb-3 flex items-center justify-center overflow-hidden">
                            <Image
                              src={watch.image_url || '/placeholder-watch.jpg'}
                              alt={`${watch.brand} ${watch.model}`}
                              width={200}
                              height={200}
                              className="object-contain max-w-full max-h-full"
                              onError={(e) => {
                                e.currentTarget.src = '/placeholder-watch.jpg';
                              }}
                            />
                          </div>
                          
                          <h3 className="text-white font-semibold text-sm mb-2 line-clamp-2">
                            {watch.brand} {watch.model}
                          </h3>
                          
                          <div className="space-y-1 text-xs text-gray-300">
                            <p>Price: <span className="text-green-400">${watch.price?.toLocaleString()}</span></p>
                            <p>Movement: {watch.specs?.movement || 'N/A'}</p>
                            <p>Case: {watch.specs?.case_material || 'N/A'}</p>
                          </div>
                          
                          <div className="flex gap-2 mt-3">
                            <button
                              onClick={() => handleLikeFromSeries(watch)}
                              className="flex-1 bg-red-600 hover:bg-red-700 text-white py-2 px-3 rounded text-xs transition-colors flex items-center justify-center gap-1"
                            >
                              <Heart className="w-3 h-3" />
                              Like
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          
          {currentView === 'liked' && (
            <motion.div
              key="liked"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="h-full overflow-y-auto"
            >
              {renderLikedView()}
            </motion.div>
          )}
          
          {currentView === 'history' && (
            <motion.div
              key="history"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="h-full overflow-y-auto"
            >
              {renderHistoryView()}
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Loading Overlay */}
      <AnimatePresence>
        {isLoading && currentScreen === 'main' && (
          <motion.div
            className="fixed inset-0 bg-black/20 flex items-center justify-center z-50"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="bg-white rounded-xl p-6 text-center shadow-lg">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
              >
                <Compass className="w-8 h-8 text-blue-600 mx-auto mb-2" />
              </motion.div>
              <p className="text-gray-600">Loading...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
} 