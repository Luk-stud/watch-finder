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

      // Get all series we've already seen
      const seenSeries = new Set([
        ...likedWatches.map(w => w.specs?.serie),
        ...dislikedWatches.map(w => w.specs?.serie),
        ...currentWatches.map(w => w.specs?.serie)
      ].filter(serie => serie && serie !== '-' && serie !== 'All'));
      
      const response = await apiService.getRecommendations(
        likedIndices,
        dislikedIndices,
        currentCandidates,
        step + 1
      );
      
      if (response.status === 'success' && response.recommendations.length > 0) {
        // Filter out watches from series we've already seen
        const filteredRecommendations = response.recommendations.filter(watch => {
          const series = watch.specs?.serie;
          return !series || series === '-' || series === 'All' || !seenSeries.has(series);
        });

        if (filteredRecommendations.length > 0) {
          setCurrentWatches(filteredRecommendations);
          setCurrentIndex(0);
          setStep(response.step);
        } else {
          // If all recommendations were filtered out, try getting more
          await getMoreRecommendations();
        }
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

      const response = await apiService.startSession(7);
      
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
      // Add the current watch's series to seen series if it has one
      if (watch.specs?.serie && watch.specs.serie !== '-' && watch.specs.serie !== 'All') {
        // Filter out any remaining watches from the same series in currentWatches
        const filteredWatches = currentWatches.filter(w => 
          w.index === watch.index || // Keep the current watch
          !w.specs?.serie || // Keep watches without series
          w.specs.serie === '-' || // Keep watches with no series
          w.specs.serie === 'All' || // Keep watches with 'All' series
          w.specs.serie !== watch.specs?.serie // Keep watches from different series
        );

        if (filteredWatches.length !== currentWatches.length) {
          setCurrentWatches(filteredWatches);
        }
      }

      addLikedWatch(watch);
      await moveToNext();
    } catch (error) {
      console.error('Error liking watch:', error);
    }
  }, [getCurrentWatch, currentWatches, addLikedWatch, moveToNext]);

  const handlePass = useCallback(async () => {
    const watch = getCurrentWatch();
    if (!watch) return;

    try {
      // Add the current watch's series to seen series if it has one
      if (watch.specs?.serie && watch.specs.serie !== '-' && watch.specs.serie !== 'All') {
        // Filter out any remaining watches from the same series in currentWatches
        const filteredWatches = currentWatches.filter(w => 
          w.index === watch.index || // Keep the current watch
          !w.specs?.serie || // Keep watches without series
          w.specs.serie === '-' || // Keep watches with no series
          w.specs.serie === 'All' || // Keep watches with different series
          w.specs.serie !== watch.specs?.serie // Keep watches from different series
        );

        if (filteredWatches.length !== currentWatches.length) {
          setCurrentWatches(filteredWatches);
        }
      }

      addDislikedWatch(watch);
      await moveToNext();
    } catch (error) {
      console.error('Error passing watch:', error);
    }
  }, [getCurrentWatch, currentWatches, addDislikedWatch, moveToNext]);

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
        <h2 className="text-xl font-bold mb-2" style={{ color: '#35342f' }}>Liked Watches</h2>
        <p style={{ color: '#35342f' }}>Your favorite timepieces</p>
      </div>
      
      {likedWatches.length === 0 ? (
        <div className="text-center py-16">
          <Heart className="w-12 h-12 mx-auto mb-4" style={{ color: '#37bbe4' }} />
          <h3 className="text-lg font-semibold mb-2" style={{ color: '#35342f' }}>No liked watches yet</h3>
          <p style={{ color: '#35342f' }}>Start discovering to build your collection</p>
        </div>
      ) : (
        <div className="space-y-4">
          {likedWatches.map((watch) => (
            <motion.div
              key={watch.index}
              className="rounded-xl p-4 shadow-sm border"
              style={{ backgroundColor: '#f1f2f0', borderColor: '#e1e0dd' }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="flex gap-4">
                <div className="w-16 h-16 rounded-lg flex items-center justify-center flex-shrink-0 overflow-hidden" style={{ backgroundColor: '#e1e0dd' }}>
                  <Image
                    src={watch.image_url || getPlaceholderImage()}
                    alt={`${watch.brand} ${watch.model}`}
                    width={64}
                    height={64}
                    className="object-contain w-full h-full max-w-full max-h-full"
                    sizes="64px"
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold truncate" style={{ color: '#35342f' }}>{watch.brand}</h3>
                  <p className="truncate" style={{ color: '#35342f' }}>{watch.model}</p>
                  {watch.price && (
                    <p className="font-medium mt-1" style={{ color: '#37bbe4' }}>
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
          <h2 className="text-xl font-bold mb-2" style={{ color: '#35342f' }}>Watch History</h2>
          <p style={{ color: '#35342f' }}>All watches you&apos;ve reviewed</p>
        </div>
        
        {allWatches.length === 0 ? (
          <div className="text-center py-16">
            <History className="w-12 h-12 mx-auto mb-4" style={{ color: '#37bbe4' }} />
            <h3 className="text-lg font-semibold mb-2" style={{ color: '#35342f' }}>No history yet</h3>
            <p style={{ color: '#35342f' }}>Start discovering to see your history</p>
          </div>
        ) : (
          <div className="space-y-3">
            {allWatches.map((watch) => {
              const isLiked = likedWatches.some(w => w.index === watch.index);
              return (
                <motion.div
                  key={watch.index}
                  className="flex items-center gap-3 p-3 rounded-lg shadow-sm border"
                  style={{ backgroundColor: '#f1f2f0', borderColor: '#e1e0dd' }}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                >
                  <div className="w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0 overflow-hidden" style={{ backgroundColor: '#e1e0dd' }}>
                    <Image
                      src={watch.image_url || getPlaceholderImage()}
                      alt={`${watch.brand} ${watch.model}`}
                      width={48}
                      height={48}
                      className="object-contain w-full h-full max-w-full max-h-full"
                      sizes="48px"
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium truncate" style={{ color: '#35342f' }}>{watch.brand}</h3>
                    <p className="text-sm truncate" style={{ color: '#35342f' }}>{watch.model}</p>
                  </div>
                  <div className={`p-2 rounded-full ${isLiked ? '' : ''}`} style={{ backgroundColor: isLiked ? '#37bbe4' : '#e1e0dd' }}>
                    {isLiked ? (
                      <Heart className="w-4 h-4 text-white" />
                    ) : (
                      <X className="w-4 h-4" style={{ color: '#35342f' }} />
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
      <div className="min-h-screen flex items-center justify-center p-6" style={{ backgroundColor: '#e1e0dd' }}>
        <div className="text-center max-w-md">
          <AlertTriangle className="w-16 h-16 mx-auto mb-4" style={{ color: '#37bbe4' }} />
          <h2 className="text-2xl font-bold mb-2" style={{ color: '#35342f' }}>Connection Error</h2>
          <p className="mb-6" style={{ color: '#35342f' }}>{error}</p>
          <button
            onClick={initializeApp}
            className="px-6 py-3 text-white rounded-lg font-medium hover:opacity-80 transition-colors"
            style={{ backgroundColor: '#37bbe4' }}
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  const handleMenuClose = () => {
    setCurrentView('discover');
  };

  return (
    <div 
      className="fixed inset-0 flex flex-col" 
      style={{ 
        backgroundColor: '#bfbabe',
        minHeight: '-webkit-fill-available',
        paddingTop: 'env(safe-area-inset-top)',
        paddingBottom: 'env(safe-area-inset-bottom)',
        paddingLeft: 'env(safe-area-inset-left)',
        paddingRight: 'env(safe-area-inset-right)',
        height: '100dvh'
      }}
    >
      {/* Main Content with enhanced container */}
      <main 
        className="max-w-sm mx-auto w-full flex-1 overflow-visible px-4"
        style={{
          height: 'calc(100dvh - env(safe-area-inset-top) - env(safe-area-inset-bottom))',
          maxHeight: '-webkit-fill-available'
        }}
      >
        <AnimatePresence mode="wait">
          {currentView === 'discover' && (
            <div className="flex-1 flex flex-col h-full overflow-visible">
              {/* Discovery Section */}
              <div className="flex-1 h-full">
                <motion.div className="h-full relative flex flex-col rounded-2xl">
                  {currentWatches.length > 0 && (
                    <div className="flex justify-center flex-1 overflow-visible pb-8 pt-4">
                      <div className="w-full max-w-sm h-full relative">
                        <AnimatePresence mode="popLayout" initial={false}>
                          <WatchCard
                            key={currentWatches[currentIndex].index}
                            watch={currentWatches[currentIndex]}
                            onLike={handleLike}
                            onPass={handlePass}
                            onShowSeries={handleShowSeries}
                            currentView={currentView}
                            likedCount={getTotalLiked()}
                            onViewChange={handleViewChange}
                            onMenuClose={handleMenuClose}
                          />
                        </AnimatePresence>
                      </div>
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

              {/* Series View Section - Now an Overlay */}
              <AnimatePresence>
                {seriesView.isVisible && (
                  <motion.div
                    className="fixed inset-0 z-40 bg-black/70 backdrop-blur-md flex items-center justify-center p-4"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <motion.div 
                      className="border shadow-2xl rounded-2xl w-full max-w-2xl max-h-[90vh] flex flex-col overflow-hidden"
                      style={{ backgroundColor: '#f1f2f0', borderColor: '#bfbabe' }}
                      initial={{ scale: 0.9, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      exit={{ scale: 0.9, opacity: 0 }}
                      transition={{ type: "spring", damping: 20, stiffness: 150 }}
                    >
                      <div className="p-6 flex-shrink-0 border-b" style={{ borderColor: '#bfbabe', backgroundColor: '#558a86' }}>
                        <div className="flex justify-between items-center">
                          <h2 className="text-xl font-bold text-white">
                            {seriesView.seriesName} Series ({seriesView.watches.length} watches)
                          </h2>
                          <button
                            onClick={handleCloseSeriesView}
                            className="text-white hover:text-white/80 transition-colors p-2 rounded-full hover:bg-white/20"
                          >
                            <X className="w-6 h-6" />
                          </button>
                        </div>
                      </div>
                      
                      <div className="p-6 flex-1 overflow-y-auto overscroll-behavior-y-contain">
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                          {seriesView.watches.map((watch) => (
                            <div
                              key={watch.index}
                              className={`rounded-xl p-4 border-2 transition-colors shadow-lg ${
                                watch.index === seriesView.currentWatchIndex
                                  ? ''
                                  : 'hover:opacity-80'
                              }`}
                              style={{ 
                                backgroundColor: '#f1f2f0', 
                                borderColor: watch.index === seriesView.currentWatchIndex ? '#37bbe4' : '#e1e0dd'
                              }}
                            >
                              <div className="aspect-square rounded-lg mb-3 flex items-center justify-center overflow-hidden" style={{ backgroundColor: '#e1e0dd' }}>
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
                              
                              <h3 className="font-semibold text-sm mb-2 line-clamp-2" style={{ color: '#35342f' }}>
                                {watch.brand} {watch.model}
                              </h3>
                              
                              <div className="space-y-1 text-xs mb-3" style={{ color: '#35342f' }}>
                                <p>Price: <span style={{ color: '#37bbe4' }}>${watch.price?.toLocaleString()}</span></p>
                                <p>Movement: {watch.specs?.movement || 'N/A'}</p>
                                <p>Case: {watch.specs?.case_material || 'N/A'}</p>
                              </div>
                              
                              <button
                                onClick={() => handleLikeFromSeries(watch)}
                                className="w-full text-white py-2 px-3 rounded-lg text-xs transition-colors flex items-center justify-center gap-1 shadow-md hover:opacity-80"
                                style={{ backgroundColor: '#37bbe4' }}
                              >
                                <Heart className="w-3 h-3" />
                                Like
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    </motion.div>
                  </motion.div>
                )}
              </AnimatePresence>
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
            <div className="rounded-xl p-6 text-center shadow-lg" style={{ backgroundColor: '#f1f2f0' }}>
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
              >
                <Compass className="w-8 h-8 mx-auto mb-2" style={{ color: '#558a86' }} />
              </motion.div>
              <p style={{ color: '#35342f' }}>Loading...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
} 