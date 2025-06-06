import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import WatchCard from '../components/WatchCard';
import SpecsOverlay from '../components/SpecsOverlay';
import SeriesOverlay from '../components/SeriesOverlay';
import VariantsOverlay from '../components/VariantsOverlay';
import { Watch, formatPrice } from '../data/watchData';
import { apiService } from '../lib/api';
import { useFilters } from '../context/FiltersContext';
import { useViewportHeight } from '../hooks/useViewportHeight';
import { Heart, X, Loader2, AlertTriangle, RotateCcw, Settings, Filter } from 'lucide-react';

const Index = () => {
  const { filters, hasActiveFilters } = useFilters();
  
  // Core state - just current recommendations and UI state
  const [currentWatches, setCurrentWatches] = useState<Watch[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Track all watches that have been seen to exclude them from future recommendations
  const [seenWatchIds, setSeenWatchIds] = useState<Set<number>>(new Set());
  
  // Simple overlay state
  const [selectedWatch, setSelectedWatch] = useState<Watch | null>(null);
  const [showSpecs, setShowSpecs] = useState(false);

  // Use the viewport height hook for mobile browser compatibility
  useViewportHeight();

  const currentWatch = currentWatches[currentIndex];

  // Initialize session and get first watches
  useEffect(() => {
    initializeSession();
  }, []);

  const initializeSession = async (retryCount = 0) => {
    const maxRetries = 5;
    const baseDelay = 2000; // 2 seconds
    
    try {
      setIsLoading(true);
      setError(null);
      
      // Check backend health first
      await apiService.checkHealth();
      
      // If we have an existing session, try to get recommendations for it
      if (apiService.hasActiveSession()) {
        console.log('📱 Continuing existing session:', apiService.getSessionId());
        try {
          const response = await apiService.getRecommendations();
          
          if (response.status === 'success' && response.recommendations?.length > 0) {
            setCurrentWatches(response.recommendations);
            setCurrentIndex(0);
            
            // Don't reset seen watches for existing session
            const newSeenIds = new Set(response.recommendations.map(w => w.watch_id || w.index));
            setSeenWatchIds(newSeenIds);
            return; // Successfully continued session
          }
        } catch (sessionError) {
          console.warn('Failed to continue existing session, starting new one:', sessionError);
          // Fall through to start new session
        }
      }
      
      // Reset seen watches when starting new session
      setSeenWatchIds(new Set());
      
      // Start a new session and get initial recommendations
      const response = await apiService.startSession();
      
      if (response.status === 'success' && response.recommendations?.length > 0) {
        setCurrentWatches(response.recommendations);
        setCurrentIndex(0);
        
        // Mark these watches as seen
        const newSeenIds = new Set(response.recommendations.map(w => w.watch_id || w.index));
        setSeenWatchIds(newSeenIds);
      } else {
        throw new Error(response.message || 'Failed to start session');
      }
    } catch (err) {
      console.error('Failed to initialize session:', err);
      
      // Check if it's a system initialization error and we can retry
      const errorMessage = err instanceof Error ? err.message : 'Failed to connect to backend';
      const isInitializingError = errorMessage.includes('SYSTEM_INITIALIZING') || 
                                  errorMessage.includes('System still initializing');
      
      if (isInitializingError && retryCount < maxRetries) {
        const delay = baseDelay * Math.pow(1.5, retryCount); // Exponential backoff
        console.log(`🔄 System still initializing, retrying in ${delay/1000}s... (attempt ${retryCount + 1}/${maxRetries})`);
        
        // Show helpful message while retrying
        setError(`Backend is starting up, retrying in ${Math.round(delay/1000)}s... (${retryCount + 1}/${maxRetries})`);
        
        setTimeout(() => {
          initializeSession(retryCount + 1);
        }, delay);
        
        return; // Don't set final error state yet
      }
      
      // Set final error if we've exhausted retries or it's a different error
      if (isInitializingError) {
        setError('Backend is taking longer than expected to start. Please try refreshing the page in a moment.');
      } else {
        setError(errorMessage);
      }
    } finally {
      // Only set loading to false if we're not retrying
      if (retryCount === 0 || error !== null) {
        setIsLoading(false);
      }
    }
  };

  const getMoreRecommendations = async () => {
    try {
      setIsLoading(true);
      
      // Pass filter preferences and exclude already seen watches
      const excludeIds = Array.from(seenWatchIds);
      console.log('🚫 Excluding watch IDs:', excludeIds);
      
      const response = await apiService.getRecommendations(filters, excludeIds);
      
      console.log('🔍 Recommendations response:', response);
      
      // Check if we have recommendations (be more flexible about status field)
      const recommendations = response.recommendations || response;
      if (Array.isArray(recommendations) && recommendations.length > 0) {
        setCurrentWatches(recommendations);
        setCurrentIndex(0);
        
        // Add new watches to seen set
        const newWatchIds = recommendations.map(w => w.watch_id || w.index);
        setSeenWatchIds(prev => new Set([...prev, ...newWatchIds]));
        
        console.log('✅ Added new watches to seen list. Total seen:', seenWatchIds.size + newWatchIds.length);
      } else if (response.status === 'error') {
        throw new Error(response.message || 'No more recommendations available');
      } else {
        // If no recommendations but no error, try to get more or reset
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
    if (!currentWatch || isLoading) return;

    try {
      setIsLoading(true);
      
      // Send feedback to backend using watch_id instead of index
      await apiService.addFeedback(
        currentWatch.watch_id || currentWatch.index,
        direction === 'right' ? 'like' : 'dislike'
      );

      // If we have more watches in current batch, show next
      if (currentIndex < currentWatches.length - 1) {
        setCurrentIndex(prev => prev + 1);
      } else {
        // Otherwise get more recommendations
        await getMoreRecommendations();
      }
    } catch (err) {
      console.error('Error processing swipe:', err);
      setError(err instanceof Error ? err.message : 'Failed to process feedback');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSpecsClick = (watch: Watch) => {
    setSelectedWatch(watch);
    setShowSpecs(true);
  };

  const resetSession = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Reset seen watches
      setSeenWatchIds(new Set());
      
      // Use the API service reset method
      const response = await apiService.resetSession();
      
      if (response.status === 'success' && response.recommendations?.length > 0) {
        setCurrentWatches(response.recommendations);
        setCurrentIndex(0);
        
        // Mark these watches as seen
        const newSeenIds = new Set(response.recommendations.map(w => w.watch_id || w.index));
        setSeenWatchIds(newSeenIds);
      } else {
        throw new Error(response.message || 'Failed to reset session');
      }
    } catch (err) {
      console.error('Failed to reset session:', err);
      setError(err instanceof Error ? err.message : 'Failed to reset session');
    } finally {
      setIsLoading(false);
    }
  };

  // Error state
  if (error) {
    return (
      <div className="flex-viewport bg-background">
        <div className="flex items-center justify-center h-full p-6">
          <div className="text-center max-w-md">
            <div className="p-4 rounded-full bg-destructive/10 w-fit mx-auto mb-6">
              <AlertTriangle className="w-8 h-8 text-destructive" />
            </div>
            <h2 className="text-2xl font-semibold mb-3 text-foreground">Something went wrong</h2>
            <p className="text-muted-foreground mb-6 leading-relaxed">{error}</p>
          <button
            onClick={resetSession}
              className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground hover:bg-primary/90 h-10 px-4 py-2"
          >
            Try Again
          </button>
          </div>
        </div>
      </div>
    );
  }

  // Loading state (only show full screen loader when no watches)
  if (isLoading && currentWatches.length === 0) {
    const isRetrying = error && error.includes('retrying');
    
    return (
      <div className="flex-viewport bg-background">
        <div className="flex items-center justify-center h-full">
          <div className="text-center max-w-md px-6">
            <Loader2 className="w-8 h-8 mx-auto mb-4 animate-spin text-primary" />
            <h2 className="text-lg font-medium text-foreground mb-2">
              {isRetrying ? 'Starting backend...' : 'Loading recommendations...'}
            </h2>
            <p className="text-sm text-muted-foreground">
              {isRetrying 
                ? 'The AI system is initializing. This may take a few moments on first startup.'
                : 'Preparing your personalized watches'
              }
            </p>
            {isRetrying && (
              <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-950/30 rounded-lg border border-blue-200 dark:border-blue-800">
                <p className="text-sm text-blue-700 dark:text-blue-300">{error}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-viewport bg-background">
      {/* Header */}
      <header className="flex items-center justify-between p-4 border-b border-border bg-card">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
            <Settings className="w-4 h-4 text-primary-foreground" />
          </div>
          <h1 className="text-xl font-semibold text-foreground">
          WatchSwipe AI
        </h1>
        </div>
        
        <div className="flex items-center gap-2">
          <Link
            to="/filters"
            className={`inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-9 px-3 ${
              hasActiveFilters ? 'border-primary bg-primary/5' : ''
            }`}
            title="Filters"
          >
            <Filter className={`w-4 h-4 ${hasActiveFilters ? 'text-primary' : ''} sm:mr-2`} />
            <span className="hidden sm:inline">Filters</span>
            {hasActiveFilters && (
              <span className="ml-1 w-2 h-2 rounded-full bg-primary"></span>
            )}
          </Link>
          
          {/* Reset button only on larger screens since filters page also resets */}
          <button
            onClick={resetSession}
            disabled={isLoading}
            className="hidden md:inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-9 px-3"
            title="Reset Session"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </button>
          
        <Link
          to="/liked"
            className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 border border-input bg-background hover:bg-accent hover:text-accent-foreground h-9 px-3"
        >
            <Heart className="w-4 h-4 text-red-500 sm:mr-2" />
            <span className="hidden sm:inline">Liked</span>
        </Link>
      </div>
      </header>

      {/* Card Container */}
      <div className="card-container p-6">
        <div className="watch-card-wrapper">
          {currentWatch && (
            <div key={`watch-${currentWatch.index}`} className="relative z-10 h-full animate-scale-in">
              <WatchCard
                watch={currentWatch}
                onSwipe={handleSwipe}
                onSpecsClick={handleSpecsClick}
              />
            </div>
          )}

          {/* Loading overlay */}
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/80 backdrop-blur-sm rounded-3xl">
              <Loader2 className="w-6 h-6 text-primary animate-spin" />
            </div>
          )}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="action-buttons flex justify-center items-center gap-6 p-6 bg-card border-t border-border">
        <button
          onClick={() => handleSwipe('left')}
          className="w-14 h-14 rounded-full border-2 border-red-500 bg-red-500/10 hover:bg-red-500/20 flex items-center justify-center transition-all hover:scale-105 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
          disabled={isLoading}
        >
          <X className="w-6 h-6 text-red-500" />
        </button>
        
        <button
          onClick={() => handleSwipe('right')}
          className="w-14 h-14 rounded-full border-2 border-green-500 bg-green-500/10 hover:bg-green-500/20 flex items-center justify-center transition-all hover:scale-105 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-green-500 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
          disabled={isLoading}
        >
          <Heart className="w-6 h-6 text-green-500" />
        </button>
      </div>

      {/* Specs Overlay */}
      {showSpecs && selectedWatch && (
        <SpecsOverlay
          watch={selectedWatch}
          onClose={() => setShowSpecs(false)}
        />
      )}
    </div>
  );
};

export default Index;
