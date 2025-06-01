'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Compass, Play, Loader } from 'lucide-react';
import { apiService } from '@/lib/api';

interface WelcomeScreenProps {
  onStart: () => Promise<void>;
}

export default function WelcomeScreen({ onStart }: WelcomeScreenProps) {
  const [isStarting, setIsStarting] = useState(false);
  const [watchCount, setWatchCount] = useState(2026); // Default fallback

  useEffect(() => {
    // Fetch actual watch count from API
    const fetchWatchCount = async () => {
      try {
        const stats = await apiService.getStats();
        if (stats.status === 'success') {
          setWatchCount(stats.total_watches);
        }
      } catch (error) {
        console.warn('Failed to fetch watch count, using default:', error);
        // Keep default value of 2026
      }
    };

    fetchWatchCount();
  }, []);

  const handleStart = async () => {
    setIsStarting(true);
    try {
      await onStart();
    } catch (error) {
      console.error('Failed to start:', error);
      setIsStarting(false);
    }
  };

  return (
    <div 
      className="min-h-screen flex items-center justify-center relative overflow-hidden"
      style={{ 
        backgroundColor: '#bfbabe',
        paddingTop: 'env(safe-area-inset-top)',
        paddingBottom: 'env(safe-area-inset-bottom)',
        paddingLeft: 'env(safe-area-inset-left)',
        paddingRight: 'env(safe-area-inset-right)',
        height: '100dvh'
      }}
    >
      {/* Animated background pattern */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-20 left-10 w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: '#558a86' }} />
        <div className="absolute top-40 right-20 w-1 h-1 rounded-full animate-pulse delay-300" style={{ backgroundColor: '#a63e14' }} />
        <div className="absolute bottom-32 left-20 w-1.5 h-1.5 rounded-full animate-pulse delay-700" style={{ backgroundColor: '#558a86' }} />
        <div className="absolute bottom-20 right-10 w-2 h-2 rounded-full animate-pulse delay-1000" style={{ backgroundColor: '#a63e14' }} />
      </div>

      <AnimatePresence>
        {!isStarting && (
          <motion.div
            key="content"
            className="relative z-10 text-center px-6"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            transition={{ duration: 0.5 }}
          >
            <motion.header
              className="mb-12"
            >
              <h1 className="text-4xl font-bold mb-4 flex items-center justify-center gap-3" style={{ color: '#35342f' }}>
                <Compass className="w-8 h-8" style={{ color: '#558a86' }} />
                Watch Finder
              </h1>
              <p className="text-xl leading-relaxed" style={{ color: '#35342f' }}>
                Discover your perfect timepiece with AI-powered visual similarity
              </p>
            </motion.header>

            <motion.button
              className="group text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 transform hover:scale-105 hover:shadow-xl flex items-center gap-3 mx-auto"
              style={{ 
                backgroundColor: '#558a86',
                boxShadow: '0 10px 25px rgba(85, 138, 134, 0.3)'
              }}
              onClick={handleStart}
              whileHover={{ 
                scale: 1.05,
                boxShadow: '0 15px 35px rgba(85, 138, 134, 0.4)'
              }}
              whileTap={{ scale: 0.95 }}
            >
              <Play className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              <span>Start Discovery</span>
            </motion.button>

            <motion.div
              className="mt-8 text-sm"
              style={{ color: '#35342f' }}
            >
              AI-powered discovery from {watchCount.toLocaleString()} timepieces
            </motion.div>
          </motion.div>
        )}

        {isStarting && (
          <motion.div
            key="loading"
            className="relative z-10 text-center px-6"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
              className="w-16 h-16 mx-auto mb-6"
            >
              <Compass className="w-full h-full" style={{ color: '#558a86' }} />
            </motion.div>
            <h2 className="text-2xl font-bold mb-4" style={{ color: '#35342f' }}>
              Preparing Your Discovery
            </h2>
            <p className="text-lg" style={{ color: '#35342f' }}>
              Loading AI-powered recommendations...
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
} 