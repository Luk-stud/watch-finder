'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Compass, Play, Loader } from 'lucide-react';

interface WelcomeScreenProps {
  onStart: () => Promise<void>;
}

export default function WelcomeScreen({ onStart }: WelcomeScreenProps) {
  const [isStarting, setIsStarting] = useState(false);

  const handleStart = async () => {
    setIsStarting(true);
    await onStart();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4 overflow-hidden">
      <motion.div
        className="text-center max-w-md mx-auto relative"
        initial={{ opacity: 1 }}
        animate={{ opacity: 1 }}
      >
        <motion.div
          className="absolute inset-0 flex items-center justify-center z-0"
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: isStarting ? 1.5 : 1, opacity: 1 }}
          transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
        >
          <Compass className="w-32 h-32 text-blue-400/30" />
          <motion.div
            className="absolute inset-0 rounded-full border-4 border-blue-400/20"
            style={{ width: '10rem', height: '10rem', margin: 'auto' }}
            animate={{ rotate: 360 }}
            transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
          />
        </motion.div>

        <AnimatePresence>
          {!isStarting && (
            <motion.div
              key="content"
              className="relative z-10"
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -30 }}
              transition={{ duration: 0.5 }}
            >
              <motion.header
                className="mb-12"
              >
                <h1 className="text-4xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                  <Compass className="w-8 h-8 text-blue-400" />
                  Watch Finder
                </h1>
                <p className="text-xl text-slate-300 leading-relaxed">
                  Discover your perfect timepiece with AI-powered visual similarity
                </p>
              </motion.header>

              <motion.button
                className="group bg-blue-600 hover:bg-blue-700 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 transform hover:scale-105 hover:shadow-xl hover:shadow-blue-500/25 flex items-center gap-3 mx-auto"
                onClick={handleStart}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Play className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                <span>Start Discovery</span>
              </motion.button>

              <motion.div
                className="mt-8 text-sm text-slate-400"
              >
                AI-powered discovery from 4,994 timepieces
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {isStarting && (
            <motion.div 
                key="loader"
                className="relative z-20 flex flex-col items-center justify-center mt-10"
                initial={{opacity: 0, scale: 0.5}}
                animate={{opacity:1, scale: 1}}
                transition={{delay: 0.3, type: 'spring'}}
            >
                <Loader className="w-12 h-12 text-blue-400 animate-spin" />
                <p className="text-slate-300 mt-3">Starting session...</p>
            </motion.div>
        )}
      </motion.div>
    </div>
  );
} 