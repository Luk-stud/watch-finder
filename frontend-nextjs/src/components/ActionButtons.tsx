'use client';

import { motion } from 'framer-motion';
import { Heart, X } from 'lucide-react';

interface ActionButtonsProps {
  onLike: () => void;
  onPass: () => void;
  disabled?: boolean;
}

export default function ActionButtons({ onLike, onPass, disabled = false }: ActionButtonsProps) {
  return (
    <div className="flex items-center justify-center gap-4 py-6">
      {/* Pass Button */}
      <motion.button
        onClick={onPass}
        disabled={disabled}
        className="w-16 h-16 bg-white rounded-full shadow-lg border-2 border-red-100 hover:border-red-200 active:bg-red-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
        whileTap={{ scale: 0.95 }}
        whileHover={{ scale: 1.05 }}
      >
        <X className="w-6 h-6 text-red-500" />
      </motion.button>

      {/* Like Button */}
      <motion.button
        onClick={onLike}
        disabled={disabled}
        className="w-16 h-16 bg-white rounded-full shadow-lg border-2 border-green-100 hover:border-green-200 active:bg-green-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
        whileTap={{ scale: 0.95 }}
        whileHover={{ scale: 1.05 }}
      >
        <Heart className="w-6 h-6 text-green-500" />
      </motion.button>
    </div>
  );
} 