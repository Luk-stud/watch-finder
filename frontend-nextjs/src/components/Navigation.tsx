'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { User, X, Compass, Heart, History } from 'lucide-react';
import type { ViewType } from '@/types';
import { cn } from '@/lib/utils';

interface NavigationProps {
  currentView: ViewType;
  onViewChange: (view: ViewType) => void;
  onMenuClose?: () => void;
  likedCount: number;
}

const UserMenu = ({ onViewChange, onClose, likedCount }: { onViewChange: (view: ViewType) => void; onClose: () => void; likedCount: number; }) => {
  const menuItems = [
    {
      id: 'liked' as ViewType,
      label: 'Past Likes',
      icon: Heart,
    },
    {
      id: 'history' as ViewType,
      label: 'History',
      icon: History,
    },
  ];

  return (
    <motion.div 
      className="absolute top-12 right-0 bg-white border border-gray-200 rounded-lg shadow-xl p-4 w-64 z-[1000]"
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
    >
      <button onClick={onClose} className="absolute top-2 right-2 text-gray-500 hover:text-gray-700">
        <X size={20} />
      </button>
      <h3 className="text-lg font-semibold mb-3 text-gray-800">Menu</h3>
      <ul className="space-y-2">
        {menuItems.map(item => {
          const Icon = item.icon;
          return (
            <li key={item.id}>
              <button
                onClick={() => {
                  onViewChange(item.id);
                  onClose();
                }}
                className="w-full flex items-center gap-3 p-2 rounded-md text-gray-700 hover:bg-gray-100 hover:text-gray-900 transition-colors"
              >
                <Icon className="w-5 h-5" />
                <span className="text-sm font-medium">{item.label}</span>
              </button>
            </li>
          );
        })}
      </ul>
    </motion.div>
  );
};

export default function Navigation({ currentView, onViewChange, onMenuClose, likedCount }: NavigationProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const handleCloseMenu = () => {
    setIsMenuOpen(false);
    if (onMenuClose) {
      onMenuClose();
    }
  };

  return (
    <nav className="relative flex justify-end items-center">
      <motion.button
        onClick={toggleMenu}
        className={cn(
          'p-2 rounded-full bg-black/30 hover:bg-black/50 transition-colors',
          isMenuOpen ? 'bg-black/50' : 'bg-black/20'
        )}
        whileTap={{ scale: 0.95 }}
      >
        <User className="w-5 h-5 text-white" />
      </motion.button>

      {isMenuOpen && (
        <UserMenu 
          onViewChange={onViewChange} 
          onClose={handleCloseMenu} 
          likedCount={likedCount}
        />
      )}
    </nav>
  );
} 