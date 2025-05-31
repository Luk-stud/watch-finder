'use client';

import { motion } from 'framer-motion';
import { Compass, Heart, History } from 'lucide-react';
import type { ViewType } from '@/types';
import { cn } from '@/lib/utils';

interface NavigationProps {
  currentView: ViewType;
  likedCount: number;
  onViewChange: (view: ViewType) => void;
}

export default function Navigation({ currentView, likedCount, onViewChange }: NavigationProps) {
  const navItems = [
    {
      id: 'discover' as ViewType,
      label: 'Discover',
      icon: Compass,
      count: null,
    },
    {
      id: 'liked' as ViewType,
      label: 'Liked',
      icon: Heart,
      count: likedCount,
    },
    {
      id: 'history' as ViewType,
      label: 'History',
      icon: History,
      count: null,
    },
  ];

  return (
    <nav className="flex bg-white rounded-xl border border-gray-200 p-1">
      {navItems.map((item) => {
        const Icon = item.icon;
        const isActive = currentView === item.id;

        return (
          <motion.button
            key={item.id}
            onClick={() => onViewChange(item.id)}
            className={cn(
              'relative flex items-center justify-center gap-2 px-4 py-2 rounded-lg font-medium transition-all flex-1',
              isActive
                ? 'text-blue-600 bg-blue-50'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
            )}
            whileTap={{ scale: 0.98 }}
          >
            {isActive && (
              <motion.div
                className="absolute inset-0 bg-blue-50 rounded-lg"
                layoutId="activeTab"
                transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              />
            )}
            
            <div className="relative z-10 flex items-center gap-1">
              <Icon className="w-4 h-4" />
              <span className="text-sm">{item.label}</span>
              {item.count !== null && item.count > 0 && (
                <motion.span
                  className="bg-blue-600 text-white text-xs px-1.5 py-0.5 rounded-full min-w-[18px] text-center leading-none"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ type: 'spring', stiffness: 300 }}
                >
                  {item.count}
                </motion.span>
              )}
            </div>
          </motion.button>
        );
      })}
    </nav>
  );
} 