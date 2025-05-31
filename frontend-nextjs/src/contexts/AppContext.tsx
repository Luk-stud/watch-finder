'use client';

import React, { createContext, useContext, useReducer, useCallback, ReactNode } from 'react';
import type { AppState, Watch, ViewType } from '@/types';

interface AppContextType extends AppState {
  // Actions
  setCurrentWatches: (watches: Watch[]) => void;
  setCurrentIndex: (index: number) => void;
  addLikedWatch: (watch: Watch) => void;
  addDislikedWatch: (watch: Watch) => void;
  setStep: (step: number) => void;
  setSessionId: (sessionId: string | null) => void;
  setIsLoading: (isLoading: boolean) => void;
  setCurrentView: (view: ViewType) => void;
  resetState: () => void;
  getCurrentWatch: () => Watch | null;
  getTotalLiked: () => number;
  getTotalHistory: () => number;
}

type AppAction =
  | { type: 'SET_CURRENT_WATCHES'; payload: Watch[] }
  | { type: 'SET_CURRENT_INDEX'; payload: number }
  | { type: 'ADD_LIKED_WATCH'; payload: Watch }
  | { type: 'ADD_DISLIKED_WATCH'; payload: Watch }
  | { type: 'SET_STEP'; payload: number }
  | { type: 'SET_SESSION_ID'; payload: string | null }
  | { type: 'SET_IS_LOADING'; payload: boolean }
  | { type: 'SET_CURRENT_VIEW'; payload: ViewType }
  | { type: 'RESET_STATE' };

const initialState: AppState = {
  currentWatches: [],
  currentIndex: 0,
  likedWatches: [],
  dislikedWatches: [],
  step: 0,
  sessionId: null,
  isLoading: false,
  currentView: 'discover',
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_CURRENT_WATCHES':
      return { ...state, currentWatches: action.payload };
    
    case 'SET_CURRENT_INDEX':
      return { ...state, currentIndex: action.payload };
    
    case 'ADD_LIKED_WATCH': {
      const watch = action.payload;
      // Remove from disliked if it exists there
      const dislikedWatches = state.dislikedWatches.filter(w => w.index !== watch.index);
      // Add to liked if not already there
      const likedWatches = state.likedWatches.some(w => w.index === watch.index)
        ? state.likedWatches
        : [...state.likedWatches, watch];
      
      return { ...state, likedWatches, dislikedWatches };
    }
    
    case 'ADD_DISLIKED_WATCH': {
      const watch = action.payload;
      // Remove from liked if it exists there
      const likedWatches = state.likedWatches.filter(w => w.index !== watch.index);
      // Add to disliked if not already there
      const dislikedWatches = state.dislikedWatches.some(w => w.index === watch.index)
        ? state.dislikedWatches
        : [...state.dislikedWatches, watch];
      
      return { ...state, likedWatches, dislikedWatches };
    }
    
    case 'SET_STEP':
      return { ...state, step: action.payload };
    
    case 'SET_SESSION_ID':
      return { ...state, sessionId: action.payload };
    
    case 'SET_IS_LOADING':
      return { ...state, isLoading: action.payload };
    
    case 'SET_CURRENT_VIEW':
      return { ...state, currentView: action.payload };
    
    case 'RESET_STATE':
      return { ...initialState };
    
    default:
      return state;
  }
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  const setCurrentWatches = useCallback((watches: Watch[]) => {
    dispatch({ type: 'SET_CURRENT_WATCHES', payload: watches });
  }, []);

  const setCurrentIndex = useCallback((index: number) => {
    dispatch({ type: 'SET_CURRENT_INDEX', payload: index });
  }, []);

  const addLikedWatch = useCallback((watch: Watch) => {
    dispatch({ type: 'ADD_LIKED_WATCH', payload: watch });
  }, []);

  const addDislikedWatch = useCallback((watch: Watch) => {
    dispatch({ type: 'ADD_DISLIKED_WATCH', payload: watch });
  }, []);

  const setStep = useCallback((step: number) => {
    dispatch({ type: 'SET_STEP', payload: step });
  }, []);

  const setSessionId = useCallback((sessionId: string | null) => {
    dispatch({ type: 'SET_SESSION_ID', payload: sessionId });
  }, []);

  const setIsLoading = useCallback((isLoading: boolean) => {
    dispatch({ type: 'SET_IS_LOADING', payload: isLoading });
  }, []);

  const setCurrentView = useCallback((view: ViewType) => {
    dispatch({ type: 'SET_CURRENT_VIEW', payload: view });
  }, []);

  const resetState = useCallback(() => {
    dispatch({ type: 'RESET_STATE' });
  }, []);

  const getCurrentWatch = useCallback(() => {
    return state.currentWatches[state.currentIndex] || null;
  }, [state.currentWatches, state.currentIndex]);

  const getTotalLiked = useCallback(() => {
    return state.likedWatches.length;
  }, [state.likedWatches.length]);

  const getTotalHistory = useCallback(() => {
    return state.likedWatches.length + state.dislikedWatches.length;
  }, [state.likedWatches.length, state.dislikedWatches.length]);

  const value: AppContextType = {
    ...state,
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
    getTotalHistory,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
} 