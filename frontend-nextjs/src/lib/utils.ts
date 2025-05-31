import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const CONFIG = {
  API_BASE_URL: typeof window !== 'undefined' && 
    (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:5001/api'
    : process.env.NEXT_PUBLIC_API_URL || 'https://web-production-a75cb.up.railway.app/api',
  SWIPE_THRESHOLD: 50,
  DRAG_THRESHOLD: 100,
  ANIMATION_DURATION: 300,
  MAX_RETRIES: 3,
  RETRY_DELAY: 1000,
};

export function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

export function formatPrice(price?: number): string {
  if (!price) return 'Price not available';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(price);
}

export function getPlaceholderImage(): string {
  return 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjQwMCIgdmlld0JveD0iMCAwIDQwMCA0MDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSI0MDAiIGhlaWdodD0iNDAwIiBmaWxsPSIjRjNGNEY2Ii8+CjxwYXRoIGQ9Ik0yMDAgMTAwQzE0NC40NzcgMTAwIDEwMCAxNDQuNDc3IDEwMCAyMDBTMTQ0LjQ3NyAzMDAgMjAwIDMwMFMyOTkuOTk5IDI1NS41MjMgMjk5Ljk5OSAyMDBTMjU1LjUyMiAxMDAgMjAwIDEwMFpNMjAwIDEyMEMyNDQuMTgzIDEyMCAyODAgMTU1LjgxNyAyODAgMjAwUzI0NC4xODMgMjgwIDIwMCAyODBTMTIwIDI0NC4xODMgMTIwIDIwMFMxNTUuODE3IDEyMCAyMDAgMTIwWiIgZmlsbD0iIzlDQTNBRiIvPgo8L3N2Zz4K';
}

export function debounce<T extends (...args: unknown[]) => unknown>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

/**
 * Get image URL - now uses external CDN URLs directly from the data
 */
export function getLocalImagePath(watch: any): string {
  // Use external CDN URL directly from main_image field
  const imageUrl = watch?.main_image || watch?.specs?.main_image || watch?.image_url;
  
  if (imageUrl) {
    return imageUrl;
  }
  
  return getPlaceholderImage();
}

/**
 * Get image URL - now uses local images first, falls back to external
 */
export function getImageUrl(watch: any): string {
  // Always try local images first
  const localPath = getLocalImagePath(watch);
  
  // For now, return local path directly
  // In production, you might want to add fallback logic here
  return localPath;
} 