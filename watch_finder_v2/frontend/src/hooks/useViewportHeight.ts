import { useEffect } from 'react';

/**
 * Custom hook to handle dynamic viewport height changes in mobile browsers
 * This is particularly important for Safari where the browser UI can show/hide
 */
export const useViewportHeight = () => {
  useEffect(() => {
    const updateViewportHeight = () => {
      // Get the actual viewport height
      const vh = window.innerHeight * 0.01;
      // Update the CSS custom property
      document.documentElement.style.setProperty('--viewport-height', `${window.innerHeight}px`);
      // Also set the vh unit for legacy support
      document.documentElement.style.setProperty('--vh', `${vh}px`);
    };

    // Set initial viewport height
    updateViewportHeight();

    // Update on resize (when browser UI shows/hides)
    window.addEventListener('resize', updateViewportHeight);
    
    // Update on orientation change
    window.addEventListener('orientationchange', () => {
      // Small delay to allow browser to finish orientation change
      setTimeout(updateViewportHeight, 100);
    });

    // Visual viewport API support for more precise control
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', updateViewportHeight);
    }

    // Cleanup event listeners
    return () => {
      window.removeEventListener('resize', updateViewportHeight);
      window.removeEventListener('orientationchange', updateViewportHeight);
      if (window.visualViewport) {
        window.visualViewport.removeEventListener('resize', updateViewportHeight);
      }
    };
  }, []);
}; 