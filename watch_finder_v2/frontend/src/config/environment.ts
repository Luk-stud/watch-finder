/**
 * Environment Configuration
 * Manage API URLs and settings for different environments
 */

// Vite environment variable access
interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_ENVIRONMENT?: string;
  readonly VITE_DEBUG?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

export type Environment = 'development' | 'production' | 'test';

export interface EnvironmentConfig {
  ENVIRONMENT: Environment;
  API_BASE_URL: string;
  FALLBACK_URLS: string[];
  DEBUG: boolean;
  FEATURES: {
    ENABLE_DEBUG_PANELS: boolean;
    ENABLE_CONNECTION_TEST: boolean;
    ENABLE_ANALYTICS: boolean;
  };
}

// Get environment from build-time variables or runtime detection
const getEnvironment = (): Environment => {
  // First check Vite environment variable
  if (import.meta?.env?.VITE_ENVIRONMENT) {
    return import.meta.env.VITE_ENVIRONMENT as Environment;
  }
  
  // Fallback to hostname detection
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    if (hostname.includes('netlify.app') || hostname.includes('netlify.com')) {
      return 'production';
    }
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return 'development';
    }
  }
  
  return 'development';
};

// Get API base URL based on environment
const getApiBaseUrl = (): string => {
  // First check for explicit environment variable
  if (import.meta?.env?.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  
  const environment = getEnvironment();
  
  switch (environment) {
    case 'production':
      // Railway backend URL - using actual deployment URL
      return 'https://web-production-36c3.up.railway.app/api';
    
    case 'development':
    default:
      // Smart network detection for development
      if (typeof window !== 'undefined') {
        const hostname = window.location.hostname;
        
        // If accessing via IP address (mobile on network), use the same IP for backend
        if (hostname.match(/^\d+\.\d+\.\d+\.\d+$/)) {
          return `http://${hostname}:5001/api`;
        }
        
        // If accessing via local network name, use the same hostname
        if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
          return `http://${hostname}:5001/api`;
        }
      }
      
      // Default to localhost for local development
      return 'http://localhost:5001/api';
  }
};

// Environment-specific configurations
const createEnvironmentConfig = (): EnvironmentConfig => {
  const environment = getEnvironment();
  const isDebug = import.meta?.env?.VITE_DEBUG === 'true' || environment === 'development';
  
  const baseConfig = {
    ENVIRONMENT: environment,
    API_BASE_URL: getApiBaseUrl(),
    DEBUG: isDebug
  };

  switch (environment) {
    case 'production':
      return {
        ...baseConfig,
  FALLBACK_URLS: [
          'https://web-production-36c3.up.railway.app/api',
          // Add additional production fallback URLs if needed
        ],
        FEATURES: {
          ENABLE_DEBUG_PANELS: false,
          ENABLE_CONNECTION_TEST: false,
          ENABLE_ANALYTICS: true,
        }
      };

    case 'development':
    default:
      // Dynamic fallback URLs based on current hostname
      const getFallbackUrls = (): string[] => {
        const fallbacks = [
          'http://localhost:5001/api',
          'http://127.0.0.1:5001/api'
        ];
        
        if (typeof window !== 'undefined') {
          const hostname = window.location.hostname;
          if (hostname.match(/^\d+\.\d+\.\d+\.\d+$/)) {
            fallbacks.unshift(`http://${hostname}:5001/api`);
          }
        }
        
        return fallbacks;
      };
      
      return {
        ...baseConfig,
        FALLBACK_URLS: getFallbackUrls(),
        FEATURES: {
          ENABLE_DEBUG_PANELS: true,
          ENABLE_CONNECTION_TEST: true,
          ENABLE_ANALYTICS: false,
        }
      };
  }
};

// Export the configuration
export const ENV_CONFIG = createEnvironmentConfig();

// Utility functions
export const isDevelopment = () => ENV_CONFIG.ENVIRONMENT === 'development';
export const isProduction = () => ENV_CONFIG.ENVIRONMENT === 'production';
export const shouldDebugLog = () => ENV_CONFIG.DEBUG;

// Log environment on startup
if (shouldDebugLog()) {
  console.group('🔧 Environment Configuration');
  console.log('Environment:', ENV_CONFIG.ENVIRONMENT);
  console.log('API Base URL:', ENV_CONFIG.API_BASE_URL);
  console.log('Debug Mode:', ENV_CONFIG.DEBUG);
  console.log('Features:', ENV_CONFIG.FEATURES);
  console.groupEnd();
} 