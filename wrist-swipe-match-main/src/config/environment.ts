/**
 * Environment Configuration
 * Manage API URLs and settings for different environments
 */

export interface EnvironmentConfig {
  API_BASE_URL: string;
  FALLBACK_URLS: string[];
  IS_PRODUCTION: boolean;
  IS_DEVELOPMENT: boolean;
  DEBUG_LOGGING: boolean;
}

// Detect environment
const hostname = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
const isProductionEnv = hostname.includes('netlify.app') || hostname.includes('vercel.app') || hostname.includes('railway.app');
const isDevelopmentEnv = hostname === 'localhost' || hostname === '127.0.0.1' || hostname.startsWith('192.168.');

console.log('🔧 DEBUG Environment Detection:', {
  hostname,
  isProductionEnv,
  isDevelopmentEnv,
  windowAvailable: typeof window !== 'undefined',
  location: typeof window !== 'undefined' ? window.location.href : 'N/A'
});

// Production configuration
const PRODUCTION_CONFIG: EnvironmentConfig = {
  API_BASE_URL: 'https://watch-finder-backend.railway.app/api', // Update with your Railway domain
  FALLBACK_URLS: [],
  IS_PRODUCTION: true,
  IS_DEVELOPMENT: false,
  DEBUG_LOGGING: false,
};

// Development configuration
const DEVELOPMENT_CONFIG: EnvironmentConfig = {
  API_BASE_URL: 'http://localhost:5001/api',
  FALLBACK_URLS: [
    'http://localhost:5001/api',
    'http://127.0.0.1:5001/api',
    'http://192.168.50.5:5001/api', // Your backend's network IP
    'http://192.168.0.209:5001/api', // Frontend's network IP attempt
  ],
  IS_PRODUCTION: false,
  IS_DEVELOPMENT: true,
  DEBUG_LOGGING: true,
};

// Local network development configuration
const NETWORK_DEVELOPMENT_CONFIG: EnvironmentConfig = {
  API_BASE_URL: 'http://localhost:5001/api', // Always try localhost first
  FALLBACK_URLS: [
    'http://localhost:5001/api',
    'http://127.0.0.1:5001/api',
    'http://192.168.50.5:5001/api', // Known working backend IP
  ],
  IS_PRODUCTION: false,
  IS_DEVELOPMENT: true,
  DEBUG_LOGGING: true,
};

// Select configuration based on environment
export const getEnvironmentConfig = (): EnvironmentConfig => {
  // Check for environment variable override
  if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) {
    console.log('🔧 Using environment variable override:', import.meta.env.VITE_API_URL);
    return {
      API_BASE_URL: import.meta.env.VITE_API_URL,
      FALLBACK_URLS: DEVELOPMENT_CONFIG.FALLBACK_URLS, // Use development fallbacks
      IS_PRODUCTION: isProductionEnv,
      IS_DEVELOPMENT: isDevelopmentEnv,
      DEBUG_LOGGING: isDevelopmentEnv,
    };
  }

  if (isProductionEnv) {
    console.log('🚀 Using PRODUCTION configuration');
    return PRODUCTION_CONFIG;
  }

  // Force localhost to use development config
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    console.log('🏠 Using DEVELOPMENT configuration (localhost)');
    return DEVELOPMENT_CONFIG;
  }

  console.log('🌐 Using NETWORK DEVELOPMENT configuration for hostname:', hostname);
  return NETWORK_DEVELOPMENT_CONFIG;
};

// Export the current configuration
export const ENV_CONFIG = getEnvironmentConfig();

// Helper functions
export const checkIsProduction = () => ENV_CONFIG.IS_PRODUCTION;
export const checkIsDevelopment = () => ENV_CONFIG.IS_DEVELOPMENT;
export const shouldDebugLog = () => ENV_CONFIG.DEBUG_LOGGING;

// Log current configuration
if (typeof window !== 'undefined' && ENV_CONFIG.DEBUG_LOGGING) {
  console.log('🔧 Environment Configuration:', {
    hostname,
    apiBaseUrl: ENV_CONFIG.API_BASE_URL,
    fallbackUrls: ENV_CONFIG.FALLBACK_URLS,
    isProduction: ENV_CONFIG.IS_PRODUCTION,
    isDevelopment: ENV_CONFIG.IS_DEVELOPMENT,
  });
} 