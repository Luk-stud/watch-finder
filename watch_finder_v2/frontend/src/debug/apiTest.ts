/**
 * Debug API Test - Simple script to test API connections
 */

import { ENV_CONFIG } from '../config/environment';

export async function debugApiConnection() {
  console.log('🔧 DEBUG: Starting API connection test');
  console.log('🔧 DEBUG: Environment Config:', ENV_CONFIG);
  
  const testUrls = [
    'http://localhost:5001/api/health',
    'http://127.0.0.1:5001/api/health',
    'http://192.168.50.5:5001/api/health',
  ];
  
  for (const url of testUrls) {
    try {
      console.log(`🔧 DEBUG: Testing ${url}...`);
      
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log(`✅ DEBUG: SUCCESS ${url}:`, data);
      } else {
        console.log(`❌ DEBUG: HTTP Error ${response.status} for ${url}`);
      }
    } catch (error) {
      console.log(`❌ DEBUG: Network Error for ${url}:`, error);
    }
  }
}

// Auto-run on import
if (typeof window !== 'undefined') {
  console.log('🔧 DEBUG: Auto-running API connection test...');
  debugApiConnection();
} 