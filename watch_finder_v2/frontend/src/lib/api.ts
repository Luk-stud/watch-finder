// API Service for Watch Finder Backend

import { ENV_CONFIG, shouldDebugLog } from '../config/environment';
import { findWorkingApiUrl } from '../utils/connectionTest';

// Vite environment variable access
declare global {
  interface ImportMetaEnv {
    readonly VITE_API_URL?: string;
  }

  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

export interface WatchSpecs {
  brand: string;
  model: string;
  serie?: string;
  reference?: string;
  watch_type?: string;
  second_watch_type?: string;
  url?: string;
  price_usd?: string;
  msrp_eur?: string;
  launch_price_eur?: string;
  availability?: string;
  quantity_produced?: string;
  limited_edition_name?: string;
  launch_date?: string;
  case_material?: string;
  bottom_case_material?: string;
  diameter_mm?: string;
  thickness_with_crystal_mm?: string;
  thickness_without_crystal_mm?: string;
  lug_to_lug_mm?: string;
  case_shape?: string;
  case_coating_pvd_dlc?: string;
  case_finishing?: string;
  drill_lugs_on_case?: string;
  dial_color?: string;
  dial_material?: string;
  dial_type?: string;
  dial_pattern?: string;
  indices_type?: string;
  hands_types?: string;
  full_lume?: string;
  lume_1?: string;
  lume_2?: string;
  color_match_date_wheel?: string;
  movement?: string;
  winding?: string;
  power_reserve_hour?: string;
  mph?: string;
  hacking?: string;
  manual_winding?: string;
  complication_chronograph?: string;
  complication_date?: string;
  complication_dual_time?: string;
  complication_flying_tourbillon?: string;
  complication_gmt?: string;
  complication_jump_hour?: string;
  complication_power_reserve?: string;
  complication_small_second?: string;
  complication_sub_24_hour?: string;
  complication_sub_second?: string;
  complication_2nd_bezel_timezone?: string;
  complication_moonphase?: string;
  complication_world_time_zone?: string;
  complication_alarm?: string;
  complication_chronoscope?: string;
  crystal_material?: string;
  crystal_type_shape?: string;
  inner_coating?: string;
  outside_coating?: string;
  bezel_insert_material?: string;
  external_bezel_material?: string;
  bezel_type?: string;
  uni_bi_direction_bezel?: string;
  number_of_clicks?: string;
  internal_bezel?: string;
  main_crown_type?: string;
  other_crowns_function?: string;
  strap_1_material?: string;
  strap_2_material?: string;
  strap_3_material?: string;
  width_mm?: string;
  bracelet_tapper_to_clasp_mm?: string;
  bracelet_type?: string;
  bracelet_links_type?: string;
  bracelet_finishing?: string;
  strap_bracelet_attachment_system?: string;
  clasp_type?: string;
  clasp_material?: string;
  waterproofing_meters?: string;
  warranty_year?: string;
  brand_country?: string;
  made_in?: string;
  assembled_in?: string;
  country?: string;
  waterproofing?: string;
  specific_info_from_brand?: string;
  brand_website?: string;
}

export interface ModernWatch {
  // Core watch data
  index?: number;
  brand: string;
  model: string;
  reference?: string;
  case_material?: string;
  case_diameter?: string;
  movement?: string;
  water_resistance?: string;
  price?: number;
  price_range?: string;
  description?: string;
  image_url?: string;
  main_image?: string;
  local_image_path?: string;
  product_url?: string;
  source?: string;
  local_image?: boolean;
  specifications?: Record<string, string | number | boolean>;
  specs?: WatchSpecs;
  brand_website?: string;
  
  // Algorithm fields
  algorithm?: string;
  confidence?: number;
  score?: number;
  watch_id?: string | number;
}

export interface SessionResponse {
  status: 'success' | 'error';
  session_id: string;
  recommendations: ModernWatch[];
  message?: string;
}

export interface RecommendationsResponse {
  status: 'success' | 'error';
  recommendations: ModernWatch[];
  message?: string;
}

export interface LikedWatchesResponse {
  status: 'success' | 'error';
  liked_watches: ModernWatch[];
  message?: string;
}

export interface FeedbackResponse {
  status: 'success' | 'error';
  message?: string;
}

export interface HealthResponse {
  status: 'healthy' | 'unhealthy' | 'error' | 'initializing';
  timestamp?: string;
  system_metrics?: {
    status: string;
    active_sessions: number;
    max_concurrent_sessions: number;
    global_metrics: any;
    session_analytics_count: number;
    system_info: any;
  };
  active_sessions?: number;  // Legacy support
  message?: string;
}

export interface WatchDetailsResponse {
  status: 'success' | 'error';
  watch: ModernWatch;
  message?: string;
}

// Use environment configuration
const API_BASE_URL = ENV_CONFIG.API_BASE_URL;

export class ApiService {
  private baseUrl: string;
  private sessionId: string | null = null;
  private fallbackUrls: string[] = [];
  private hasTestedConnections: boolean = false;
  
  // Session persistence key
  private static readonly SESSION_STORAGE_KEY = 'watch_finder_session_id';

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
    
    // Setup fallback URLs from environment configuration
    this.fallbackUrls = ENV_CONFIG.FALLBACK_URLS;
    
    // If the primary URL contains an unknown IP, fix it immediately
    if (this.baseUrl.includes('192.168.0.209')) {
      console.warn('🔧 Detected wrong IP in primary URL, switching to localhost');
      this.baseUrl = 'http://localhost:5001/api';
    }
    
    // Load existing session ID from localStorage
    this.loadSessionId();
    
    if (shouldDebugLog()) {
      console.log('🔧 API Service initialized with:', {
        baseUrl: this.baseUrl,
        fallbackUrls: this.fallbackUrls,
        existingSessionId: this.sessionId ? 'loaded' : 'none'
      });
    }
  }

  private loadSessionId(): void {
    try {
      const storedSessionId = localStorage.getItem(ApiService.SESSION_STORAGE_KEY);
      if (storedSessionId) {
        this.sessionId = storedSessionId;
        if (shouldDebugLog()) {
          console.log('📱 Loaded session ID from storage:', storedSessionId);
        }
      }
    } catch (error) {
      console.warn('Failed to load session ID from localStorage:', error);
    }
  }

  private saveSessionId(sessionId: string): void {
    try {
      localStorage.setItem(ApiService.SESSION_STORAGE_KEY, sessionId);
      if (shouldDebugLog()) {
        console.log('💾 Saved session ID to storage:', sessionId);
      }
    } catch (error) {
      console.warn('Failed to save session ID to localStorage:', error);
    }
  }

  private clearSessionId(): void {
    try {
      localStorage.removeItem(ApiService.SESSION_STORAGE_KEY);
      this.sessionId = null;
      if (shouldDebugLog()) {
        console.log('🗑️ Cleared session ID from storage');
      }
    } catch (error) {
      console.warn('Failed to clear session ID from localStorage:', error);
    }
  }

  private async findBestApiUrl(): Promise<string | null> {
    if (this.hasTestedConnections) {
      return null; // Don't test again in the same session
    }
    
    const urlsToTest = [this.baseUrl, ...this.fallbackUrls];
    
    if (shouldDebugLog()) {
      console.log('🔍 Testing API URLs:', urlsToTest);
    }
    
    const workingUrl = await findWorkingApiUrl(urlsToTest);
    
    if (workingUrl && workingUrl !== this.baseUrl) {
      console.log(`🔄 Switching to working URL: ${workingUrl}`);
      this.baseUrl = workingUrl;
    }
    
    this.hasTestedConnections = true;
    return workingUrl;
  }

  private async requestWithFallback<T = unknown>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    let primaryError: Error;

    try {
      return await this.makeRequest<T>(this.baseUrl, endpoint, options);
    } catch (error) {
      primaryError = error as Error;
      
      if (shouldDebugLog()) {
        console.warn(`⚠️ Primary URL failed (${this.baseUrl}), trying fallbacks...`);
      }
      
      // Try fallback URLs
      for (const fallbackUrl of this.fallbackUrls) {
        if (fallbackUrl === this.baseUrl) continue; // Skip if it's the same as primary
        
        try {
          if (shouldDebugLog()) {
            console.log(`🔄 Trying fallback: ${fallbackUrl}`);
          }
          
          const result = await this.makeRequest<T>(fallbackUrl, endpoint, options);
          
          // If successful, update the primary URL
          this.baseUrl = fallbackUrl;
          console.log(`✅ Fallback successful, updated primary URL to: ${fallbackUrl}`);
          
          return result;
        } catch (fallbackError) {
          if (shouldDebugLog()) {
            console.warn(`❌ Fallback failed: ${fallbackUrl}`, fallbackError);
          }
          continue;
        }
      }
      
      // As a last resort, try to find a working URL
      if (!this.hasTestedConnections) {
        const workingUrl = await this.findBestApiUrl();
        if (workingUrl && workingUrl !== this.baseUrl) {
          try {
            return await this.makeRequest<T>(workingUrl, endpoint, options);
          } catch (testError) {
            if (shouldDebugLog()) {
              console.warn('❌ Even the tested URL failed:', testError);
            }
          }
        }
      }
      
      // If everything failed, throw the original error
      throw primaryError;
    }
  }

  private async makeRequest<T = unknown>(
    baseUrl: string,
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    // Remove /api from endpoint if base URL already includes it
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    const url = baseUrl.endsWith('/api') 
      ? `${baseUrl}${cleanEndpoint}`
      : `${baseUrl}/api${cleanEndpoint}`;

    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        // Add session header if we have a session
        ...(this.sessionId && { 'X-Session-ID': this.sessionId }),
      },
      ...options,
    };

    if (shouldDebugLog()) {
      console.log(`🌐 Making API request to: ${url}`);
      console.log(`📋 Headers:`, defaultOptions.headers);
    }

    const response = await fetch(url, defaultOptions);

    if (!response.ok) {
      const errorText = await response.text();
      if (shouldDebugLog()) {
        console.error(`❌ API Error: ${response.status} - ${errorText}`);
      }
      throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    if (shouldDebugLog()) {
      console.log(`✅ API Response:`, data);
    }

    if (data.status === 'error') {
      // Handle session expiration
      if (data.message?.includes('Invalid session') || data.message?.includes('session')) {
        this.clearSessionId();
        throw new Error('Session expired. Please start a new session.');
      }
      throw new Error(data.message || 'API request failed');
    }

    return data;
  }

  private async request<T = unknown>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    try {
      return await this.requestWithFallback<T>(endpoint, options);
    } catch (error) {
      console.error(`❌ API Error [${endpoint}]:`, error);
      throw error;
    }
  }

  getSessionId(): string | null {
    return this.sessionId;
  }

  hasActiveSession(): boolean {
    return this.sessionId !== null;
  }

  async checkHealth(): Promise<HealthResponse> {
    const data = await this.request<HealthResponse>('/health');

    // Handle production backend format
    if (data.system_metrics) {
    return {
        ...data,
        active_sessions: data.system_metrics.active_sessions || 0
    };
    }
    
    // Legacy format support
    return data;
  }

  async startSession(): Promise<SessionResponse> {
    if (shouldDebugLog()) {
      console.log('🚀 Starting new session...');
    }
    
    try {
      // First create a session with the backend
      const sessionData = await this.request<any>('/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: this.sessionId
        })
    });

      // Update session ID from response
      if (sessionData.session_id) {
        this.sessionId = sessionData.session_id;
        this.saveSessionId(this.sessionId);
      }
      
      // Then get initial recommendations
      const recommendations = await this.getRecommendations();
      
      return {
        status: 'success',
        session_id: this.sessionId!,
        recommendations: recommendations.recommendations || []
      };
      
    } catch (error) {
      console.error('❌ Error starting session:', error);
      return {
        status: 'error',
        session_id: '',
        recommendations: [],
        message: error instanceof Error ? error.message : 'Failed to start session'
      };
    }
  }

  async getRecommendations(filterPreferences?: any, excludeIds?: (string | number)[]): Promise<RecommendationsResponse> {
    if (shouldDebugLog()) {
      console.log('🎯 Getting recommendations with filter preferences:', filterPreferences, 'excluding:', excludeIds);
    }
    
    const requestBody: any = {
      session_id: this.sessionId,
      exclude_ids: excludeIds || []
    };
    
    // Add filter preferences if provided
    if (filterPreferences) {
      requestBody.filter_preferences = filterPreferences;
    }

    return this.request<RecommendationsResponse>('/recommendations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });
  }

  async addFeedback(
    watchId: number,
    action: 'like' | 'dislike'
  ): Promise<FeedbackResponse> {
    if (!this.sessionId) {
      throw new Error('No session available. Please start a session first.');
    }

    return this.request<FeedbackResponse>('/feedback', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: this.sessionId,
        watch_id: watchId, 
        feedback: action  // Changed from 'type' to 'feedback' to match production_linucb_app.py
      }),
    });
  }

  async getLikedWatches(): Promise<LikedWatchesResponse> {
    if (!this.sessionId) {
      throw new Error('No active session. Please start a new session.');
    }

    return this.request<LikedWatchesResponse>('/liked-watches', {
      method: 'GET',
    });
  }

  async getWatchDetails(watchId: number): Promise<WatchDetailsResponse> {
    return this.request<WatchDetailsResponse>(`/watch/${watchId}/details`, {
      method: 'GET',
    });
  }

  // Legacy methods for backward compatibility - these will use the watch_id instead of index
  async getSeries(watchIndex: number): Promise<RecommendationsResponse> {
    // For now, just return more recommendations since LinUCB doesn't have series concept
    return this.getRecommendations();
  }

  async getVariants(watchIndex: number): Promise<RecommendationsResponse> {
    // For now, just return more recommendations since LinUCB doesn't have variants concept
    return this.getRecommendations();
  }

  async getStats(): Promise<{ status: string; message: string }> {
    // Not implemented in LinUCB backend yet
    return { status: 'error', message: 'Stats not implemented in LinUCB backend' };
  }

  async getDiagnostics(): Promise<{ status: string; message: string }> {
    // Not implemented in LinUCB backend yet
    return { status: 'error', message: 'Diagnostics not implemented in LinUCB backend' };
  }

  async resetSession(): Promise<SessionResponse> {
    try {
      // If we have an existing session, delete it first
      if (this.sessionId) {
        await this.request(`/session/${this.sessionId}`, {
          method: 'DELETE',
        });
      }
    } catch (error) {
      // If deletion fails, log it but continue with creating new session
      console.warn('Failed to delete old session, creating new one anyway:', error);
    }

    // Clear the current session ID
    this.clearSessionId();

    // Create a new session
    return this.startSession();
  }
}

export const apiService = new ApiService(); 