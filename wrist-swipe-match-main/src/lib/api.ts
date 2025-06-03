// API Service for Watch Finder Modern Backend

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
  index: number;
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
  
  // Modern backend enhancement fields
  algorithm?: string;
  confidence?: number;
  score?: number;
  is_seed?: boolean;
  is_precomputed_seed?: boolean;
  seed_style?: string;
  cluster_id?: number;
  recommendation_timestamp?: string;
  user_engagement_level?: string;
}

export interface SessionResponse {
  status: 'success' | 'error';
  seeds: ModernWatch[];
  session_id: string;
  session_type: string;
  algorithm_used?: string;
  processing_time?: number;
}

export interface RecommendationsResponse {
  status: 'success' | 'error';
  recommendations: ModernWatch[];
  step: number;
  total_processed: number;
  message?: string;
  confidence_scores?: number[];
  diversity_score?: number;
  exploration_rate?: number;
  algorithm_used?: string;
  processing_time?: number;
  user_profile_summary?: {
    engagement_level: string;
    total_feedback: number;
    preference_clusters: number;
    exploration_rate: number;
    dominant_brands: Record<string, number>;
    dominant_styles: Record<string, number>;
    session_duration: number;
  };
  next_exploration_suggestions?: string[];
}

export interface SeriesResponse {
  status: 'success' | 'error';
  session_id: string;
  watch_index: number;
  series_count: number;
  series_watches: ModernWatch[];
}

export interface VariantsResponse {
  status: 'success' | 'error';
  watch_index: number;
  target_watch: ModernWatch;
  brand: string;
  model: string;
  signature: string;
  variant_count: number;
  variants: ModernWatch[];
  info: string;
}

// Use environment configuration
const API_BASE_URL = ENV_CONFIG.API_BASE_URL;

export class ApiService {
  private baseUrl: string;
  private sessionId: string | null = null;
  private fallbackUrls: string[] = [];
  private hasTestedConnections: boolean = false;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
    
    // Setup fallback URLs from environment configuration
    this.fallbackUrls = ENV_CONFIG.FALLBACK_URLS;
    
    // If the primary URL contains an unknown IP, fix it immediately
    if (this.baseUrl.includes('192.168.0.209')) {
      console.warn('🔧 Detected wrong IP in primary URL, switching to localhost');
      this.baseUrl = 'http://localhost:5001/api';
    }
    
    if (shouldDebugLog()) {
      console.log(`🔧 API Service initialized:`);
      console.log(`   Primary URL: ${this.baseUrl}`);
      console.log(`   Fallback URLs:`, this.fallbackUrls);
    }
  }

  private async findBestApiUrl(): Promise<string | null> {
    if (this.hasTestedConnections) {
      return null; // Don't test multiple times
    }
    
    this.hasTestedConnections = true;
    
    const candidateUrls = [this.baseUrl, ...this.fallbackUrls];
    if (shouldDebugLog()) {
      console.log('🔍 Testing API connections to find working endpoint...');
    }
    
    const workingUrl = await findWorkingApiUrl(candidateUrls);
    
    if (workingUrl && workingUrl !== this.baseUrl) {
      if (shouldDebugLog()) {
        console.log(`🎯 Found working API URL: ${workingUrl}`);
      }
      this.baseUrl = workingUrl;
      return workingUrl;
    }
    
    return null;
  }

  private async requestWithFallback<T = unknown>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    // Try primary URL first
    try {
      return await this.makeRequest<T>(this.baseUrl, endpoint, options);
    } catch (primaryError) {
      if (shouldDebugLog()) {
        console.warn(`❌ Primary URL failed (${this.baseUrl}):`, primaryError);
      }
      
      // Try fallback URLs if available
      for (let i = 0; i < this.fallbackUrls.length; i++) {
        const fallbackUrl = this.fallbackUrls[i];
        if (shouldDebugLog()) {
          console.log(`🔄 Trying fallback URL ${i + 1}/${this.fallbackUrls.length}: ${fallbackUrl}`);
        }
        
        try {
          const result = await this.makeRequest<T>(fallbackUrl, endpoint, options);
          if (shouldDebugLog()) {
            console.log(`✅ Fallback URL ${i + 1} succeeded! Updating primary URL.`);
          }
          
          // Update primary URL for future requests
          this.baseUrl = fallbackUrl;
          return result;
        } catch (fallbackError) {
          if (shouldDebugLog()) {
            console.warn(`❌ Fallback URL ${i + 1} failed:`, fallbackError);
          }
        }
      }
      
      // If all fallbacks failed, try to find a working URL via connection testing
      if (!this.hasTestedConnections) {
        if (shouldDebugLog()) {
          console.log('🔍 All URLs failed, running connection diagnostics...');
        }
        
        const workingUrl = await this.findBestApiUrl();
        if (workingUrl) {
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
      },
      ...options,
    };

    if (shouldDebugLog()) {
      console.log(`🌐 Making API request to: ${url}`);
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
      if (data.error_type === 'session_expired') {
        this.sessionId = null;
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

  async checkHealth(): Promise<{ status: string; session_manager_initialized: boolean; active_sessions: number }> {
    const data = await this.request<any>('/health');

    // Backend health response format: { status: "healthy", system_status: { active_sessions: number, ... } }
    // The session manager is considered initialized if we get a successful response
    const isHealthy = data.status === 'healthy';
    const activeSessions = data.system_status?.active_sessions || 0;

    return {
      status: data.status,
      session_manager_initialized: isHealthy, // If backend responds, session manager is working
      active_sessions: activeSessions
    };
  }

  async startSession(numSeeds: number = 7): Promise<SessionResponse> {
    const response = await this.request<SessionResponse>('/start-session', {
      method: 'POST',
      body: JSON.stringify({ num_seeds: numSeeds }),
    });

    // Store session ID for future requests
    if (response.session_id) {
      this.sessionId = response.session_id;
    }

    return response;
  }

  async getRecommendations(
    likedIndices: number[],
    dislikedIndices: number[],
    currentCandidates: number[],
    step: number,
    numRecommendations: number = 7
  ): Promise<RecommendationsResponse> {
    if (!this.sessionId) {
      throw new Error('No active session. Please start a new session.');
    }

    return this.request<RecommendationsResponse>('/get-recommendations', {
      method: 'POST',
      body: JSON.stringify({
        session_id: this.sessionId,
        liked_indices: likedIndices,
        disliked_indices: dislikedIndices,
        current_candidates: currentCandidates,
        num_recommendations: numRecommendations,
        step: step,
        exploration_factor: 0.3,
        diversity_threshold: 0.7
      }),
    });
  }

  async getSeries(watchIndex: number): Promise<SeriesResponse> {
    if (!this.sessionId) {
      throw new Error('No active session. Please start a new session.');
    }

    return this.request(`/get-series?session_id=${this.sessionId}&index=${watchIndex}`);
  }

  async addFeedback(
    watchIndex: number,
    action: 'like' | 'dislike',
    confidence: number = 1.0
  ): Promise<{ status: string; message?: string }> {
    if (!this.sessionId) {
      throw new Error('No active session. Please start a new session.');
    }

    return this.request('/add-feedback', {
      method: 'POST',
      body: JSON.stringify({
        session_id: this.sessionId,
        watch_index: watchIndex,
        feedback_type: action,
        confidence,
      }),
    });
  }

  async getStats(): Promise<{
    status: 'success' | 'error';
    total_watches: number;
    brands: Record<string, number>;
    price_ranges: Record<string, number>;
    embedding_dimension: number;
    active_sessions: number;
    performance_metrics?: {
      total_requests: number;
      avg_response_time: number;
      cache_hit_rate: number;
      user_satisfaction: number;
    };
  }> {
    return this.request(`/stats`);
  }

  async getDiagnostics(): Promise<{
    status: string;
    recommendation_engine: string;
    similarity_index: string;
    smart_seeds_loaded: boolean;
    performance_metrics: {
      total_requests: number;
      avg_response_time: number;
      cache_hit_rate: number;
      user_satisfaction: number;
    };
  }> {
    return this.request('/diagnostics');
  }

  async getVariants(watchIndex: number): Promise<VariantsResponse> {
    if (!this.sessionId) {
      throw new Error('No active session. Please start a new session.');
    }

    return this.request(`/get-variants/${watchIndex}`);
  }
}

export const apiService = new ApiService(); 