// API Service for Watch Finder Modern Backend

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

// Flexible API URL configuration for different development scenarios
const getApiBaseUrl = (): string => {
  // Check environment variable first
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  
  // Auto-detect based on current host
  if (typeof window !== 'undefined') {
    const currentHost = window.location.hostname;
    
    // If running on network IP, use the same IP for backend
    if (currentHost !== 'localhost' && currentHost !== '127.0.0.1') {
      return `http://${currentHost}:5001/api`;
    }
    
    // Default for localhost
    return 'http://localhost:5001/api';
  }
  
  // Server-side fallback
  return 'http://localhost:5001/api';
};

const API_BASE_URL = getApiBaseUrl();

export class ApiService {
  private baseUrl: string;
  private sessionId: string | null = null;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T = unknown>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    // Remove /api from endpoint if base URL already includes it
    const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    const url = this.baseUrl.endsWith('/api') 
      ? `${this.baseUrl}${cleanEndpoint}`
      : `${this.baseUrl}/api${cleanEndpoint}`;

    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    };

    try {
      console.log(`🌐 Making API request to: ${url}`);  // Debug log
      const response = await fetch(url, defaultOptions);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`❌ API Error: ${response.status} - ${errorText}`);  // Debug log
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      console.log(`✅ API Response:`, data);  // Debug log

      if (data.status === 'error') {
        // Handle session expiration
        if (data.error_type === 'session_expired') {
          this.sessionId = null;
          throw new Error('Session expired. Please start a new session.');
        }
        throw new Error(data.message || 'API request failed');
      }

      return data;
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
}

export const apiService = new ApiService(); 