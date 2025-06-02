export interface WatchSpecs {
  brand: string;
  model: string;
  serie?: string;
  reference?: string;
  watch_type?: string;
  second_watch_type?: string;
  url: string;
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
}

export interface Watch {
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
  specifications?: Record<string, string | number | boolean>;
  variant_info?: {
    group_id: number;
    representative_index: number;
    variant_count: number;
    is_representative: boolean;
  };
  
  // New detailed specs structure
  specs?: WatchSpecs;
  main_image?: string;
  brand_website?: string;
}

export interface ApiResponse<T = unknown> {
  status: 'success' | 'error';
  message?: string;
  data?: T;
}

export interface SessionResponse {
  status: 'success' | 'error';
  seeds: Watch[];
  session_id: string;
  session_type: string;
  algorithm_used?: string;
  processing_time?: number;
}

export interface RecommendationsResponse {
  status: 'success' | 'error';
  recommendations: Watch[];
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

export interface VariantsResponse {
  status: 'success' | 'error';
  variants: Watch[];
  representative_index: number;
  group_id: number;
}

export interface AppState {
  currentWatches: Watch[];
  currentIndex: number;
  likedWatches: Watch[];
  dislikedWatches: Watch[];
  step: number;
  sessionId: string | null;
  isLoading: boolean;
  currentView: 'discover' | 'liked' | 'history';
}

export interface DragState {
  isDragging: boolean;
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
}

export interface RecommendationMetadata {
  algorithm_used: string;
  confidence: number;
  score: number;
  exploration_rate?: number;
  diversity_score?: number;
  is_seed?: boolean;
  is_precomputed_seed?: boolean;
  seed_style?: string;
  cluster_id?: number;
  recommendation_timestamp?: string;
  user_engagement_level?: string;
}

export interface EnhancedWatch extends Watch {
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

export type ViewType = 'discover' | 'liked' | 'history';
export type ActionType = 'like' | 'pass'; 