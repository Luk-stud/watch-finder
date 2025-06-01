import { CONFIG } from './utils';
import type {
  ApiResponse,
  SessionResponse,
  RecommendationsResponse,
  VariantsResponse,
  Watch,
} from '@/types';

export class ApiService {
  private baseUrl: string;
  private sessionId: string | null = null;

  constructor(baseUrl: string = CONFIG.API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T = unknown>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    };

    try {
      const response = await fetch(url, defaultOptions);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

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
      console.error(`API Error [${endpoint}]:`, error);
      throw error;
    }
  }

  getSessionId(): string | null {
    return this.sessionId;
  }

  async checkHealth(): Promise<{ status: string; session_manager_initialized: boolean; active_sessions: number }> {
    const data = await this.request<{ 
      status: string; 
      session_manager_initialized: boolean; 
      active_sessions: number;
    }>('/health');

    if (!data.session_manager_initialized) {
      throw new Error('Session manager not initialized');
    }

    return data;
  }

  async startSession(numSeeds: number = 1): Promise<SessionResponse> {
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
    step: number
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
        step: step,
      }),
    });
  }

  async getVariants(watchIndex: number): Promise<VariantsResponse> {
    if (!this.sessionId) {
      throw new Error('No active session. Please start a new session.');
    }

    return this.request<VariantsResponse>(`/get-variants?session_id=${this.sessionId}&index=${watchIndex}`);
  }

  async getSeries(watchIndex: number): Promise<{
    status: 'success' | 'error';
    session_id: string;
    watch_index: number;
    series_count: number;
    series_watches: Watch[];
  }> {
    if (!this.sessionId) {
      throw new Error('No active session. Please start a new session.');
    }

    return this.request(`/get-series?session_id=${this.sessionId}&index=${watchIndex}`);
  }

  async addFeedback(
    watchIndex: number,
    action: 'like' | 'dislike',
    confidence: number = 1.0
  ): Promise<ApiResponse> {
    if (!this.sessionId) {
      throw new Error('No active session. Please start a new session.');
    }

    return this.request<ApiResponse>('/add-feedback', {
      method: 'POST',
      body: JSON.stringify({
        session_id: this.sessionId,
        watch_index: watchIndex,
        feedback_type: action,
        confidence,
      }),
    });
  }

  async resetSession(): Promise<ApiResponse> {
    if (!this.sessionId) {
      throw new Error('No active session to reset.');
    }

    return this.request<ApiResponse>('/reset-session', {
      method: 'POST',
      body: JSON.stringify({
        session_id: this.sessionId,
      }),
    });
  }

  async getSessionInfo(): Promise<{
    status: string;
    active_sessions: number;
    sessions: Array<{
      session_id: string;
      created_at: number;
      last_activity: number;
      seen_watches: number;
      feedback_count: number;
      likes: number;
      dislikes: number;
    }>;
  }> {
    return this.request('/session-info');
  }

  async getStats(): Promise<{
    status: 'success' | 'error';
    total_watches: number;
    brands: Record<string, number>;
    price_ranges: Record<string, number>;
    embedding_dimension: number;
    active_sessions: number;
  }> {
    return this.request(`/stats`);
  }
}

export const apiService = new ApiService(); 