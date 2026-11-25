import axios, { AxiosInstance, AxiosResponse } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

class ApiService {
  private client: AxiosInstance;
  private unauthorizedHandler: (() => void) | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      withCredentials: true,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.client.interceptors.request.use((config) => {
      if (typeof document !== 'undefined' && config.method && !['get', 'head', 'options'].includes(config.method.toLowerCase())) {
        const csrfCookie = document.cookie
          .split('; ')
          .find((cookie) => cookie.startsWith('csrf_access_token='));
        if (csrfCookie) {
          config.headers.set('X-CSRF-TOKEN', decodeURIComponent(csrfCookie.slice('csrf_access_token='.length)));
        }
      }
      return config;
    });

    // Handle auth errors
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          this.unauthorizedHandler?.();
        }
        return Promise.reject(error);
      }
    );
  }

  setUnauthorizedHandler(handler: (() => void) | null) {
    this.unauthorizedHandler = handler;
  }

  // Auth
  async login(username: string, password: string): Promise<AxiosResponse> {
    return this.client.post('/auth/login', { username, password });
  }

  async getProfile(): Promise<AxiosResponse> {
    return this.client.get('/auth/profile');
  }

  async logout(): Promise<AxiosResponse> {
    return this.client.post('/auth/logout');
  }

  // Predictions
  async predict(bookingData: Record<string, unknown>, model?: string): Promise<AxiosResponse> {
    return this.client.post('/predict', { booking_data: bookingData, model });
  }

  async bulkPredict(bookings: Record<string, unknown>[], model?: string): Promise<AxiosResponse> {
    return this.client.post('/bulk-predict', { bookings, model });
  }

  async getModels(): Promise<AxiosResponse> {
    return this.client.get('/models');
  }

  // Analytics
  async getAnalytics(days?: number): Promise<AxiosResponse> {
    return this.client.get('/analytics/overview', { params: { days } });
  }

  async getRouteAnalytics(route?: string): Promise<AxiosResponse> {
    return this.client.get('/analytics/routes', { params: { route } });
  }

  async getChannelAnalytics(): Promise<AxiosResponse> {
    return this.client.get('/analytics/channels');
  }

  async getTrends(groupBy?: string, days?: number): Promise<AxiosResponse> {
    return this.client.get('/analytics/trends', { params: { group_by: groupBy, days } });
  }

  async getHighRiskBookings(minProbability?: number, limit?: number): Promise<AxiosResponse> {
    return this.client.get('/analytics/high-risk', { 
      params: { min_probability: minProbability, limit } 
    });
  }

  async getPassengerAnalytics(passengerId?: string, limit?: number): Promise<AxiosResponse> {
    return this.client.get('/analytics/passengers', { 
      params: { passenger_id: passengerId, limit } 
    });
  }

  async getModelPerformance(): Promise<AxiosResponse> {
    return this.client.get('/models/performance');
  }

  async getGlobalFeatureImportance(model?: string): Promise<AxiosResponse> {
    return this.client.get('/explainability/global', { params: { model } });
  }

  // Overbooking
  async simulateOverbooking(params: {
    capacity: number;
    bookings: number;
    avg_noshow_prob: number;
    ticket_price: number;
    max_overbooking?: number;
    cost_params?: Record<string, number>;
  }): Promise<AxiosResponse> {
    return this.client.post('/overbooking/simulate', params);
  }

  // Health
  async healthCheck(): Promise<AxiosResponse> {
    return this.client.get('/health');
  }
}

export const api = new ApiService();
export default api;