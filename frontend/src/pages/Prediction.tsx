import React, { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle, Info } from 'lucide-react';
import api from '../services/api';

interface PredictionResult {
  prediction: number;
  prediction_label: string;
  noshow_probability: number;
  risk_category: string;
  confidence: number;
  model_used: string;
  explanation?: {
    positive_factors: Array<{ feature: string; value: number }>;
    negative_factors: Array<{ feature: string; value: number }>;
  } | null;
}

const AIRPORTS = ['BOS', 'DFW', 'JFK', 'LAX', 'MIA', 'ORD', 'SEA', 'SFO'];

function formatDateInput(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

export default function Prediction() {
  const [bookingData, setBookingData] = useState(() => {
    const bookingDate = new Date();
    const flightDate = new Date(bookingDate);
    flightDate.setDate(flightDate.getDate() + 14);
    return {
      booking_date: formatDateInput(bookingDate),
      flight_date: formatDateInput(flightDate),
      departure_airport: 'JFK',
      arrival_airport: 'LAX',
      booking_channel_encoded: 0,
      cabin_class_encoded: 0,
      ticket_price: 350,
      loyalty_tier_encoded: 1,
    };
  });
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [predictionAvailable, setPredictionAvailable] = useState<boolean | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.getModels()
      .then((response) => setPredictionAvailable(response.data.prediction_enabled === true))
      .catch(() => {
        setPredictionAvailable(null);
        setError('Could not verify production model approval. Predictions remain disabled.');
      });
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const response = await api.predict(bookingData);
      setResult(response.data);
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { error?: string; message?: string } } };
      setResult(null);
      setError(
        axiosError.response?.data?.message
        || axiosError.response?.data?.error
        || 'Prediction service is unavailable. Please try again later.'
      );
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (category: string) => {
    switch (category) {
      case 'LOW': return 'text-green-600 bg-green-50';
      case 'MODERATE': return 'text-blue-600 bg-blue-50';
      case 'ELEVATED': return 'text-yellow-600 bg-yellow-50';
      case 'HIGH': return 'text-orange-600 bg-orange-50';
      case 'CRITICAL': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Passenger Prediction</h1>
        <p className="text-gray-500">Predict no-show probability for individual passengers</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <div className="card">
          <h3 className="card-header">Booking Information</h3>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="booking-date" className="form-label">Booking Date</label>
                <input
                  id="booking-date"
                  type="date"
                  value={bookingData.booking_date}
                  onChange={(e) => setBookingData({ ...bookingData, booking_date: e.target.value })}
                  className="form-input"
                  required
                />
              </div>
              <div>
                <label htmlFor="flight-date" className="form-label">Flight Date</label>
                <input
                  id="flight-date"
                  type="date"
                  value={bookingData.flight_date}
                  onChange={(e) => setBookingData({ ...bookingData, flight_date: e.target.value })}
                  className="form-input"
                  min={bookingData.booking_date}
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label htmlFor="departure-airport" className="form-label">Departure Airport</label>
                <select
                  id="departure-airport"
                  value={bookingData.departure_airport}
                  onChange={(e) => setBookingData({ ...bookingData, departure_airport: e.target.value })}
                  className="form-input"
                  required
                >
                  {AIRPORTS.map((airport) => <option key={airport} value={airport}>{airport}</option>)}
                </select>
              </div>
              <div>
                <label htmlFor="arrival-airport" className="form-label">Arrival Airport</label>
                <select
                  id="arrival-airport"
                  value={bookingData.arrival_airport}
                  onChange={(e) => setBookingData({ ...bookingData, arrival_airport: e.target.value })}
                  className="form-input"
                  required
                >
                  {AIRPORTS.map((airport) => <option key={airport} value={airport}>{airport}</option>)}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="form-label">Ticket Price ($)</label>
                <input
                  type="number"
                  value={bookingData.ticket_price}
                  onChange={(e) => setBookingData({ ...bookingData, ticket_price: parseFloat(e.target.value) })}
                  className="form-input"
                  min="0"
                  required
                />
              </div>
              <div>
                <label className="form-label">Loyalty Tier</label>
                <select
                  value={bookingData.loyalty_tier_encoded}
                  onChange={(e) => setBookingData({ ...bookingData, loyalty_tier_encoded: parseInt(e.target.value) })}
                  className="form-input"
                >
                  <option value="0">Standard</option>
                  <option value="1">Silver</option>
                  <option value="2">Gold</option>
                  <option value="3">Platinum</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="form-label">Booking Channel</label>
                <select
                  value={bookingData.booking_channel_encoded}
                  onChange={(e) => setBookingData({ ...bookingData, booking_channel_encoded: parseInt(e.target.value) })}
                  className="form-input"
                >
                  <option value="0">Web</option>
                  <option value="1">Mobile</option>
                  <option value="2">Agent</option>
                  <option value="3">Call Center</option>
                </select>
              </div>
              <div>
                <label className="form-label">Cabin Class</label>
                <select
                  value={bookingData.cabin_class_encoded}
                  onChange={(e) => setBookingData({ ...bookingData, cabin_class_encoded: parseInt(e.target.value) })}
                  className="form-input"
                >
                  <option value="0">Economy</option>
                  <option value="1">Premium Economy</option>
                  <option value="2">Business</option>
                  <option value="3">First</option>
                </select>
              </div>
            </div>

            {predictionAvailable === false && (
              <div role="status" className="p-4 bg-amber-50 border border-amber-200 rounded-lg text-amber-800 text-sm">
                No production model is approved. Predictions remain disabled until a candidate passes the approval gate.
              </div>
            )}

            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading || predictionAvailable !== true}
              className="btn btn-primary w-full py-3"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="spinner h-5 w-5" />
                  Analyzing...
                </span>
              ) : (
                'Predict No-Show Probability'
              )}
            </button>
          </form>
        </div>

        {/* Results */}
        <div className="card">
          <h3 className="card-header">Prediction Result</h3>
          {result ? (
            <div className="space-y-6">
              {/* Main Result */}
              <div className={`p-6 rounded-xl ${result.prediction === 1 ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'}`}>
                <div className="flex items-center gap-3 mb-4">
                  {result.prediction === 1 ? (
                    <AlertTriangle className="h-8 w-8 text-red-600" />
                  ) : (
                    <CheckCircle className="h-8 w-8 text-green-600" />
                  )}
                  <div>
                    <p className="text-sm text-gray-600">Predicted Outcome</p>
                    <p className={`text-2xl font-bold ${result.prediction === 1 ? 'text-red-600' : 'text-green-600'}`}>
                      {result.prediction_label.replace('_', ' ')}
                    </p>
                  </div>
                </div>
                <div className="text-center">
                  <p className="text-sm text-gray-500">No-Show Probability</p>
                  <p className="text-5xl font-bold text-gray-900">{(result.noshow_probability * 100).toFixed(1)}%</p>
                </div>
              </div>

              {/* Risk Badge */}
              <div className={`p-4 rounded-lg text-center ${getRiskColor(result.risk_category)}`}>
                <p className="text-sm">Risk Category</p>
                <p className="text-2xl font-bold">{result.risk_category}</p>
              </div>

              {/* Confidence */}
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-500">Model Confidence</span>
                  <span className="font-medium">{(result.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-primary-600 rounded-full transition-all"
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
              </div>

              {/* Probability Bar */}
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-green-600">Will Show ({((1 - result.noshow_probability) * 100).toFixed(1)}%)</span>
                  <span className="text-red-600">No-Show ({(result.noshow_probability * 100).toFixed(1)}%)</span>
                </div>
                <div className="flex h-6 rounded-full overflow-hidden">
                  <div 
                    className="bg-green-500 transition-all"
                    style={{ width: `${(1 - result.noshow_probability) * 100}%` }}
                  />
                  <div 
                    className="bg-red-500 transition-all"
                    style={{ width: `${result.noshow_probability * 100}%` }}
                  />
                </div>
              </div>

              {/* Info */}
              <div className="flex items-start gap-2 p-4 bg-blue-50 rounded-lg text-blue-700 text-sm">
                <Info className="h-5 w-5 flex-shrink-0 mt-0.5" />
                <p>Model: {result.model_used}. Predictions are probabilistic and should be used as one input in operational decisions.</p>
              </div>

              {result.explanation && (
                <div>
                  <h4 className="mb-3 font-semibold text-gray-900">Factors influencing this prediction</h4>
                  <ul className="space-y-2 text-sm">
                    {[...result.explanation.positive_factors, ...result.explanation.negative_factors]
                      .slice(0, 6)
                      .map((factor) => (
                        <li key={factor.feature} className="flex justify-between gap-4">
                          <span>{factor.feature.replace(/_/g, ' ')}</span>
                          <span className={factor.value >= 0 ? 'text-red-600' : 'text-green-600'}>
                            {factor.value > 0 ? '+' : ''}{factor.value.toFixed(3)}
                          </span>
                        </li>
                      ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <p>Enter booking information and click "Predict" to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}