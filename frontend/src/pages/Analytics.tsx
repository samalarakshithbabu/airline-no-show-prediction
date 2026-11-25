import { useEffect, useState } from 'react';
import { BarChartComponent, PieChartComponent, TrendChart } from '../components/Charts';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';

interface TrendData extends Record<string, unknown> { period: string; total_bookings: number; noshow_count: number; noshow_rate: number; }
interface RouteData extends Record<string, unknown> { route: string; total_bookings: number; noshow_rate: number; avg_ticket_price: number; }
interface ChannelData { channel: string; total_bookings: number; noshow_rate: number; avg_ticket_price: number; }
interface HighRiskBooking {
  booking_code: string;
  passenger_name: string;
  route: string;
  noshow_probability: number;
  risk_category: string;
}

export default function Analytics() {
  const [trends, setTrends] = useState<TrendData[]>([]);
  const [routes, setRoutes] = useState<RouteData[]>([]);
  const [channels, setChannels] = useState<ChannelData[]>([]);
  const [highRiskBookings, setHighRiskBookings] = useState<HighRiskBooking[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('trends');
  const [error, setError] = useState('');
  const [highRiskAccess, setHighRiskAccess] = useState<'loading' | 'allowed' | 'forbidden' | 'unavailable'>('loading');
  const { user } = useAuth();

  useEffect(() => {
    const fetchData = async () => {
      const results = await Promise.allSettled([
        api.getTrends('day', 30),
        api.getRouteAnalytics(),
        api.getChannelAnalytics(),
      ]);
      const [trendsResult, routesResult, channelsResult] = results;
      if (trendsResult.status === 'fulfilled') setTrends(trendsResult.value.data.trends || []);
      if (routesResult.status === 'fulfilled') setRoutes(routesResult.value.data.routes || []);
      if (channelsResult.status === 'fulfilled') setChannels(channelsResult.value.data.channels || []);
      if (results.some((result) => result.status === 'rejected')) {
        setError('Some general analytics could not be loaded. Available analytics remain accessible.');
      }

      if (user?.role !== 'admin') {
        setHighRiskAccess('forbidden');
      } else {
        try {
          const highRiskRes = await api.getHighRiskBookings();
          setHighRiskBookings(highRiskRes.data.bookings || []);
          setHighRiskAccess('allowed');
        } catch (requestError: unknown) {
          const status = (requestError as { response?: { status?: number } }).response?.status;
          setHighRiskAccess(status === 403 ? 'forbidden' : 'unavailable');
        }
      }
      setLoading(false);
    };
    fetchData();
  }, [user?.role]);

  const pieData = channels.map(c => ({ name: c.channel, value: c.total_bookings }));

  const channelNoshowData = channels.map(c => ({
    name: c.channel,
    noshow_rate: c.noshow_rate,
    bookings: c.total_bookings / 100,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
        <p className="text-gray-500">Comprehensive no-show analytics and trends</p>
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-200">
        {['trends', 'routes', 'channels', 'high-risk'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors capitalize ${
              activeTab === tab
                ? 'border-primary-600 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab.replace('-', ' ')}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="spinner" />
        </div>
      ) : (
        <>
          {activeTab === 'trends' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <TrendChart
                data={trends}
                xKey="period"
                yKey="total_bookings"
                title="Bookings Over Time"
              />
              <BarChartComponent
                data={trends}
                xKey="period"
                yKey="noshow_count"
                title="No-Shows by Week"
                color="#ef4444"
              />
            </div>
          )}

          {activeTab === 'routes' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="card">
                <h3 className="card-header">Route Performance</h3>
                <div className="overflow-x-auto">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Route</th>
                        <th>Bookings</th>
                        <th>No-Show Rate</th>
                        <th>Avg Price</th>
                      </tr>
                    </thead>
                    <tbody>
                      {routes.map((r) => (
                        <tr key={r.route}>
                          <td className="font-medium">{r.route}</td>
                          <td>{r.total_bookings.toLocaleString()}</td>
                          <td>
                            <span className={`badge ${r.noshow_rate > 13 ? 'badge-danger' : r.noshow_rate > 11 ? 'badge-warning' : 'badge-success'}`}>
                              {r.noshow_rate.toFixed(1)}%
                            </span>
                          </td>
                          <td>${r.avg_ticket_price}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <BarChartComponent
                data={routes}
                xKey="route"
                yKey="noshow_rate"
                title="No-Show Rate by Route"
                color="#f59e0b"
              />
            </div>
          )}

          {activeTab === 'channels' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <PieChartComponent data={pieData} title="Booking Distribution by Channel" />
              <BarChartComponent
                data={channelNoshowData}
                xKey="name"
                yKey="noshow_rate"
                title="No-Show Rate by Channel"
                color="#8b5cf6"
              />
            </div>
          )}

          {activeTab === 'high-risk' && (
            <div className="card">
              <h3 className="card-header">High-Risk Bookings</h3>
              {highRiskAccess === 'forbidden' ? (
                <p role="status" className="text-sm text-gray-600">
                  Administrator access is required. High-risk booking and passenger details are restricted.
                </p>
              ) : highRiskAccess === 'unavailable' ? (
                <p role="alert" className="text-sm text-red-700">
                  Restricted high-risk analytics could not be loaded.
                </p>
              ) : (
              <div className="overflow-x-auto">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Booking</th>
                      <th>Passenger</th>
                      <th>Route</th>
                      <th>Probability</th>
                      <th>Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {highRiskBookings.map((booking) => (
                      <tr key={booking.booking_code}>
                        <td className="font-medium">{booking.booking_code}</td>
                        <td>{booking.passenger_name}</td>
                        <td>{booking.route}</td>
                        <td>{(booking.noshow_probability * 100).toFixed(0)}%</td>
                        <td>
                          <span className={`badge ${
                            booking.risk_category === 'CRITICAL' ? 'badge-danger' :
                            booking.risk_category === 'HIGH' ? 'badge-warning' : 'badge-info'
                          }`}>
                            {booking.risk_category}
                          </span>
                        </td>
                      </tr>
                    ))}
                    {highRiskBookings.length === 0 && (
                      <tr>
                        <td colSpan={5} className="py-8 text-center text-gray-500">No high-risk bookings in the available data.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}