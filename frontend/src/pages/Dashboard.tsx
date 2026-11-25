import { useEffect, useState } from 'react';
import { Users, Plane, DollarSign, TrendingDown, AlertTriangle, CheckCircle, Sparkles, ArrowRight, Activity } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import { TrendChart, PieChartComponent } from '../components/Charts';
import api from '../services/api';

interface DashboardData {
  period_days: number;
  total_bookings: number;
  total_flights: number;
  predicted_noshows: number;
  actual_noshows: number;
  noshow_rate: number;
  avg_noshow_probability: number;
  total_revenue: number;
  unique_passengers: number;
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [trends, setTrends] = useState<Array<Record<string, unknown>>>([]);
  const [routes, setRoutes] = useState<Array<Record<string, unknown>>>([]);
  const [channels, setChannels] = useState<Array<Record<string, unknown>>>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [dashboardRes, trendsRes, routesRes, channelsRes] = await Promise.all([
          api.getAnalytics(30),
          api.getTrends('day', 30),
          api.getRouteAnalytics(),
          api.getChannelAnalytics(),
        ]);
        setData(dashboardRes.data);
        setTrends(trendsRes.data.trends || []);
        setRoutes(routesRes.data.routes || []);
        setChannels(channelsRes.data.channels || []);
      } catch {
        setError('Could not load dashboard data. Check the API and database connection.');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-blue-50">
        <div className="text-center">
          <div className="relative">
            <div className="w-20 h-20 border-4 border-blue-200 rounded-full animate-spin border-t-blue-600 mx-auto"></div>
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              <Activity className="w-8 h-8 text-blue-600 animate-pulse" />
            </div>
          </div>
          <p className="mt-4 text-gray-600 font-medium">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  const channelData = channels.map((channel) => ({
    name: String(channel.channel),
    value: Number(channel.total_bookings),
  }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">
      <div className="max-w-7xl mx-auto space-y-8">
        
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <div className="p-2 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Executive Dashboard
              </h1>
            </div>
            <p className="text-gray-500 ml-14">Real-time insights into no-show predictions and analytics</p>
          </div>
          <div className="flex items-center gap-3 ml-14 md:ml-0">
            <span className="inline-flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
              Historical data
            </span>
          </div>
        </div>

        {error && (
          <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <MetricCard
            title="Total Bookings"
            value={data?.total_bookings.toLocaleString() || '0'}
            subtitle={`Last ${data?.period_days || 30} days`}
            icon={<Users className="h-6 w-6" />}
            color="blue"
          />
          <MetricCard
            title="No-Show Rate"
            value={`${data?.noshow_rate.toFixed(1)}%`}
            subtitle={`Average predicted probability: ${(data?.avg_noshow_probability || 0).toFixed(1)}%`}
            icon={<TrendingDown className="h-6 w-6" />}
            color="green"
          />
          <MetricCard
            title="Total Revenue"
            value={`$${((data?.total_revenue || 0) / 1000000).toFixed(2)}M`}
            subtitle="Recorded ticket revenue"
            icon={<DollarSign className="h-6 w-6" />}
            color="purple"
          />
          <MetricCard
            title="Active Flights"
            value={data?.total_flights.toLocaleString() || '0'}
            subtitle="This month"
            icon={<Plane className="h-6 w-6" />}
            color="yellow"
          />
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 bg-white rounded-2xl shadow-xl shadow-blue-100/50 border border-blue-100 overflow-hidden">
            <div className="p-6 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold text-gray-900">Booking Trends</h3>
                  <p className="text-sm text-gray-500">Monthly booking volumes over time</p>
                </div>
                <div className="p-2 bg-blue-50 rounded-lg">
                  <TrendingDown className="w-5 h-5 text-blue-600" />
                </div>
              </div>
            </div>
            <div className="p-6">
              <TrendChart
                data={trends}
                xKey="period"
                yKey="total_bookings"
                title="Booking Trends"
              />
            </div>
          </div>
          <div className="bg-white rounded-2xl shadow-xl shadow-blue-100/50 border border-blue-100 overflow-hidden">
            <div className="p-6 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-bold text-gray-900">Booking Channels</h3>
                  <p className="text-sm text-gray-500">Distribution by channel</p>
                </div>
                <div className="p-2 bg-purple-50 rounded-lg">
                  <Sparkles className="w-5 h-5 text-purple-600" />
                </div>
              </div>
            </div>
            <div className="p-6">
              <PieChartComponent
                data={channelData}
                title="Booking Channels"
              />
            </div>
          </div>
        </div>

        {/* Routes Table */}
        <div className="bg-white rounded-2xl shadow-xl shadow-blue-100/50 border border-blue-100 overflow-hidden">
          <div className="p-6 border-b border-gray-100">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-gray-900">Route Performance</h3>
                <p className="text-sm text-gray-500">Top routes by booking volume</p>
              </div>
              <button className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg text-sm font-medium hover:shadow-lg hover:shadow-blue-500/30 transition-all duration-200">
                View All Routes
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50/50">
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Route</th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Bookings</th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">No-Show Rate</th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Avg Ticket</th>
                  <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {routes.map((route: Record<string, unknown>) => (
                  <tr key={route.route as string} className="hover:bg-blue-50/50 transition-colors duration-150">
                    <td className="px-6 py-4">
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 text-white">
                        {(route.route as string)}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-gray-900 font-medium">{(route.total_bookings as number).toLocaleString()}</span>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold ${
                        (route.noshow_rate as number) > 13 
                          ? 'bg-red-100 text-red-700' 
                          : 'bg-green-100 text-green-700'
                      }`}>
                        {(route.noshow_rate as number).toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-gray-900 font-semibold">${(route.avg_ticket_price as number).toFixed(0)}</span>
                    </td>
                    <td className="px-6 py-4">
                      {(route.noshow_rate as number) > 13 ? (
                        <span className="inline-flex items-center gap-2 px-3 py-1 bg-red-50 text-red-700 rounded-full text-sm font-medium">
                          <AlertTriangle size={14} /> Needs Attention
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-2 px-3 py-1 bg-green-50 text-green-700 rounded-full text-sm font-medium">
                          <CheckCircle size={14} /> On Track
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <a href="/predict" className="group relative overflow-hidden bg-gradient-to-br from-blue-600 to-indigo-600 rounded-2xl p-6 text-white hover:shadow-2xl hover:shadow-blue-500/30 transition-all duration-300 hover:-translate-y-1">
            <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16 transform transition-transform group-hover:scale-150"></div>
            <div className="relative">
              <div className="p-3 bg-white/20 rounded-xl w-fit mb-4">
                <TrendingDown className="w-6 h-6" />
              </div>
              <h4 className="text-xl font-bold mb-2">Make Predictions</h4>
              <p className="text-blue-100 text-sm">Predict no-show probability for new bookings with AI-powered insights</p>
            </div>
          </a>
          <a href="/analytics" className="group relative overflow-hidden bg-gradient-to-br from-purple-600 to-pink-600 rounded-2xl p-6 text-white hover:shadow-2xl hover:shadow-purple-500/30 transition-all duration-300 hover:-translate-y-1">
            <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16 transform transition-transform group-hover:scale-150"></div>
            <div className="relative">
              <div className="p-3 bg-white/20 rounded-xl w-fit mb-4">
                <Activity className="w-6 h-6" />
              </div>
              <h4 className="text-xl font-bold mb-2">View Analytics</h4>
              <p className="text-purple-100 text-sm">Detailed analytics, trends, and comprehensive reports</p>
            </div>
          </a>
          <a href="/overbooking" className="group relative overflow-hidden bg-gradient-to-br from-emerald-600 to-teal-600 rounded-2xl p-6 text-white hover:shadow-2xl hover:shadow-emerald-500/30 transition-all duration-300 hover:-translate-y-1">
            <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full -mr-16 -mt-16 transform transition-transform group-hover:scale-150"></div>
            <div className="relative">
              <div className="p-3 bg-white/20 rounded-xl w-fit mb-4">
                <DollarSign className="w-6 h-6" />
              </div>
              <h4 className="text-xl font-bold mb-2">Overbooking Simulator</h4>
              <p className="text-emerald-100 text-sm">Optimize overbooking strategies with Monte Carlo simulation</p>
            </div>
          </a>
        </div>

        {/* Footer */}
        <div className="text-center text-sm text-gray-400 pt-4">
          <p>Airline No-Show Prediction System • Powered by Machine Learning</p>
        </div>
      </div>
    </div>
  );
}