import { useEffect, useState } from 'react';
import { BarChartComponent, ComparisonChart } from '../components/Charts';
import api from '../services/api';

interface ModelPerf {
  model_name: string;
  model_version: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  pr_auc: number;
}

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface ApprovedModel {
  model_name: string;
  version: string;
}

export default function Models() {
  const [models, setModels] = useState<ModelPerf[]>([]);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [approvedModel, setApprovedModel] = useState<ApprovedModel | null>(null);
  const [predictionEnabled, setPredictionEnabled] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const statusResponse = await api.getModels();
        setApprovedModel(statusResponse.data.approved_model || null);
        setPredictionEnabled(statusResponse.data.prediction_enabled === true);
      } catch {
        setPredictionEnabled(false);
        setError('Could not verify production model approval.');
      }
      try {
        const response = await api.getModelPerformance();
        setModels(response.data.models || []);
      } catch {
        setError('Could not load evaluation metrics from the API.');
      }
      try {
        const importanceResponse = await api.getGlobalFeatureImportance();
        setFeatureImportance(importanceResponse.data.features || []);
      } catch {
        setFeatureImportance([]);
      } finally {
        setLoading(false);
      }
    };
    fetchModels();
  }, []);

  const getBestModel = () => {
    return models.reduce((best, m) => (m.f1_score > best.f1_score ? m : best), models[0] || null);
  };

  const chartData = models.map(m => ({
    name: m.model_name,
    accuracy: m.accuracy * 100,
    precision: m.precision * 100,
    recall: m.recall * 100,
    f1: m.f1_score * 100,
    roc: m.roc_auc * 100,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Evaluation</h1>
        <p className="text-gray-500">Compare ML model performance and feature importance</p>
      </div>

      <div role="status" className={`rounded-lg border px-4 py-3 text-sm ${
        predictionEnabled
          ? 'border-green-200 bg-green-50 text-green-800'
          : 'border-amber-200 bg-amber-50 text-amber-900'
      }`}>
        {predictionEnabled && approvedModel
          ? `Production model approved: ${approvedModel.model_name} (${approvedModel.version}).`
          : 'No production model is approved. Candidate metrics below are research-only and cannot serve predictions.'}
      </div>

      {error && (
        <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="spinner" />
        </div>
      ) : (
        <>
          {/* Best Model Highlight */}
          {getBestModel() && (
            <div className="card bg-gradient-to-r from-amber-50 to-amber-100 border-amber-200">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-amber-700 font-medium">Top historical candidate — not approved</p>
                  <p className="text-3xl font-bold text-amber-900">{getBestModel()?.model_name}</p>
                  <p className="text-amber-800">Version {getBestModel()?.model_version}</p>
                </div>
                <div className="grid grid-cols-3 gap-6 text-center">
                  <div>
                    <p className="text-2xl font-bold text-amber-900">{(getBestModel()?.accuracy * 100).toFixed(1)}%</p>
                    <p className="text-xs text-amber-800">Accuracy</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-amber-900">{(getBestModel()?.f1_score * 100).toFixed(1)}%</p>
                    <p className="text-xs text-amber-800">F1 Score</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-amber-900">{(getBestModel()?.roc_auc * 100).toFixed(1)}%</p>
                    <p className="text-xs text-amber-800">ROC-AUC</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {models.length ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ComparisonChart
                data={chartData}
                bars={[
                  { key: 'accuracy', name: 'Accuracy', color: '#3b82f6' },
                  { key: 'precision', name: 'Precision', color: '#22c55e' },
                  { key: 'recall', name: 'Recall', color: '#f59e0b' },
                  { key: 'f1', name: 'F1', color: '#8b5cf6' },
                ]}
                title="Model Performance Comparison"
              />
              <BarChartComponent
                data={models.map(m => ({ name: m.model_name, roc: m.roc_auc * 100 }))}
                xKey="name"
                yKey="roc"
                title="ROC-AUC by Model"
                color="#10b981"
              />
            </div>
          ) : (
            <div className="card text-center text-gray-500">No verified model evaluation metrics are available.</div>
          )}

          {/* Detailed Metrics Table */}
          <div className="card">
            <h3 className="card-header">Detailed Model Metrics</h3>
            <div className="overflow-x-auto">
              <table className="table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Version</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>ROC-AUC</th>
                    <th>PR-AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => (
                    <tr key={`${m.model_name}-${m.model_version}`}>
                      <td className="font-medium">{m.model_name}</td>
                      <td>{m.model_version}</td>
                      <td>{(m.accuracy * 100).toFixed(1)}%</td>
                      <td>{(m.precision * 100).toFixed(1)}%</td>
                      <td>{(m.recall * 100).toFixed(1)}%</td>
                      <td className="font-medium">{(m.f1_score * 100).toFixed(1)}%</td>
                      <td>{(m.roc_auc * 100).toFixed(1)}%</td>
                      <td>{(m.pr_auc * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Feature Importance */}
          <div className="card">
            <h3 className="card-header">
              Feature Importance {approvedModel ? `(${approvedModel.model_name})` : '(approved production model)'}
            </h3>
            {featureImportance.length ? <div className="space-y-3">
              {featureImportance.slice(0, 10).map((f) => (
                <div key={f.feature}>
                  <div className="flex justify-between text-sm mb-1">
                    <span>{f.feature.replace(/_/g, ' ')}</span>
                    <span className="font-medium">{(f.importance * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-primary-500 to-primary-600 rounded-full"
                      style={{ width: `${f.importance * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div> : <p className="text-sm text-gray-500">Feature importance is unavailable for the loaded model.</p>}
          </div>
        </>
      )}
    </div>
  );
}