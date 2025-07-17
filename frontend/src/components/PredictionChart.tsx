import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface Prediction {
  id: number;
  technology_id: number;
  prediction_date: string;
  target_date: string;
  adoption_probability: number;
  market_impact_score: number;
  risk_score: number;
  confidence_interval: number;
  model_used: string;
  features_used: string[];
  prediction_reasoning: string;
}

interface PredictionChartProps {
  predictions: Prediction[];
}

const PredictionChart: React.FC<PredictionChartProps> = ({ predictions }) => {
  if (predictions.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-gray-400 text-6xl mb-4">📊</div>
        <p className="text-gray-500">No prediction data available</p>
        <p className="text-sm text-gray-400">Predictions will appear here once generated</p>
      </div>
    );
  }

  // Prepare data for chart
  const sortedPredictions = [...predictions].sort((a, b) => 
    new Date(a.target_date).getTime() - new Date(b.target_date).getTime()
  );

  const labels = sortedPredictions.map(pred => 
    new Date(pred.target_date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  );

  const adoptionData = sortedPredictions.map(pred => pred.adoption_probability * 100);
  const impactData = sortedPredictions.map(pred => pred.market_impact_score * 100);
  const riskData = sortedPredictions.map(pred => pred.risk_score * 100);

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Adoption Probability (%)',
        data: adoptionData,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: true,
      },
      {
        label: 'Market Impact (%)',
        data: impactData,
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.4,
        fill: false,
      },
      {
        label: 'Risk Score (%)',
        data: riskData,
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.4,
        fill: false,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          usePointStyle: true,
          padding: 20,
        },
      },
      title: {
        display: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          callback: function(value: any) {
            return value + '%';
          },
        },
      },
    },
    interaction: {
      intersect: false,
      mode: 'index' as const,
    },
  };

  // Calculate summary stats
  const avgAdoption = predictions.reduce((sum, pred) => sum + pred.adoption_probability, 0) / predictions.length;
  const avgImpact = predictions.reduce((sum, pred) => sum + pred.market_impact_score, 0) / predictions.length;
  const avgRisk = predictions.reduce((sum, pred) => sum + pred.risk_score, 0) / predictions.length;

  return (
    <div>
      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="text-center">
          <p className="text-2xl font-bold text-blue-600">{(avgAdoption * 100).toFixed(1)}%</p>
          <p className="text-xs text-gray-500">Avg Adoption</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-green-600">{(avgImpact * 100).toFixed(1)}%</p>
          <p className="text-xs text-gray-500">Avg Impact</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-red-600">{(avgRisk * 100).toFixed(1)}%</p>
          <p className="text-xs text-gray-500">Avg Risk</p>
        </div>
      </div>

      {/* Chart */}
      <div className="h-64">
        <Line data={chartData} options={options} />
      </div>

      {/* Recent Predictions */}
      <div className="mt-6">
        <h4 className="text-sm font-medium text-gray-900 mb-3">Recent Predictions</h4>
        <div className="space-y-2">
          {predictions.slice(0, 3).map((prediction) => (
            <div key={prediction.id} className="flex items-center justify-between text-sm">
              <div>
                <span className="font-medium">Tech #{prediction.technology_id}</span>
                <span className="text-gray-500 ml-2">
                  {new Date(prediction.target_date).toLocaleDateString()}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  prediction.adoption_probability >= 0.7 ? 'bg-green-100 text-green-800' :
                  prediction.adoption_probability >= 0.4 ? 'bg-yellow-100 text-yellow-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {(prediction.adoption_probability * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default PredictionChart; 