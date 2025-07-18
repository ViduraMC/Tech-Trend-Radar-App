import React, { useState } from 'react';
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

interface Technology {
  id: number;
  name: string;
  category: string;
  description: string;
  keywords: string[];
  first_detected: string;
  last_updated: string;
}

interface TimeSeriesPrediction {
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

interface TimeSeriesPredictionsProps {
  predictions: TimeSeriesPrediction[];
  technologies: Technology[];
}

const TimeSeriesPredictions: React.FC<TimeSeriesPredictionsProps> = ({ predictions, technologies }) => {
  const [selectedTechnology, setSelectedTechnology] = useState<Technology | null>(null);
  const [showModal, setShowModal] = useState(false);

  if (predictions.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-gray-400 text-6xl mb-4">📈</div>
        <p className="text-gray-500">No time series prediction data available</p>
        <p className="text-sm text-gray-400">Time-based predictions will appear here once generated</p>
      </div>
    );
  }

  // Get unique technologies that have time series data
  const uniqueTechnologies = Array.from(new Set(predictions.map(pred => pred.technology_id)));
  
  // Get the first technology by default, or let user select
  const currentTech = selectedTechnology || technologies.find(tech => tech.id === uniqueTechnologies[0]);
  
  if (!currentTech) {
    return <div>No technology data available</div>;
  }

  // Get time series data for the selected technology
  const techPredictions = predictions
    .filter(pred => pred.technology_id === currentTech.id)
    .sort((a, b) => new Date(a.target_date).getTime() - new Date(b.target_date).getTime());

  // Prepare chart data
  const labels = techPredictions.map(pred => 
    new Date(pred.target_date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
  );

  const adoptionData = techPredictions.map(pred => pred.adoption_probability * 100);
  const impactData = techPredictions.map(pred => pred.market_impact_score * 100);
  const riskData = techPredictions.map(pred => pred.risk_score * 100);

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Adoption Probability (%)',
        data: adoptionData,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        tension: 0.4,
        fill: false,
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
        display: true,
        text: `${currentTech.name} - Future Predictions`,
        font: {
          size: 16,
          weight: 'bold' as const
        }
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Time Period'
        }
      },
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'Percentage (%)'
        },
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

  // Calculate trend analysis
  const calculateTrend = (data: number[]) => {
    if (data.length < 2) return 'stable';
    const first = data[0];
    const last = data[data.length - 1];
    const change = last - first;
    if (change > 5) return 'rising';
    if (change < -5) return 'falling';
    return 'stable';
  };

  const adoptionTrend = calculateTrend(adoptionData);
  const impactTrend = calculateTrend(impactData);
  const riskTrend = calculateTrend(riskData);

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'rising': return '📈';
      case 'falling': return '📉';
      default: return '➡️';
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'rising': return '#16a34a';
      case 'falling': return '#dc2626';
      default: return '#6b7280';
    }
  };

  return (
    <div>
      {/* Technology Selector */}
      <div style={{ marginBottom: '1.5rem' }}>
        <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: '500', color: '#111827' }}>
          Select Technology for Time Series Analysis:
        </label>
        <select
          value={currentTech.id}
          onChange={(e) => {
            const tech = technologies.find(t => t.id === parseInt(e.target.value));
            setSelectedTechnology(tech || null);
          }}
          style={{
            padding: '0.5rem',
            borderRadius: '0.375rem',
            border: '1px solid #d1d5db',
            fontSize: '0.875rem',
            minWidth: '200px'
          }}
        >
          {uniqueTechnologies.map(techId => {
            const tech = technologies.find(t => t.id === techId);
            return tech ? (
              <option key={techId} value={techId}>
                {tech.name} ({tech.category})
              </option>
            ) : null;
          })}
        </select>
      </div>

      {/* Trend Summary */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(3, 1fr)', 
        gap: '1rem', 
        marginBottom: '1.5rem',
        padding: '1rem',
        background: '#f8fafc',
        borderRadius: '0.5rem',
        border: '1px solid #e2e8f0'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '1.5rem', marginBottom: '0.25rem' }}>
            {getTrendIcon(adoptionTrend)}
          </div>
          <p style={{ fontSize: '0.875rem', fontWeight: '500', color: getTrendColor(adoptionTrend), margin: '0' }}>
            Adoption Trend: {adoptionTrend.toUpperCase()}
          </p>
          <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: '0' }}>
            {adoptionData.length > 0 ? `${adoptionData[0].toFixed(1)}% → ${adoptionData[adoptionData.length - 1].toFixed(1)}%` : 'No data'}
          </p>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '1.5rem', marginBottom: '0.25rem' }}>
            {getTrendIcon(impactTrend)}
          </div>
          <p style={{ fontSize: '0.875rem', fontWeight: '500', color: getTrendColor(impactTrend), margin: '0' }}>
            Impact Trend: {impactTrend.toUpperCase()}
          </p>
          <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: '0' }}>
            {impactData.length > 0 ? `${impactData[0].toFixed(1)}% → ${impactData[impactData.length - 1].toFixed(1)}%` : 'No data'}
          </p>
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '1.5rem', marginBottom: '0.25rem' }}>
            {getTrendIcon(riskTrend)}
          </div>
          <p style={{ fontSize: '0.875rem', fontWeight: '500', color: getTrendColor(riskTrend), margin: '0' }}>
            Risk Trend: {riskTrend.toUpperCase()}
          </p>
          <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: '0' }}>
            {riskData.length > 0 ? `${riskData[0].toFixed(1)}% → ${riskData[riskData.length - 1].toFixed(1)}%` : 'No data'}
          </p>
        </div>
      </div>

      {/* Time Series Chart */}
      <div style={{ height: '20rem', marginBottom: '1.5rem' }}>
        <Line data={chartData} options={options} />
      </div>

      {/* Prediction Details */}
      <div style={{ marginTop: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <h4 style={{ fontSize: '0.875rem', fontWeight: '500', color: '#111827', margin: '0' }}>
            Monthly Predictions for {currentTech.name}
          </h4>
          <button
            onClick={() => setShowModal(true)}
            style={{
              background: '#667eea',
              color: 'white',
              border: 'none',
              padding: '0.5rem 1rem',
              borderRadius: '0.375rem',
              fontSize: '0.75rem',
              fontWeight: '500',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => e.currentTarget.style.background = '#5a67d8'}
            onMouseOut={(e) => e.currentTarget.style.background = '#667eea'}
          >
            View Details
          </button>
        </div>
        
        <div style={{ 
          background: 'white', 
          borderRadius: '0.5rem', 
          border: '1px solid #e5e7eb',
          overflow: 'hidden'
        }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: '#f9fafb' }}>
                <th style={{ padding: '0.75rem', textAlign: 'left', fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                  Month
                </th>
                <th style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                  Adoption
                </th>
                <th style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                  Impact
                </th>
                <th style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                  Risk
                </th>
                <th style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                  Confidence
                </th>
              </tr>
            </thead>
            <tbody>
              {techPredictions.slice(0, 6).map((prediction, index) => (
                <tr key={prediction.id} style={{ borderBottom: index < techPredictions.length - 1 ? '1px solid #f3f4f6' : 'none' }}>
                  <td style={{ padding: '0.75rem', fontSize: '0.875rem', color: '#111827' }}>
                    {new Date(prediction.target_date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })}
                  </td>
                  <td style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.875rem', fontWeight: '500', color: '#2563eb' }}>
                    {(prediction.adoption_probability * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.875rem', fontWeight: '500', color: '#16a34a' }}>
                    {(prediction.market_impact_score * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.875rem', fontWeight: '500', color: '#dc2626' }}>
                    {(prediction.risk_score * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.875rem', color: '#6b7280' }}>
                    {(prediction.confidence_interval * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Modal for detailed view */}
      {showModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            background: 'white',
            borderRadius: '1rem',
            padding: '2rem',
            maxWidth: '90vw',
            maxHeight: '90vh',
            overflow: 'auto',
            position: 'relative'
          }}>
            <button
              onClick={() => setShowModal(false)}
              style={{
                position: 'absolute',
                top: '1rem',
                right: '1rem',
                background: 'none',
                border: 'none',
                fontSize: '1.5rem',
                cursor: 'pointer',
                color: '#6b7280'
              }}
            >
              ×
            </button>
            
            <h3 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '1rem', color: '#111827' }}>
              Time Series Analysis: {currentTech.name}
            </h3>
            
            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ fontSize: '1rem', fontWeight: '500', marginBottom: '0.5rem', color: '#374151' }}>
                Technology Information
              </h4>
              <p style={{ fontSize: '0.875rem', color: '#6b7280', margin: '0 0 0.5rem 0' }}>
                <strong>Category:</strong> {currentTech.category}
              </p>
              <p style={{ fontSize: '0.875rem', color: '#6b7280', margin: '0 0 0.5rem 0' }}>
                <strong>Description:</strong> {currentTech.description}
              </p>
              <p style={{ fontSize: '0.875rem', color: '#6b7280', margin: '0' }}>
                <strong>Keywords:</strong> {currentTech.keywords.join(', ')}
              </p>
            </div>

            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ fontSize: '1rem', fontWeight: '500', marginBottom: '0.5rem', color: '#374151' }}>
                Trend Analysis Summary
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
                <div style={{ padding: '1rem', background: '#f0f9ff', borderRadius: '0.5rem', border: '1px solid #bae6fd' }}>
                  <p style={{ fontSize: '0.875rem', fontWeight: '500', color: '#0369a1', margin: '0 0 0.25rem 0' }}>
                    Adoption Trend: {adoptionTrend.toUpperCase()}
                  </p>
                  <p style={{ fontSize: '0.75rem', color: '#0c4a6e', margin: '0' }}>
                    {adoptionData.length > 0 ? 
                      `From ${adoptionData[0].toFixed(1)}% to ${adoptionData[adoptionData.length - 1].toFixed(1)}%` : 
                      'Insufficient data'
                    }
                  </p>
                </div>
                <div style={{ padding: '1rem', background: '#f0fdf4', borderRadius: '0.5rem', border: '1px solid #bbf7d0' }}>
                  <p style={{ fontSize: '0.875rem', fontWeight: '500', color: '#166534', margin: '0 0 0.25rem 0' }}>
                    Impact Trend: {impactTrend.toUpperCase()}
                  </p>
                  <p style={{ fontSize: '0.75rem', color: '#14532d', margin: '0' }}>
                    {impactData.length > 0 ? 
                      `From ${impactData[0].toFixed(1)}% to ${impactData[impactData.length - 1].toFixed(1)}%` : 
                      'Insufficient data'
                    }
                  </p>
                </div>
                <div style={{ padding: '1rem', background: '#fef2f2', borderRadius: '0.5rem', border: '1px solid #fecaca' }}>
                  <p style={{ fontSize: '0.875rem', fontWeight: '500', color: '#991b1b', margin: '0 0 0.25rem 0' }}>
                    Risk Trend: {riskTrend.toUpperCase()}
                  </p>
                  <p style={{ fontSize: '0.75rem', color: '#7f1d1d', margin: '0' }}>
                    {riskData.length > 0 ? 
                      `From ${riskData[0].toFixed(1)}% to ${riskData[riskData.length - 1].toFixed(1)}%` : 
                      'Insufficient data'
                    }
                  </p>
                </div>
              </div>
            </div>

            <div>
              <h4 style={{ fontSize: '1rem', fontWeight: '500', marginBottom: '0.5rem', color: '#374151' }}>
                Complete Prediction Timeline
              </h4>
              <div style={{ 
                background: 'white', 
                borderRadius: '0.5rem', 
                border: '1px solid #e5e7eb',
                overflow: 'hidden'
              }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ background: '#f9fafb' }}>
                      <th style={{ padding: '0.75rem', textAlign: 'left', fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                        Month
                      </th>
                      <th style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                        Adoption
                      </th>
                      <th style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                        Impact
                      </th>
                      <th style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                        Risk
                      </th>
                      <th style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                        Confidence
                      </th>
                      <th style={{ padding: '0.75rem', textAlign: 'left', fontSize: '0.75rem', fontWeight: '500', color: '#6b7280', borderBottom: '1px solid #e5e7eb' }}>
                        Model Used
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {techPredictions.map((prediction, index) => (
                      <tr key={prediction.id} style={{ borderBottom: index < techPredictions.length - 1 ? '1px solid #f3f4f6' : 'none' }}>
                        <td style={{ padding: '0.75rem', fontSize: '0.875rem', color: '#111827' }}>
                          {new Date(prediction.target_date).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })}
                        </td>
                        <td style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.875rem', fontWeight: '500', color: '#2563eb' }}>
                          {(prediction.adoption_probability * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.875rem', fontWeight: '500', color: '#16a34a' }}>
                          {(prediction.market_impact_score * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.875rem', fontWeight: '500', color: '#dc2626' }}>
                          {(prediction.risk_score * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '0.75rem', textAlign: 'center', fontSize: '0.875rem', color: '#6b7280' }}>
                          {(prediction.confidence_interval * 100).toFixed(0)}%
                        </td>
                        <td style={{ padding: '0.75rem', fontSize: '0.75rem', color: '#6b7280' }}>
                          {prediction.model_used}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TimeSeriesPredictions; 