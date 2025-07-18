import React, { useState } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
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
  technologies: Technology[];
}

const PredictionChart: React.FC<PredictionChartProps> = ({ predictions, technologies }) => {
  const [showModal, setShowModal] = useState(false);
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  if (predictions.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-gray-400 text-6xl mb-4">📊</div>
        <p className="text-gray-500">No prediction data available</p>
        <p className="text-sm text-gray-400">Predictions will appear here once generated</p>
      </div>
    );
  }

  // Filter for comparison predictions only
  const comparisonPredictions = predictions.filter(pred => 
    pred.model_used === 'comparison_model' || pred.model_used === 'fallback_model'
  );
  
  if (comparisonPredictions.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-gray-400 text-6xl mb-4">📊</div>
        <p className="text-gray-500">No comparison prediction data available</p>
        <p className="text-sm text-gray-400">Comparison predictions will appear here once generated</p>
      </div>
    );
  }

  // Prepare data for chart - show by technology, not by date
  const uniqueTechnologies = Array.from(new Set(comparisonPredictions.map(pred => pred.technology_id)));
  
  // Get the latest prediction for each technology
  const latestPredictions = uniqueTechnologies.map(techId => {
    const techPredictions = comparisonPredictions.filter(pred => pred.technology_id === techId);
    return techPredictions.sort((a, b) => new Date(b.prediction_date).getTime() - new Date(a.prediction_date).getTime())[0];
  }).slice(0, 20); // Limit to 20 technologies to prevent overcrowding

  const labels = latestPredictions.map(pred => {
    const technology = technologies.find(tech => tech.id === pred.technology_id);
    return technology ? technology.name : `Tech #${pred.technology_id}`;
  });

  const adoptionData = latestPredictions.map(pred => pred.adoption_probability * 100);
  const impactData = latestPredictions.map(pred => pred.market_impact_score * 100);
  const riskData = latestPredictions.map(pred => pred.risk_score * 100);

  const chartData = {
    labels,
    datasets: [
      {
        label: 'Adoption Probability (%)',
        data: adoptionData,
        backgroundColor: 'rgba(59, 130, 246, 0.8)',
        borderColor: 'rgb(59, 130, 246)',
        borderWidth: 1,
      },
      {
        label: 'Market Impact (%)',
        data: impactData,
        backgroundColor: 'rgba(34, 197, 94, 0.8)',
        borderColor: 'rgb(34, 197, 94)',
        borderWidth: 1,
      },
      {
        label: 'Risk Score (%)',
        data: riskData,
        backgroundColor: 'rgba(239, 68, 68, 0.8)',
        borderColor: 'rgb(239, 68, 68)',
        borderWidth: 1,
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
      x: {
        ticks: {
          maxRotation: 45,
          minRotation: 0,
          autoSkip: true,
          maxTicksLimit: 10,
        },
      },
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
  const avgAdoption = comparisonPredictions.reduce((sum, pred) => sum + pred.adoption_probability, 0) / comparisonPredictions.length;
  const avgImpact = comparisonPredictions.reduce((sum, pred) => sum + pred.market_impact_score, 0) / comparisonPredictions.length;
  const avgRisk = comparisonPredictions.reduce((sum, pred) => sum + pred.risk_score, 0) / comparisonPredictions.length;

  const getTechnologyName = (technologyId: number) => {
    const technology = technologies.find(tech => tech.id === technologyId);
    return technology ? technology.name : `Tech #${technologyId}`;
  };

  const handlePredictionClick = (prediction: Prediction) => {
    setSelectedPrediction(prediction);
    setShowModal(true);
  };

  return (
    <div>
      {/* Summary Stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ textAlign: 'center' }}>
          <p style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#2563eb', margin: '0' }}>{(avgAdoption * 100).toFixed(1)}%</p>
          <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: '0' }}>Avg Adoption</p>
        </div>
        <div style={{ textAlign: 'center' }}>
          <p style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#16a34a', margin: '0' }}>{(avgImpact * 100).toFixed(1)}%</p>
          <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: '0' }}>Avg Impact</p>
        </div>
        <div style={{ textAlign: 'center' }}>
          <p style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#dc2626', margin: '0' }}>{(avgRisk * 100).toFixed(1)}%</p>
          <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: '0' }}>Avg Risk</p>
        </div>
      </div>

      {/* Chart */}
      <div style={{ height: '16rem' }}>
        <Bar data={chartData} options={options} />
      </div>

      {/* Recent Predictions */}
      <div style={{ marginTop: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <h4 style={{ fontSize: '0.875rem', fontWeight: '500', color: '#111827', margin: '0' }}>Recent Predictions</h4>
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
            View All
          </button>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {predictions.slice(0, 3).map((prediction) => {
            const technology = technologies.find(tech => tech.id === prediction.technology_id);
            return (
              <div 
                key={prediction.id} 
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  fontSize: '0.875rem',
                  padding: '0.5rem',
                  borderRadius: '0.375rem',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onMouseOver={(e) => e.currentTarget.style.background = '#f3f4f6'}
                onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
                onClick={() => handlePredictionClick(prediction)}
              >
                <div>
                  <span style={{ fontWeight: '500', color: '#111827' }}>
                    {technology ? technology.name : `Tech #${prediction.technology_id}`}
                  </span>
                  <span style={{ color: '#6b7280', marginLeft: '0.5rem' }}>
                    {new Date(prediction.target_date).toLocaleDateString()}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{
                    padding: '0.25rem 0.5rem',
                    borderRadius: '0.25rem',
                    fontSize: '0.75rem',
                    fontWeight: '500',
                    background: prediction.adoption_probability >= 0.7 ? '#dcfce7' : 
                               prediction.adoption_probability >= 0.4 ? '#fef3c7' : '#fee2e2',
                    color: prediction.adoption_probability >= 0.7 ? '#166534' : 
                          prediction.adoption_probability >= 0.4 ? '#92400e' : '#991b1b'
                  }}>
                    {(prediction.adoption_probability * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Prediction Detail Modal */}
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
            boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
            position: 'relative'
          }}>
            {/* Close Button */}
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
                color: '#6b7280',
                width: '2rem',
                height: '2rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                borderRadius: '0.375rem'
              }}
              onMouseOver={(e) => e.currentTarget.style.background = '#f3f4f6'}
              onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
            >
              ×
            </button>

            <h2 style={{ 
              fontSize: '1.5rem', 
              fontWeight: '600', 
              color: '#111827', 
              marginBottom: '1.5rem',
              paddingRight: '2rem'
            }}>
              🔮 Predictions Analysis
            </h2>

            {/* All Predictions Table */}
            <div style={{ marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '600', color: '#111827', marginBottom: '1rem' }}>
                All Predictions
              </h3>
              <div style={{
                background: '#f9fafb',
                borderRadius: '0.5rem',
                overflow: 'hidden',
                border: '1px solid #e5e7eb'
              }}>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr 1fr',
                  gap: '1rem',
                  padding: '1rem',
                  background: '#f3f4f6',
                  borderBottom: '1px solid #e5e7eb',
                  fontWeight: '600',
                  fontSize: '0.875rem',
                  color: '#374151'
                }}>
                  <div>Technology</div>
                  <div>Adoption</div>
                  <div>Impact</div>
                  <div>Risk</div>
                  <div>Confidence</div>
                  <div>Target Date</div>
                </div>
                {predictions.map((prediction) => {
                  const technology = technologies.find(tech => tech.id === prediction.technology_id);
                  return (
                    <div 
                      key={prediction.id}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: '2fr 1fr 1fr 1fr 1fr 1fr',
                        gap: '1rem',
                        padding: '1rem',
                        borderBottom: '1px solid #e5e7eb',
                        fontSize: '0.875rem',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                      }}
                      onMouseOver={(e) => e.currentTarget.style.background = '#f3f4f6'}
                      onMouseOut={(e) => e.currentTarget.style.background = 'white'}
                      onClick={() => handlePredictionClick(prediction)}
                    >
                      <div style={{ fontWeight: '500', color: '#111827' }}>
                        {technology ? technology.name : `Tech #${prediction.technology_id}`}
                        {technology && (
                          <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.25rem' }}>
                            {technology.category}
                          </div>
                        )}
                      </div>
                      <div style={{
                        color: prediction.adoption_probability >= 0.7 ? '#16a34a' : 
                               prediction.adoption_probability >= 0.4 ? '#ca8a04' : '#dc2626',
                        fontWeight: '600'
                      }}>
                        {(prediction.adoption_probability * 100).toFixed(1)}%
                      </div>
                      <div style={{
                        color: prediction.market_impact_score >= 0.7 ? '#16a34a' : 
                               prediction.market_impact_score >= 0.4 ? '#ca8a04' : '#dc2626',
                        fontWeight: '600'
                      }}>
                        {(prediction.market_impact_score * 100).toFixed(1)}%
                      </div>
                      <div style={{
                        color: prediction.risk_score <= 0.3 ? '#16a34a' : 
                               prediction.risk_score <= 0.6 ? '#ca8a04' : '#dc2626',
                        fontWeight: '600'
                      }}>
                        {(prediction.risk_score * 100).toFixed(1)}%
                      </div>
                      <div style={{
                        color: prediction.confidence_interval >= 0.8 ? '#16a34a' : 
                               prediction.confidence_interval >= 0.6 ? '#ca8a04' : '#dc2626',
                        fontWeight: '600'
                      }}>
                        {(prediction.confidence_interval * 100).toFixed(1)}%
                      </div>
                      <div style={{ color: '#6b7280' }}>
                        {new Date(prediction.target_date).toLocaleDateString()}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Selected Prediction Details */}
            {selectedPrediction && (
              <div>
                <h3 style={{ fontSize: '1.125rem', fontWeight: '600', color: '#111827', marginBottom: '1rem' }}>
                  Prediction Details
                </h3>
                <div style={{
                  background: '#f9fafb',
                  borderRadius: '0.5rem',
                  padding: '1.5rem',
                  border: '1px solid #e5e7eb'
                }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                    <div>
                      <h4 style={{ fontSize: '1rem', fontWeight: '600', color: '#111827', marginBottom: '0.5rem' }}>
                        Technology
                      </h4>
                      <p style={{ color: '#6b7280', margin: '0' }}>
                        {getTechnologyName(selectedPrediction.technology_id)}
                      </p>
                    </div>
                    <div>
                      <h4 style={{ fontSize: '1rem', fontWeight: '600', color: '#111827', marginBottom: '0.5rem' }}>
                        Model Used
                      </h4>
                      <p style={{ color: '#6b7280', margin: '0' }}>
                        {selectedPrediction.model_used}
                      </p>
                    </div>
                    <div>
                      <h4 style={{ fontSize: '1rem', fontWeight: '600', color: '#111827', marginBottom: '0.5rem' }}>
                        Prediction Date
                      </h4>
                      <p style={{ color: '#6b7280', margin: '0' }}>
                        {new Date(selectedPrediction.prediction_date).toLocaleDateString()}
                      </p>
                    </div>
                    <div>
                      <h4 style={{ fontSize: '1rem', fontWeight: '600', color: '#111827', marginBottom: '0.5rem' }}>
                        Target Date
                      </h4>
                      <p style={{ color: '#6b7280', margin: '0' }}>
                        {new Date(selectedPrediction.target_date).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  
                  {selectedPrediction.prediction_reasoning && (
                    <div style={{ marginTop: '1.5rem' }}>
                      <h4 style={{ fontSize: '1rem', fontWeight: '600', color: '#111827', marginBottom: '0.5rem' }}>
                        Reasoning
                      </h4>
                      <p style={{ color: '#6b7280', margin: '0', lineHeight: '1.5' }}>
                        {selectedPrediction.prediction_reasoning}
                      </p>
                    </div>
                  )}
                  
                  {selectedPrediction.features_used && selectedPrediction.features_used.length > 0 && (
                    <div style={{ marginTop: '1.5rem' }}>
                      <h4 style={{ fontSize: '1rem', fontWeight: '600', color: '#111827', marginBottom: '0.5rem' }}>
                        Features Used
                      </h4>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                        {selectedPrediction.features_used.map((feature, index) => (
                          <span key={index} style={{
                            background: '#e5e7eb',
                            color: '#374151',
                            padding: '0.25rem 0.5rem',
                            borderRadius: '0.25rem',
                            fontSize: '0.75rem'
                          }}>
                            {feature}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PredictionChart; 