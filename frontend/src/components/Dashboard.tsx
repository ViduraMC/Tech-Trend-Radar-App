import React, { useState, useEffect } from 'react';
import axios from 'axios';
import TrendCard from './TrendCard';
import PredictionChart from './PredictionChart';
import TimeSeriesPredictions from './TimeSeriesPredictions';
import TechnologyList from './TechnologyList';
import AnalyticsSummary from './AnalyticsSummary';

interface Technology {
  id: number;
  name: string;
  category: string;
  description: string;
  keywords: string[];
  first_detected: string;
  last_updated: string;
}

interface TrendData {
  id: number;
  technology_id: number;
  source: string;
  date: string;
  github_stars: number;
  github_forks: number;
  github_issues: number;
  arxiv_papers: number;
  patent_filings: number;
  job_postings: number;
  social_mentions: number;
  trend_score: number;
  momentum_score: number;
  adoption_score: number;
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

const Dashboard: React.FC = () => {
  const [technologies, setTechnologies] = useState<Technology[]>([]);
  const [trends, setTrends] = useState<TrendData[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'trends' | 'predictions' | 'technologies'>('overview');

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [techResponse, trendsResponse, predictionsResponse] = await Promise.all([
        axios.get('http://localhost:8000/api/technologies/?limit=1000'),
        axios.get('http://localhost:8000/api/trends/'),
        axios.get('http://localhost:8000/api/predictions/?limit=100')
      ]);

      setTechnologies(techResponse.data);
      setTrends(trendsResponse.data.trends || []);
      setPredictions(predictionsResponse.data.predictions || []);
    } catch (err) {
      setError('Failed to fetch dashboard data');
      console.error('Dashboard data fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Hybrid ranking algorithm that considers multiple factors
  const calculateTrendingScore = (trend: TrendData) => {
    // Base popularity (GitHub stars) - 40% weight
    const maxStars = Math.max(...trends.map(t => t.github_stars));
    const normalizedStars = trend.github_stars / maxStars;
    
    // Community engagement (forks) - 20% weight
    const maxForks = Math.max(...trends.map(t => t.github_forks));
    const normalizedForks = trend.github_forks / maxForks;
    
    // Market demand (job postings) - 25% weight
    const maxJobs = Math.max(...trends.map(t => t.job_postings)) || 1;
    const normalizedJobs = trend.job_postings / maxJobs;
    
    // Social buzz (mentions) - 10% weight
    const maxSocial = Math.max(...trends.map(t => t.social_mentions)) || 1;
    const normalizedSocial = trend.social_mentions / maxSocial;
    
    // Research activity (papers) - 5% weight
    const maxPapers = Math.max(...trends.map(t => t.arxiv_papers)) || 1;
    const normalizedPapers = trend.arxiv_papers / maxPapers;
    
    // Calculate weighted score
    const trendingScore = (
      normalizedStars * 0.40 +
      normalizedForks * 0.20 +
      normalizedJobs * 0.25 +
      normalizedSocial * 0.10 +
      normalizedPapers * 0.05
    );
    
    return trendingScore;
  };

  // Get sorted trends by hybrid ranking
  const getTrendingTechnologies = () => {
    return trends
      .map(trend => ({
        ...trend,
        trendingScore: calculateTrendingScore(trend)
      }))
      .sort((a, b) => b.trendingScore - a.trendingScore);
  };

  if (loading) {
    return (
      <div style={{ 
        minHeight: '100vh', 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center' 
      }}>
        <div className="text-center">
          <div style={{ 
            width: '4rem', 
            height: '4rem', 
            border: '3px solid rgba(255,255,255,0.3)', 
            borderTop: '3px solid white', 
            borderRadius: '50%', 
            animation: 'spin 1s linear infinite', 
            margin: '0 auto' 
          }}></div>
          <p style={{ marginTop: '1rem', color: 'white', fontSize: '1.1rem', fontWeight: '500' }}>
            Loading Tech Trend Radar...
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ 
        minHeight: '100vh', 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center' 
      }}>
        <div className="text-center">
          <div style={{ fontSize: '4rem', color: 'white', marginBottom: '1rem' }}>⚠️</div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'white', marginBottom: '0.5rem' }}>
            Error Loading Dashboard
          </h2>
          <p style={{ color: 'rgba(255,255,255,0.8)', marginBottom: '1rem' }}>{error}</p>
          <button 
            onClick={fetchDashboardData}
            style={{
              background: 'white',
              color: '#667eea',
              border: 'none',
              padding: '0.75rem 1.5rem',
              borderRadius: '0.5rem',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
            onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const trendingTechnologies = getTrendingTechnologies();

  return (
    <div style={{ 
      minHeight: '100vh', 
      background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)'
    }}>
      {/* Header */}
      <header style={{ 
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
        color: 'white'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 1rem' }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            padding: '1.5rem 0' 
          }}>
            <div>
              <h1 style={{ 
                fontSize: '2rem', 
                fontWeight: 'bold', 
                margin: '0',
                textShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                🚀 Tech Trend Radar
              </h1>
              <p style={{ 
                margin: '0.25rem 0 0 0', 
                opacity: 0.9,
                fontSize: '1rem'
              }}>
                Monitor and predict emerging technology trends
              </p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <p style={{ fontSize: '0.875rem', opacity: 0.8, margin: '0' }}>Last updated</p>
              <p style={{ fontSize: '0.875rem', fontWeight: '600', margin: '0' }}>
                {new Date().toLocaleDateString()}
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div style={{ 
        background: 'white', 
        borderBottom: '1px solid #e5e7eb',
        boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 1rem' }}>
          <div style={{ display: 'flex', gap: '0' }}>
            {[
              { id: 'overview', label: '📊 Overview', icon: '📊' },
              { id: 'trends', label: '📈 Trends', icon: '📈' },
              { id: 'predictions', label: '🔮 Predictions', icon: '🔮' },
              { id: 'technologies', label: '🔧 Technologies', icon: '🔧' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                style={{
                  background: activeTab === tab.id ? '#667eea' : 'transparent',
                  color: activeTab === tab.id ? 'white' : '#6b7280',
                  border: 'none',
                  padding: '1rem 1.5rem',
                  fontSize: '1rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  borderBottom: activeTab === tab.id ? '3px solid white' : '3px solid transparent'
                }}
                onMouseOver={(e) => {
                  if (activeTab !== tab.id) {
                    e.currentTarget.style.background = '#f3f4f6';
                  }
                }}
                onMouseOut={(e) => {
                  if (activeTab !== tab.id) {
                    e.currentTarget.style.background = 'transparent';
                  }
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem 1rem' }}>
        {activeTab === 'overview' && (
          <>
            {/* Analytics Summary */}
            <div style={{ marginBottom: '2rem' }}>
              <AnalyticsSummary 
                technologies={technologies}
                trends={trends}
                predictions={predictions}
              />
            </div>

            {/* Quick Stats Grid */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
              gap: '1.5rem',
              marginBottom: '2rem'
            }}>
              {/* Recent Trends */}
              <div style={{
                background: 'white',
                borderRadius: '1rem',
                padding: '1.5rem',
                boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
                border: '1px solid #e5e7eb'
              }}>
                <h3 style={{ 
                  fontSize: '1.25rem', 
                  fontWeight: '600', 
                  color: '#111827', 
                  marginBottom: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}>
                  🔥 Hot Trends
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {trendingTechnologies
                    .slice(0, 3)
                    .map((trend, index) => {
                      const technology = technologies.find(tech => tech.id === trend.technology_id);
                      const rank = index + 1;
                      return (
                        <div key={trend.id} style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          padding: '0.75rem',
                          background: '#f9fafb',
                          borderRadius: '0.5rem'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                            <div style={{
                              width: '2rem',
                              height: '2rem',
                              background: rank === 1 ? 'linear-gradient(135deg, #ffd700, #ffed4e)' :
                                        rank === 2 ? 'linear-gradient(135deg, #c0c0c0, #e5e5e5)' :
                                        'linear-gradient(135deg, #cd7f32, #daa520)',
                              borderRadius: '50%',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontSize: '0.875rem',
                              fontWeight: 'bold',
                              color: '#374151',
                              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                            }}>
                              {rank === 1 ? '🥇' : rank === 2 ? '🥈' : '🥉'}
                            </div>
                            <div>
                              <div style={{ fontWeight: '600', color: '#111827' }}>
                                {technology ? technology.name : `Tech #${trend.technology_id}`}
                              </div>
                              <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                                {technology?.category} • {trend.github_stars.toLocaleString()} stars
                              </div>
                            </div>
                          </div>
                          <div style={{
                            background: trend.trendingScore > 0.7 ? '#10b981' : trend.trendingScore > 0.4 ? '#f59e0b' : '#ef4444',
                            color: 'white',
                            padding: '0.25rem 0.5rem',
                            borderRadius: '0.25rem',
                            fontSize: '0.875rem',
                            fontWeight: '600'
                          }}>
                            {trend.trendingScore.toFixed(2)}
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>

              {/* Predictions Chart */}
              <div style={{
                background: 'white',
                borderRadius: '1rem',
                padding: '1.5rem',
                boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
                border: '1px solid #e5e7eb'
              }}>
                <h3 style={{ 
                  fontSize: '1.25rem', 
                  fontWeight: '600', 
                  color: '#111827', 
                  marginBottom: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}>
                  🔮 Predictions
                </h3>
                <PredictionChart predictions={predictions} technologies={technologies} />
              </div>
            </div>
          </>
        )}

        {activeTab === 'trends' && (
          <>
            {/* Top Rankings Summary */}
            <div style={{
              background: 'white',
              borderRadius: '1rem',
              padding: '1.5rem',
              boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
              border: '1px solid #e5e7eb',
              marginBottom: '1.5rem'
            }}>
              <h3 style={{ 
                fontSize: '1.25rem', 
                fontWeight: '600', 
                color: '#111827', 
                marginBottom: '1rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                🏆 Top Trending Technologies
              </h3>
              <p style={{ 
                fontSize: '0.875rem', 
                color: '#6b7280', 
                marginBottom: '1rem',
                lineHeight: '1.5'
              }}>
                Ranked by hybrid score: GitHub stars (40%), community engagement (20%), market demand (25%), social buzz (10%), research activity (5%)
              </p>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                gap: '1rem' 
              }}>
                {trendingTechnologies
                  .slice(0, 3)
                  .map((trend, index) => {
                    const technology = technologies.find(tech => tech.id === trend.technology_id);
                    const rank = index + 1;
                    return (
                      <div key={trend.id} style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '1rem',
                        padding: '1rem',
                        background: rank === 1 ? 'linear-gradient(135deg, #ffd70020, #ffed4e20)' :
                                  rank === 2 ? 'linear-gradient(135deg, #c0c0c020, #e5e5e520)' :
                                  'linear-gradient(135deg, #cd7f3220, #daa52020)',
                        borderRadius: '0.75rem',
                        border: rank === 1 ? '2px solid #ffd700' :
                                rank === 2 ? '2px solid #c0c0c0' :
                                '2px solid #cd7f32'
                      }}>
                        <div style={{
                          width: '3rem',
                          height: '3rem',
                          background: rank === 1 ? 'linear-gradient(135deg, #ffd700, #ffed4e)' :
                                    rank === 2 ? 'linear-gradient(135deg, #c0c0c0, #e5e5e5)' :
                                    'linear-gradient(135deg, #cd7f32, #daa520)',
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '1.5rem',
                          fontWeight: 'bold',
                          color: '#374151',
                          boxShadow: '0 4px 8px rgba(0,0,0,0.15)'
                        }}>
                          {rank === 1 ? '🥇' : rank === 2 ? '🥈' : '🥉'}
                        </div>
                        <div>
                          <div style={{ 
                            fontWeight: 'bold', 
                            color: '#111827',
                            fontSize: '1.1rem'
                          }}>
                            {technology ? technology.name : `Tech #${trend.technology_id}`}
                          </div>
                          <div style={{ 
                            fontSize: '0.875rem', 
                            color: '#6b7280',
                            marginTop: '0.25rem'
                          }}>
                            Score: {trend.trendingScore.toFixed(2)} • {trend.github_stars.toLocaleString()} stars
                          </div>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* All Trends */}
            <div style={{
              background: 'white',
              borderRadius: '1rem',
              padding: '1.5rem',
              boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
              border: '1px solid #e5e7eb'
            }}>
              <h2 style={{ 
                fontSize: '1.5rem', 
                fontWeight: '600', 
                color: '#111827', 
                marginBottom: '1.5rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                📈 All Trending Technologies
              </h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {trendingTechnologies
                  .slice(0, 10)
                  .map((trend, index) => {
                    const technology = technologies.find(tech => tech.id === trend.technology_id);
                    return (
                      <TrendCard 
                        key={trend.id} 
                        trend={trend} 
                        technology={technology}
                        rank={index + 1}
                      />
                    );
                  })}
              </div>
            </div>
          </>
        )}

        {activeTab === 'predictions' && (
          <div style={{
            background: 'white',
            borderRadius: '1rem',
            padding: '1.5rem',
            boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
            border: '1px solid #e5e7eb'
          }}>
            <h2 style={{ 
              fontSize: '1.5rem', 
              fontWeight: '600', 
              color: '#111827', 
              marginBottom: '1.5rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              🔮 Technology Predictions
            </h2>
            
            {/* Two types of predictions */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
              {/* Comparison Predictions */}
              <div>
                <h3 style={{ 
                  fontSize: '1.25rem', 
                  fontWeight: '600', 
                  color: '#111827', 
                  marginBottom: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}>
                  📊 Comparison Predictions
                </h3>
                <p style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: '1rem' }}>
                  Compare adoption, impact, and risk across different technologies
                </p>
                <PredictionChart predictions={predictions.filter(p => p.model_used !== 'time_series_ml_model')} technologies={technologies} />
              </div>

              {/* Time Series Predictions */}
              <div>
                <h3 style={{ 
                  fontSize: '1.25rem', 
                  fontWeight: '600', 
                  color: '#111827', 
                  marginBottom: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}>
                  📈 Time Series Predictions
                </h3>
                <p style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: '1rem' }}>
                  See how technologies will evolve over time with monthly forecasts
                </p>
                <TimeSeriesPredictions predictions={predictions.filter(p => p.model_used === 'time_series_ml_model')} technologies={technologies} />
              </div>
            </div>
          </div>
        )}

        {activeTab === 'technologies' && (
          <div style={{
            background: 'white',
            borderRadius: '1rem',
            padding: '1.5rem',
            boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
            border: '1px solid #e5e7eb'
          }}>
            <h2 style={{ 
              fontSize: '1.5rem', 
              fontWeight: '600', 
              color: '#111827', 
              marginBottom: '1.5rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}>
              🔧 All Technologies
            </h2>
            <TechnologyList technologies={technologies} />
          </div>
        )}
      </main>
    </div>
  );
};

export default Dashboard; 