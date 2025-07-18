import React, { useState } from 'react';

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
  trendingScore?: number;
}

interface Technology {
  id: number;
  name: string;
  category: string;
  description: string;
}

interface TrendCardProps {
  trend: TrendData;
  technology?: Technology;
  rank?: number;
}

const TrendCard: React.FC<TrendCardProps> = ({ trend, technology, rank }) => {
  const [showModal, setShowModal] = useState(false);

  const getTrendScoreColor = (score: number) => {
    if (score >= 0.8) return { bg: '#dcfce7', text: '#166534', border: '#22c55e' };
    if (score >= 0.6) return { bg: '#fef3c7', text: '#92400e', border: '#f59e0b' };
    if (score >= 0.4) return { bg: '#fed7d7', text: '#991b1b', border: '#ef4444' };
    return { bg: '#f3f4f6', text: '#6b7280', border: '#9ca3af' };
  };

  const getTrendScoreLabel = (score: number) => {
    if (score >= 0.8) return '🔥 Hot';
    if (score >= 0.6) return '📈 Rising';
    if (score >= 0.4) return '📊 Stable';
    return '📉 Declining';
  };

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      'AI/ML': '#8b5cf6',
      'Blockchain': '#f59e0b',
      'Cloud': '#3b82f6',
      'Frontend': '#10b981',
      'Backend': '#ef4444',
      'Mobile': '#6366f1',
      'Data': '#ec4899',
      'DevOps': '#6b7280',
      'IoT': '#f97316',
      'Quantum': '#06b6d4',
      'AR/VR': '#059669',
      'Cybersecurity': '#dc2626'
    };
    return colors[category] || '#6b7280';
  };

  // Display technology name if available, otherwise fallback to ID
  const displayName = technology ? technology.name : `Technology #${trend.technology_id}`;
  const scoreColors = getTrendScoreColor(trend.trend_score);

  return (
    <>
      <div style={{
        background: 'white',
        borderRadius: '1rem',
        padding: '1.5rem',
        boxShadow: '0 4px 6px rgba(0,0,0,0.05)',
        border: '1px solid #e5e7eb',
        transition: 'all 0.2s ease',
        cursor: 'pointer'
      }}
      onMouseOver={(e) => {
        e.currentTarget.style.transform = 'translateY(-2px)';
        e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.1)';
      }}
      onMouseOut={(e) => {
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = '0 4px 6px rgba(0,0,0,0.05)';
      }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            {/* Ranking Badge */}
            {rank && (
              <div style={{
                width: '3rem',
                height: '3rem',
                background: rank <= 3 ? 
                  rank === 1 ? 'linear-gradient(135deg, #ffd700, #ffed4e)' : // Gold
                  rank === 2 ? 'linear-gradient(135deg, #c0c0c0, #e5e5e5)' : // Silver
                  'linear-gradient(135deg, #cd7f32, #daa520)' : // Bronze
                  'linear-gradient(135deg, #f3f4f6, #e5e7eb)', // Default
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: rank <= 3 ? '2px solid #374151' : '2px solid #d1d5db',
                fontWeight: 'bold',
                fontSize: '1.25rem',
                color: rank <= 3 ? '#374151' : '#6b7280',
                boxShadow: rank <= 3 ? '0 2px 8px rgba(0,0,0,0.15)' : 'none'
              }}>
                {rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : rank}
              </div>
            )}
            <div style={{
              width: '3rem',
              height: '3rem',
              background: `linear-gradient(135deg, ${getCategoryColor(technology?.category || 'Other')}20, ${getCategoryColor(technology?.category || 'Other')}40)`,
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              border: `2px solid ${getCategoryColor(technology?.category || 'Other')}`
            }}>
              <span style={{ 
                fontSize: '1.25rem', 
                fontWeight: 'bold',
                color: getCategoryColor(technology?.category || 'Other')
              }}>
                {technology ? technology.name.charAt(0).toUpperCase() : 'T'}
              </span>
            </div>
            <div>
              <h3 style={{ 
                fontSize: '1.25rem', 
                fontWeight: 'bold', 
                color: '#111827',
                margin: '0 0 0.25rem 0',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                {rank && (
                  <span style={{
                    fontSize: '0.875rem',
                    fontWeight: '600',
                    color: rank <= 3 ? '#374151' : '#6b7280',
                    background: rank <= 3 ? '#f3f4f6' : 'transparent',
                    padding: '0.25rem 0.5rem',
                    borderRadius: '0.25rem',
                    border: rank <= 3 ? '1px solid #d1d5db' : 'none'
                  }}>
                    #{rank}
                  </span>
                )}
                {displayName}
              </h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{
                  background: `${getCategoryColor(technology?.category || 'Other')}20`,
                  color: getCategoryColor(technology?.category || 'Other'),
                  padding: '0.25rem 0.5rem',
                  borderRadius: '0.5rem',
                  fontSize: '0.75rem',
                  fontWeight: '600'
                }}>
                  {technology?.category || 'Unknown'}
                </span>
                <span style={{ color: '#6b7280', fontSize: '0.875rem' }}>
                  {new Date(trend.date).toLocaleDateString()}
                </span>
              </div>
            </div>
          </div>
          
                  <div style={{
          background: scoreColors.bg,
          color: scoreColors.text,
          border: `2px solid ${scoreColors.border}`,
          padding: '0.5rem 1rem',
          borderRadius: '2rem',
          fontSize: '0.875rem',
          fontWeight: '700',
          textAlign: 'center'
        }}>
          {trend.trendingScore ? `Trending ${trend.trendingScore.toFixed(2)}` : getTrendScoreLabel(trend.trend_score)}
          <div style={{ fontSize: '0.75rem', opacity: 0.8 }}>
            {trend.trendingScore ? 'Hybrid Score' : trend.trend_score.toFixed(2)}
          </div>
        </div>
        </div>

        {/* Metrics Grid */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', 
          gap: '1rem',
          marginBottom: '1rem'
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ 
              fontSize: '1.5rem', 
              fontWeight: 'bold', 
              color: '#111827',
              marginBottom: '0.25rem'
            }}>
              {trend.github_stars.toLocaleString()}
            </div>
            <div style={{ 
              fontSize: '0.75rem', 
              color: '#6b7280',
              fontWeight: '500'
            }}>
              ⭐ GitHub Stars
            </div>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <div style={{ 
              fontSize: '1.5rem', 
              fontWeight: 'bold', 
              color: '#111827',
              marginBottom: '0.25rem'
            }}>
              {trend.github_forks.toLocaleString()}
            </div>
            <div style={{ 
              fontSize: '0.75rem', 
              color: '#6b7280',
              fontWeight: '500'
            }}>
              🔄 Forks
            </div>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <div style={{ 
              fontSize: '1.5rem', 
              fontWeight: 'bold', 
              color: '#111827',
              marginBottom: '0.25rem'
            }}>
              {trend.job_postings.toLocaleString()}
            </div>
            <div style={{ 
              fontSize: '0.75rem', 
              color: '#6b7280',
              fontWeight: '500'
            }}>
              💼 Job Postings
            </div>
          </div>
          
          <div style={{ textAlign: 'center' }}>
            <div style={{ 
              fontSize: '1.5rem', 
              fontWeight: 'bold', 
              color: '#111827',
              marginBottom: '0.25rem'
            }}>
              {trend.arxiv_papers.toLocaleString()}
            </div>
            <div style={{ 
              fontSize: '0.75rem', 
              color: '#6b7280',
              fontWeight: '500'
            }}>
              📄 Research Papers
            </div>
          </div>
        </div>

        {/* Additional Metrics */}
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          padding: '0.75rem',
          background: '#f9fafb',
          borderRadius: '0.5rem',
          fontSize: '0.875rem'
        }}>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <span style={{ color: '#6b7280', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              🐛 {trend.github_issues} issues
            </span>
            <span style={{ color: '#6b7280', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              📋 {trend.patent_filings} patents
            </span>
            <span style={{ color: '#6b7280', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              💬 {trend.social_mentions} mentions
            </span>
          </div>
          
          <button 
            onClick={() => setShowModal(true)}
            style={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white',
              border: 'none',
              padding: '0.5rem 1rem',
              borderRadius: '0.5rem',
              fontSize: '0.875rem',
              fontWeight: '600',
              cursor: 'pointer',
              transition: 'all 0.2s'
            }}
            onMouseOver={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
            onMouseOut={(e) => e.currentTarget.style.transform = 'scale(1)'}
          >
            View Details →
          </button>
        </div>
      </div>

      {/* Detailed Modal */}
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
          zIndex: 1000,
          padding: '1rem'
        }}
        onClick={() => setShowModal(false)}
        >
          <div style={{
            background: 'white',
            borderRadius: '1rem',
            padding: '2rem',
            maxWidth: '800px',
            width: '100%',
            maxHeight: '90vh',
            overflow: 'auto',
            boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)'
          }}
          onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                {rank && (
                  <div style={{
                    width: '4rem',
                    height: '4rem',
                    background: rank <= 3 ? 
                      rank === 1 ? 'linear-gradient(135deg, #ffd700, #ffed4e)' :
                      rank === 2 ? 'linear-gradient(135deg, #c0c0c0, #e5e5e5)' :
                      'linear-gradient(135deg, #cd7f32, #daa520)' :
                      'linear-gradient(135deg, #f3f4f6, #e5e7eb)',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: rank <= 3 ? '3px solid #374151' : '3px solid #d1d5db',
                    fontWeight: 'bold',
                    fontSize: '2rem',
                    color: rank <= 3 ? '#374151' : '#6b7280',
                    boxShadow: rank <= 3 ? '0 4px 12px rgba(0,0,0,0.2)' : 'none'
                  }}>
                    {rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : rank}
                  </div>
                )}
                <div>
                  <h2 style={{ 
                    fontSize: '2rem', 
                    fontWeight: 'bold', 
                    color: '#111827',
                    margin: '0 0 0.5rem 0'
                  }}>
                    {displayName}
                  </h2>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <span style={{
                      background: `${getCategoryColor(technology?.category || 'Other')}20`,
                      color: getCategoryColor(technology?.category || 'Other'),
                      padding: '0.5rem 1rem',
                      borderRadius: '2rem',
                      fontSize: '1rem',
                      fontWeight: '600'
                    }}>
                      {technology?.category || 'Unknown'}
                    </span>
                    <span style={{ color: '#6b7280', fontSize: '1rem' }}>
                      Last updated: {new Date(trend.date).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>
              <button 
                onClick={() => setShowModal(false)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '2rem',
                  cursor: 'pointer',
                  color: '#6b7280',
                  padding: '0.5rem',
                  borderRadius: '0.5rem',
                  transition: 'all 0.2s'
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = '#f3f4f6';
                  e.currentTarget.style.color = '#374151';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'none';
                  e.currentTarget.style.color = '#6b7280';
                }}
              >
                ×
              </button>
            </div>

            {/* Technology Description */}
            {technology?.description && (
              <div style={{ marginBottom: '2rem' }}>
                <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#111827', marginBottom: '1rem' }}>
                  📝 Description
                </h3>
                <p style={{ 
                  fontSize: '1rem', 
                  color: '#374151', 
                  lineHeight: '1.6',
                  background: '#f9fafb',
                  padding: '1rem',
                  borderRadius: '0.5rem',
                  border: '1px solid #e5e7eb'
                }}>
                  {technology.description}
                </p>
              </div>
            )}

            {/* Detailed Metrics Grid */}
            <div style={{ marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#111827', marginBottom: '1rem' }}>
                📊 Detailed Metrics
              </h3>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                gap: '1.5rem' 
              }}>
                {/* GitHub Metrics */}
                <div style={{
                  background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
                  padding: '1.5rem',
                  borderRadius: '1rem',
                  border: '1px solid #e2e8f0'
                }}>
                  <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#111827', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    🐙 GitHub Activity
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ color: '#6b7280' }}>Stars:</span>
                      <span style={{ fontWeight: '600', color: '#111827' }}>{trend.github_stars.toLocaleString()}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ color: '#6b7280' }}>Forks:</span>
                      <span style={{ fontWeight: '600', color: '#111827' }}>{trend.github_forks.toLocaleString()}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ color: '#6b7280' }}>Issues:</span>
                      <span style={{ fontWeight: '600', color: '#111827' }}>{trend.github_issues.toLocaleString()}</span>
                    </div>
                  </div>
                </div>

                {/* Research & Development */}
                <div style={{
                  background: 'linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%)',
                  padding: '1.5rem',
                  borderRadius: '1rem',
                  border: '1px solid #e0f2fe'
                }}>
                  <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#111827', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    🔬 Research & Development
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ color: '#6b7280' }}>Research Papers:</span>
                      <span style={{ fontWeight: '600', color: '#111827' }}>{trend.arxiv_papers.toLocaleString()}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ color: '#6b7280' }}>Patent Filings:</span>
                      <span style={{ fontWeight: '600', color: '#111827' }}>{trend.patent_filings.toLocaleString()}</span>
                    </div>
                  </div>
                </div>

                {/* Market Activity */}
                <div style={{
                  background: 'linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%)',
                  padding: '1.5rem',
                  borderRadius: '1rem',
                  border: '1px solid #dcfce7'
                }}>
                  <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#111827', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    💼 Market Activity
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ color: '#6b7280' }}>Job Postings:</span>
                      <span style={{ fontWeight: '600', color: '#111827' }}>{trend.job_postings.toLocaleString()}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ color: '#6b7280' }}>Social Mentions:</span>
                      <span style={{ fontWeight: '600', color: '#111827' }}>{trend.social_mentions.toLocaleString()}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Trend Analysis */}
            <div style={{ marginBottom: '2rem' }}>
              <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#111827', marginBottom: '1rem' }}>
                📈 Trend Analysis
              </h3>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                gap: '1rem' 
              }}>
                <div style={{
                  background: scoreColors.bg,
                  color: scoreColors.text,
                  border: `2px solid ${scoreColors.border}`,
                  padding: '1rem',
                  borderRadius: '0.75rem',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                    {trend.trend_score.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '0.875rem', fontWeight: '600' }}>
                    {getTrendScoreLabel(trend.trend_score)}
                  </div>
                </div>
                <div style={{
                  background: '#fef3c7',
                  color: '#92400e',
                  border: '2px solid #f59e0b',
                  padding: '1rem',
                  borderRadius: '0.75rem',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                    {trend.momentum_score.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '0.875rem', fontWeight: '600' }}>
                    Momentum Score
                  </div>
                </div>
                <div style={{
                  background: '#dcfce7',
                  color: '#166534',
                  border: '2px solid #22c55e',
                  padding: '1rem',
                  borderRadius: '0.75rem',
                  textAlign: 'center'
                }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                    {trend.adoption_score.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '0.875rem', fontWeight: '600' }}>
                    Adoption Score
                  </div>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end' }}>
              <button 
                onClick={() => setShowModal(false)}
                style={{
                  background: '#f3f4f6',
                  color: '#374151',
                  border: '1px solid #d1d5db',
                  padding: '0.75rem 1.5rem',
                  borderRadius: '0.5rem',
                  fontSize: '1rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onMouseOver={(e) => e.currentTarget.style.background = '#e5e7eb'}
                onMouseOut={(e) => e.currentTarget.style.background = '#f3f4f6'}
              >
                Close
              </button>
              <button 
                style={{
                  background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                  color: 'white',
                  border: 'none',
                  padding: '0.75rem 1.5rem',
                  borderRadius: '0.5rem',
                  fontSize: '1rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onMouseOver={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                onMouseOut={(e) => e.currentTarget.style.transform = 'scale(1)'}
              >
                Track Technology →
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default TrendCard; 