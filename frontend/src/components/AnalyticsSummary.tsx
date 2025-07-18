import React from 'react';

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

interface AnalyticsSummaryProps {
  technologies: Technology[];
  trends: TrendData[];
  predictions: Prediction[];
}

const AnalyticsSummary: React.FC<AnalyticsSummaryProps> = ({ 
  technologies, 
  trends, 
  predictions 
}) => {
  // Calculate metrics
  const totalTechnologies = technologies.length;
  const totalTrends = trends.length;
  const totalPredictions = predictions.length;
  
  const avgTrendScore = trends.length > 0 
    ? trends.reduce((sum, trend) => sum + trend.trend_score, 0) / trends.length 
    : 0;
  
  const avgAdoptionProbability = predictions.length > 0
    ? predictions.reduce((sum, pred) => sum + pred.adoption_probability, 0) / predictions.length
    : 0;

  const categoryCounts = technologies.reduce((acc, tech) => {
    acc[tech.category] = (acc[tech.category] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const topCategory = Object.entries(categoryCounts)
    .sort(([,a], [,b]) => b - a)[0] || ['None', 0];

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

  const metrics = [
    {
      name: 'Total Technologies',
      value: totalTechnologies,
      change: '+12%',
      changeType: 'positive',
      icon: '🚀',
      color: '#3b82f6',
      bgColor: '#dbeafe'
    },
    {
      name: 'Active Trends',
      value: totalTrends,
      change: '+8%',
      changeType: 'positive',
      icon: '📈',
      color: '#10b981',
      bgColor: '#d1fae5'
    },
    {
      name: 'Avg Trend Score',
      value: avgTrendScore.toFixed(2),
      change: '+5%',
      changeType: 'positive',
      icon: '⭐',
      color: '#f59e0b',
      bgColor: '#fef3c7'
    },
    {
      name: 'Predictions',
      value: totalPredictions,
      change: '+15%',
      changeType: 'positive',
      icon: '🔮',
      color: '#8b5cf6',
      bgColor: '#ede9fe'
    },
    {
      name: 'Avg Adoption Probability',
      value: `${(avgAdoptionProbability * 100).toFixed(1)}%`,
      change: '+3%',
      changeType: 'positive',
      icon: '📊',
      color: '#ec4899',
      bgColor: '#fce7f3'
    },
    {
      name: 'Top Category',
      value: topCategory[0],
      change: `${topCategory[1]} technologies`,
      changeType: 'neutral',
      icon: '🏆',
      color: '#f97316',
      bgColor: '#fed7aa'
    }
  ];

  return (
    <div style={{
      background: 'white',
      borderRadius: '1.5rem',
      padding: '2rem',
      boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
      border: '1px solid #e5e7eb'
    }}>
      <h2 style={{ 
        fontSize: '1.75rem', 
        fontWeight: 'bold', 
        color: '#111827', 
        marginBottom: '2rem',
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem'
      }}>
        📊 Analytics Summary
      </h2>
      
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', 
        gap: '1.5rem',
        marginBottom: '2rem'
      }}>
        {metrics.map((metric, index) => (
          <div key={index} style={{
            background: 'white',
            borderRadius: '1rem',
            padding: '1.5rem',
            border: `2px solid ${metric.bgColor}`,
            transition: 'all 0.2s ease',
            cursor: 'pointer'
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = 'translateY(-4px)';
            e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = 'none';
          }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <div style={{
                width: '3rem',
                height: '3rem',
                background: metric.bgColor,
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.5rem'
              }}>
                {metric.icon}
              </div>
              <div style={{
                background: metric.changeType === 'positive' ? '#dcfce7' : 
                          metric.changeType === 'negative' ? '#fed7d7' : '#f3f4f6',
                color: metric.changeType === 'positive' ? '#166534' : 
                       metric.changeType === 'negative' ? '#991b1b' : '#6b7280',
                padding: '0.25rem 0.75rem',
                borderRadius: '1rem',
                fontSize: '0.75rem',
                fontWeight: '600'
              }}>
                {metric.change}
              </div>
            </div>
            
            <div>
              <p style={{ 
                fontSize: '0.875rem', 
                fontWeight: '600', 
                color: '#6b7280',
                margin: '0 0 0.5rem 0'
              }}>
                {metric.name}
              </p>
              <p style={{ 
                fontSize: '2rem', 
                fontWeight: 'bold', 
                color: metric.color,
                margin: '0'
              }}>
                {metric.value}
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Category Distribution */}
      <div>
        <h3 style={{ 
          fontSize: '1.25rem', 
          fontWeight: '600', 
          color: '#111827', 
          marginBottom: '1.5rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          🏷️ Technology Categories
        </h3>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
          gap: '1rem'
        }}>
          {Object.entries(categoryCounts)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 8)
            .map(([category, count]) => (
            <div key={category} style={{
              background: 'white',
              borderRadius: '0.75rem',
              padding: '1rem',
              border: `2px solid ${getCategoryColor(category)}20`,
              transition: 'all 0.2s ease',
              cursor: 'pointer'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 4px 12px rgba(0,0,0,0.1)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = 'none';
            }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <p style={{ 
                    fontSize: '0.875rem', 
                    fontWeight: '600', 
                    color: getCategoryColor(category),
                    margin: '0 0 0.25rem 0'
                  }}>
                    {category}
                  </p>
                  <p style={{ 
                    fontSize: '1.5rem', 
                    fontWeight: 'bold', 
                    color: '#111827',
                    margin: '0'
                  }}>
                    {count}
                  </p>
                </div>
                <div style={{
                  width: '2rem',
                  height: '2rem',
                  background: `${getCategoryColor(category)}20`,
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.875rem',
                  fontWeight: 'bold',
                  color: getCategoryColor(category)
                }}>
                  {category.charAt(0)}
                </div>
              </div>
            </div>
          ))}
        </div>
        
        {Object.keys(categoryCounts).length > 8 && (
          <div style={{ 
            textAlign: 'center', 
            marginTop: '1rem',
            padding: '1rem',
            background: '#f9fafb',
            borderRadius: '0.5rem',
            color: '#6b7280',
            fontSize: '0.875rem'
          }}>
            +{Object.keys(categoryCounts).length - 8} more categories
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalyticsSummary; 