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

  const metrics = [
    {
      name: 'Total Technologies',
      value: totalTechnologies,
      change: '+12%',
      changeType: 'positive',
      icon: '🚀'
    },
    {
      name: 'Active Trends',
      value: totalTrends,
      change: '+8%',
      changeType: 'positive',
      icon: '📈'
    },
    {
      name: 'Avg Trend Score',
      value: avgTrendScore.toFixed(2),
      change: '+5%',
      changeType: 'positive',
      icon: '⭐'
    },
    {
      name: 'Predictions',
      value: totalPredictions,
      change: '+15%',
      changeType: 'positive',
      icon: '🔮'
    },
    {
      name: 'Avg Adoption Probability',
      value: `${(avgAdoptionProbability * 100).toFixed(1)}%`,
      change: '+3%',
      changeType: 'positive',
      icon: '📊'
    },
    {
      name: 'Top Category',
      value: topCategory[0],
      change: `${topCategory[1]} technologies`,
      changeType: 'neutral',
      icon: '🏆'
    }
  ];

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-6">Analytics Summary</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {metrics.map((metric, index) => (
          <div key={index} className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">{metric.name}</p>
                <p className="text-2xl font-bold text-gray-900">{metric.value}</p>
              </div>
              <div className="text-3xl">{metric.icon}</div>
            </div>
            <div className="mt-2">
              <span className={`text-sm font-medium ${
                metric.changeType === 'positive' ? 'text-green-600' : 
                metric.changeType === 'negative' ? 'text-red-600' : 'text-gray-600'
              }`}>
                {metric.change}
              </span>
              <span className="text-sm text-gray-500 ml-1">from last month</span>
            </div>
          </div>
        ))}
      </div>

      {/* Category Distribution */}
      <div className="mt-8">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Technology Categories</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          {Object.entries(categoryCounts).map(([category, count]) => (
            <div key={category} className="text-center">
              <div className="bg-blue-100 rounded-lg p-3">
                <p className="text-sm font-medium text-blue-900">{category}</p>
                <p className="text-2xl font-bold text-blue-600">{count}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AnalyticsSummary; 