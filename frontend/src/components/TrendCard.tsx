import React from 'react';

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

interface Technology {
  id: number;
  name: string;
  category: string;
  description: string;
}

interface TrendCardProps {
  trend: TrendData;
  technology?: Technology; // Make it optional for backward compatibility
}

const TrendCard: React.FC<TrendCardProps> = ({ trend, technology }) => {
  const getTrendScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-100';
    if (score >= 0.4) return 'text-orange-600 bg-orange-100';
    return 'text-red-600 bg-red-100';
  };

  const getTrendScoreLabel = (score: number) => {
    if (score >= 0.8) return 'High';
    if (score >= 0.6) return 'Medium';
    if (score >= 0.4) return 'Low';
    return 'Very Low';
  };

  // Display technology name if available, otherwise fallback to ID
  const displayName = technology ? technology.name : `Technology #${trend.technology_id}`;

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
            <span className="text-blue-600 font-semibold">
              {technology ? technology.name.charAt(0).toUpperCase() : 'T'}
            </span>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">{displayName}</h3>
            <p className="text-sm text-gray-500">
              {technology && <span className="text-blue-600 mr-2">{technology.category}</span>}
              {new Date(trend.date).toLocaleDateString()}
            </p>
          </div>
        </div>
        <div className={`px-3 py-1 rounded-full text-sm font-medium ${getTrendScoreColor(trend.trend_score)}`}>
          {getTrendScoreLabel(trend.trend_score)} ({trend.trend_score.toFixed(2)})
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="text-center">
          <p className="text-2xl font-bold text-gray-900">{trend.github_stars.toLocaleString()}</p>
          <p className="text-xs text-gray-500">GitHub Stars</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-gray-900">{trend.github_forks.toLocaleString()}</p>
          <p className="text-xs text-gray-500">Forks</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-gray-900">{trend.job_postings.toLocaleString()}</p>
          <p className="text-xs text-gray-500">Job Postings</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-gray-900">{trend.arxiv_papers.toLocaleString()}</p>
          <p className="text-xs text-gray-500">Research Papers</p>
        </div>
      </div>

      <div className="flex items-center justify-between text-sm text-gray-600">
        <div className="flex space-x-4">
          <span>📊 {trend.github_issues} issues</span>
          <span>📄 {trend.patent_filings} patents</span>
          <span>💬 {trend.social_mentions} mentions</span>
        </div>
        <button className="text-blue-600 hover:text-blue-800 font-medium">
          View Details →
        </button>
      </div>
    </div>
  );
};

export default TrendCard; 