import React, { useState, useEffect } from 'react';
import axios from 'axios';
import TrendCard from './TrendCard';
import PredictionChart from './PredictionChart';
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

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [techResponse, trendsResponse, predictionsResponse] = await Promise.all([
        axios.get('http://localhost:8000/api/technologies/?limit=1000'),
        axios.get('http://localhost:8000/api/trends/'),
        axios.get('http://localhost:8000/api/predictions/')
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

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', backgroundColor: '#f9fafb', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div className="text-center">
          <div style={{ width: '8rem', height: '8rem', border: '2px solid #3b82f6', borderTop: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto' }}></div>
          <p style={{ marginTop: '1rem', color: '#6b7280' }}>Loading Tech Trend Radar...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ minHeight: '100vh', backgroundColor: '#f9fafb', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div className="text-center">
          <div style={{ fontSize: '3rem', color: '#dc2626', marginBottom: '1rem' }}>⚠️</div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#1f2937', marginBottom: '0.5rem' }}>Error Loading Dashboard</h2>
          <p style={{ color: '#6b7280', marginBottom: '1rem' }}>{error}</p>
          <button 
            onClick={fetchDashboardData}
            className="btn"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f9fafb' }}>
      {/* Header */}
      <header style={{ backgroundColor: 'white', boxShadow: '0 1px 3px rgba(0,0,0,0.1)', borderBottom: '1px solid #e5e7eb' }}>
        <div className="container">
          <div className="flex justify-between items-center" style={{ padding: '1.5rem 0' }}>
            <div>
              <h1 style={{ fontSize: '1.875rem', fontWeight: 'bold', color: '#111827' }}>Tech Trend Radar</h1>
              <p style={{ color: '#6b7280' }}>Monitor and predict emerging technology trends</p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <p style={{ fontSize: '0.875rem', color: '#6b7280' }}>Last updated</p>
              <p style={{ fontSize: '0.875rem', fontWeight: '500', color: '#111827' }}>
                {new Date().toLocaleDateString()}
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container" style={{ padding: '2rem 0' }}>
        {/* Analytics Summary */}
        <div style={{ marginBottom: '2rem' }}>
          <AnalyticsSummary 
            technologies={technologies}
            trends={trends}
            predictions={predictions}
          />
        </div>

        {/* Grid Layout */}
        <div className="grid gap-8" style={{ gridTemplateColumns: '1fr' }}>
          {/* Left Column - Trends */}
          <div>
            <div className="card">
              <h2 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#111827', marginBottom: '1rem' }}>Technology Trends</h2>
              <div>
                {trends.slice(0, 5).map((trend) => {
                  // Find the corresponding technology for this trend
                  const technology = technologies.find(tech => tech.id === trend.technology_id);
                  return (
                    <TrendCard 
                      key={trend.id} 
                      trend={trend} 
                      technology={technology}
                    />
                  );
                })}
              </div>
            </div>
          </div>

          {/* Right Column - Predictions Chart */}
          <div>
            <div className="card">
              <h2 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#111827', marginBottom: '1rem' }}>Predictions</h2>
              <PredictionChart predictions={predictions} />
            </div>
          </div>
        </div>

        {/* Technologies List */}
        <div style={{ marginTop: '2rem' }}>
          <div className="card">
            <h2 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#111827', marginBottom: '1rem' }}>All Technologies</h2>
            <TechnologyList technologies={technologies} />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard; 