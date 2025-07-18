"""
Ensemble Model that combines predictions from all ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

from .base_model import BaseModel
from .trend_predictor import TrendPredictor
from .adoption_forecaster import AdoptionForecaster
from .technology_classifier import TechnologyClassifier
from .sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)

class EnsembleModel(BaseModel):
    """Ensemble model that combines predictions from all ML models"""
    
    def __init__(self):
        super().__init__("ensemble_model")
        
        # Initialize individual models
        self.trend_predictor = TrendPredictor()
        self.adoption_forecaster = AdoptionForecaster()
        self.technology_classifier = TechnologyClassifier()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Ensemble weights
        self.model_weights = {
            'trend_predictor': 0.3,
            'adoption_forecaster': 0.4,
            'technology_classifier': 0.2,
            'sentiment_analyzer': 0.1
        }
        
        # Prediction targets
        self.targets = ['adoption_probability', 'market_impact_score', 'risk_score', 'trend_score']
        
        # Required for BaseModel
        self.feature_names = []
        self.target_name = ''
        
    def train_all_models(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train all individual models"""
        logger.info("Training all ensemble models...")
        
        results = {}
        
        try:
            # Prepare data for different models
            trend_data = self._prepare_trend_data(data)
            adoption_data = self._prepare_adoption_data(data)
            classification_data = self._prepare_classification_data(data)
            sentiment_data = self._prepare_sentiment_data(data)
            
            # Train trend predictor
            if len(trend_data) > 30:  # Need sufficient data for LSTM
                logger.info("Training trend predictor...")
                trend_results = self.trend_predictor.train(
                    trend_data.drop('trend_score', axis=1),
                    trend_data['trend_score'],
                    **kwargs
                )
                results['trend_predictor'] = trend_results
            else:
                logger.warning("Insufficient data for trend predictor training")
            
            # Train adoption forecaster
            if len(adoption_data) > 10:
                logger.info("Training adoption forecaster...")
                adoption_results = self.adoption_forecaster.train(
                    adoption_data.drop(self.targets, axis=1, errors='ignore'),
                    adoption_data[self.targets].fillna(0.5),
                    **kwargs
                )
                results['adoption_forecaster'] = adoption_results
            else:
                logger.warning("Insufficient data for adoption forecaster training")
            
            # Train technology classifier
            if len(classification_data) > 5:
                logger.info("Training technology classifier...")
                classification_results = self.technology_classifier.train(
                    classification_data.drop('category', axis=1, errors='ignore'),
                    classification_data['category'],
                    **kwargs
                )
                results['technology_classifier'] = classification_results
            else:
                logger.warning("Insufficient data for technology classifier training")
            
            # Initialize sentiment analyzer
            logger.info("Initializing sentiment analyzer...")
            sentiment_results = self.sentiment_analyzer.train(sentiment_data, **kwargs)
            results['sentiment_analyzer'] = sentiment_results
            
            # Update ensemble metadata
            self.model_metadata.update({
                'last_trained': datetime.utcnow(),
                'models_trained': list(results.keys()),
                'training_results': results
            })
            
            self.is_trained = True
            logger.info("All ensemble models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training ensemble models: {e}")
            raise
        
        return results
    
    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess features for ensemble model"""
        # This is a placeholder - the ensemble model doesn't directly preprocess features
        # It delegates to individual models
        return np.array([])
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the ensemble model"""
        return self.train_all_models(X, **kwargs)
    
    def _prepare_trend_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for trend prediction"""
        # Select relevant features for trend prediction
        trend_features = [
            'trend_score', 'github_stars', 'github_forks', 'github_issues',
            'arxiv_papers', 'patent_filings', 'job_postings', 'social_mentions',
            'momentum_score', 'adoption_score'
        ]
        
        # Ensure all features exist
        for feature in trend_features:
            if feature not in data.columns:
                data[feature] = 0.0
        
        return data[trend_features].copy()
    
    def _prepare_adoption_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for adoption forecasting"""
        # Select features for adoption prediction
        adoption_features = [
            'trend_score', 'github_stars', 'github_forks', 'github_issues',
            'arxiv_papers', 'patent_filings', 'job_postings', 'social_mentions',
            'momentum_score', 'adoption_score', 'category'
        ]
        
        # Add derived features
        result_data = data[adoption_features].copy()
        
        # Calculate derived features
        if 'github_stars' in result_data.columns:
            result_data['github_star_growth_rate'] = result_data['github_stars'].pct_change().fillna(0)
        
        if 'github_issues' in result_data.columns and 'github_stars' in result_data.columns:
            result_data['issue_resolution_rate'] = (
                result_data['github_stars'] / (result_data['github_issues'] + 1)
            ).fillna(0)
        
        # Add target columns if they don't exist
        for target in self.targets:
            if target not in result_data.columns:
                result_data[target] = 0.5
        
        return result_data
    
    def _prepare_classification_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for technology classification"""
        classification_features = ['name', 'description', 'keywords', 'category']
        
        # Ensure required columns exist
        for feature in classification_features:
            if feature not in data.columns:
                if feature == 'keywords':
                    data[feature] = [[] for _ in range(len(data))]
                else:
                    data[feature] = ''
        
        return data[classification_features].copy()
    
    def _prepare_sentiment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for sentiment analysis"""
        sentiment_features = ['description', 'name']
        
        # Ensure required columns exist
        for feature in sentiment_features:
            if feature not in data.columns:
                data[feature] = ''
        
        return data[sentiment_features].copy()
    
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before making predictions")
        
        results = {}
        
        try:
            # Get predictions from each model
            if self.trend_predictor.is_trained:
                trend_data = self._prepare_trend_data(X)
                trend_predictions = self.trend_predictor.predict(trend_data)
                results['trend_predictions'] = trend_predictions
            
            if self.adoption_forecaster.is_trained:
                adoption_data = self._prepare_adoption_data(X)
                adoption_predictions = self.adoption_forecaster.predict(adoption_data)
                results['adoption_predictions'] = adoption_predictions
            
            if self.technology_classifier.is_trained:
                classification_data = self._prepare_classification_data(X)
                classification_predictions = self.technology_classifier.predict(classification_data)
                results['classification_predictions'] = classification_predictions
            
            if self.sentiment_analyzer.is_trained:
                sentiment_data = self._prepare_sentiment_data(X)
                sentiment_predictions = self.sentiment_analyzer.predict(sentiment_data)
                results['sentiment_predictions'] = sentiment_predictions
            
            # Combine predictions using weighted ensemble
            ensemble_predictions = self._combine_predictions(results)
            results['ensemble_predictions'] = ensemble_predictions
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            raise
        
        return results
    
    def _combine_predictions(self, individual_predictions: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Combine predictions from individual models"""
        ensemble_predictions = {}
        
        # Initialize prediction arrays
        n_samples = len(next(iter(individual_predictions.values())))
        
        for target in self.targets:
            predictions = []
            weights = []
            
            # Collect predictions from each model
            if 'trend_predictions' in individual_predictions and target == 'trend_score':
                predictions.append(individual_predictions['trend_predictions'])
                weights.append(self.model_weights['trend_predictor'])
            
            if 'adoption_predictions' in individual_predictions and target in individual_predictions['adoption_predictions']:
                predictions.append(individual_predictions['adoption_predictions'][target])
                weights.append(self.model_weights['adoption_forecaster'])
            
            # Combine predictions using weighted average
            if predictions:
                weights = np.array(weights) / np.sum(weights)  # Normalize weights
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
                ensemble_predictions[target] = ensemble_pred
            else:
                # Default prediction if no models available
                ensemble_predictions[target] = np.full(n_samples, 0.5)
        
        return ensemble_predictions
    
    def predict_technology_future(self, technology_data: pd.DataFrame, days_ahead: int = 90) -> Dict[str, Any]:
        """Predict comprehensive future for a technology"""
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before making predictions")
        
        # Get current predictions
        current_predictions = self.predict(technology_data)
        
        # Get trend predictions if available
        future_trends = {}
        if self.trend_predictor.is_trained:
            trend_data = self._prepare_trend_data(technology_data)
            future_trends = self.trend_predictor.predict_future_trends(trend_data, days_ahead)
        
        # Combine results
        result = {
            'current_predictions': current_predictions['ensemble_predictions'],
            'future_trends': future_trends,
            'model_confidence': self._calculate_model_confidence(current_predictions),
            'recommendations': self._generate_recommendations(current_predictions['ensemble_predictions'])
        }
        
        return result
    
    def _calculate_model_confidence(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for ensemble predictions"""
        confidence = {}
        
        # Count available models
        available_models = len([k for k in predictions.keys() if k != 'ensemble_predictions'])
        confidence['model_availability'] = available_models / len(self.model_weights)
        
        # Calculate prediction consistency
        if 'ensemble_predictions' in predictions:
            ensemble_preds = predictions['ensemble_predictions']
            consistency_scores = []
            
            for target in self.targets:
                if target in ensemble_preds:
                    pred_values = ensemble_preds[target]
                    # Calculate variance as inverse of consistency
                    variance = np.var(pred_values)
                    consistency = 1.0 / (1.0 + variance)
                    consistency_scores.append(consistency)
            
            if consistency_scores:
                confidence['prediction_consistency'] = np.mean(consistency_scores)
            else:
                confidence['prediction_consistency'] = 0.5
        
        # Overall confidence
        confidence['overall'] = (confidence.get('model_availability', 0) + 
                               confidence.get('prediction_consistency', 0)) / 2
        
        return confidence
    
    def _generate_recommendations(self, predictions: Dict[str, np.ndarray]) -> List[str]:
        """Generate recommendations based on predictions"""
        recommendations = []
        
        # Get average predictions
        avg_predictions = {}
        for target, values in predictions.items():
            avg_predictions[target] = np.mean(values)
        
        # Generate recommendations
        adoption_prob = avg_predictions.get('adoption_probability', 0.5)
        market_impact = avg_predictions.get('market_impact_score', 0.5)
        risk_score = avg_predictions.get('risk_score', 0.5)
        trend_score = avg_predictions.get('trend_score', 0.5)
        
        if adoption_prob > 0.7:
            recommendations.append("High adoption probability - Consider early investment")
        elif adoption_prob < 0.3:
            recommendations.append("Low adoption probability - Monitor closely before investing")
        
        if market_impact > 0.7:
            recommendations.append("High market impact potential - Strategic importance")
        elif market_impact < 0.3:
            recommendations.append("Low market impact - Limited strategic value")
        
        if risk_score > 0.7:
            recommendations.append("High risk - Proceed with caution")
        elif risk_score < 0.3:
            recommendations.append("Low risk - Safe investment opportunity")
        
        if trend_score > 0.7:
            recommendations.append("Strong upward trend - Momentum building")
        elif trend_score < 0.3:
            recommendations.append("Declining trend - Consider alternatives")
        
        if not recommendations:
            recommendations.append("Moderate potential - Continue monitoring")
        
        return recommendations
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models"""
        summary = {
            'ensemble_status': self.is_trained,
            'models_available': {},
            'overall_performance': {}
        }
        
        # Check individual model status
        models = {
            'trend_predictor': self.trend_predictor,
            'adoption_forecaster': self.adoption_forecaster,
            'technology_classifier': self.technology_classifier,
            'sentiment_analyzer': self.sentiment_analyzer
        }
        
        for name, model in models.items():
            summary['models_available'][name] = {
                'is_trained': model.is_trained,
                'model_type': model.model_name,
                'last_trained': model.model_metadata.get('last_trained'),
                'performance': model.model_metadata.get('performance_metrics', {})
            }
        
        # Calculate overall performance
        trained_models = sum(1 for model in models.values() if model.is_trained)
        summary['overall_performance'] = {
            'trained_models': trained_models,
            'total_models': len(models),
            'training_coverage': trained_models / len(models)
        }
        
        return summary
    
    def plot_ensemble_performance(self, save_path: Optional[str] = None):
        """Plot ensemble model performance"""
        summary = self.get_model_performance_summary()
        
        # Create performance visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model availability
        models = list(summary['models_available'].keys())
        availability = [summary['models_available'][m]['is_trained'] for m in models]
        colors = ['green' if avail else 'red' for avail in availability]
        
        ax1.bar(models, availability, color=colors)
        ax1.set_title('Model Training Status')
        ax1.set_ylabel('Trained (1) / Not Trained (0)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Model weights
        weights = list(self.model_weights.values())
        ax2.pie(weights, labels=models, autopct='%1.1f%%')
        ax2.set_title('Ensemble Model Weights')
        
        # Performance metrics (if available)
        performance_data = []
        for model_name in models:
            perf = summary['models_available'][model_name]['performance']
            if 'r2' in perf:
                performance_data.append(perf['r2'])
            elif 'accuracy' in perf:
                performance_data.append(perf['accuracy'])
            else:
                performance_data.append(0.0)
        
        ax3.bar(models, performance_data, color='skyblue')
        ax3.set_title('Model Performance (R²/Accuracy)')
        ax3.set_ylabel('Performance Score')
        ax3.tick_params(axis='x', rotation=45)
        
        # Training coverage
        coverage = summary['overall_performance']['training_coverage']
        ax4.pie([coverage, 1-coverage], labels=['Trained', 'Not Trained'], autopct='%1.1f%%')
        ax4.set_title('Overall Training Coverage')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Ensemble performance plot saved to {save_path}")
        
        plt.show() 