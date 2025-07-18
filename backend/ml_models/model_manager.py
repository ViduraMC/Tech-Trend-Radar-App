"""
Model Manager for coordinating all ML models in Tech Trend Radar
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import os
import joblib
from sqlalchemy.orm import Session

from .ensemble_model import EnsembleModel
from .trend_predictor import TrendPredictor
from .adoption_forecaster import AdoptionForecaster
from .technology_classifier import TechnologyClassifier
from .sentiment_analyzer import SentimentAnalyzer
from models.database import Technology, TrendData, Prediction

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages all ML models for Tech Trend Radar"""
    
    def __init__(self, model_dir: str = "ml_models"):
        self.model_dir = model_dir
        self.ensemble_model = EnsembleModel()
        self.is_initialized = False
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
    def initialize_models(self, db: Session) -> Dict[str, Any]:
        """Initialize and train all models with database data"""
        logger.info("Initializing ML models...")
        
        try:
            # Load data from database
            data = self._load_training_data(db)
            
            if data.empty:
                logger.warning("No training data available")
                return {'status': 'no_data', 'message': 'No training data available'}
            
            # Train ensemble model
            training_results = self.ensemble_model.train_all_models(data)
            
            # Save models
            self._save_models()
            
            self.is_initialized = True
            
            logger.info("ML models initialized successfully")
            return {
                'status': 'success',
                'training_results': training_results,
                'data_samples': len(data)
            }
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _load_training_data(self, db: Session) -> pd.DataFrame:
        """Load training data from database"""
        logger.info("Loading training data from database...")
        
        # Get technologies with their trend data
        technologies = db.query(Technology).all()
        
        training_data = []
        
        for tech in technologies:
            # Get latest trend data for this technology
            latest_trend = db.query(TrendData).filter(
                TrendData.technology_id == tech.id
            ).order_by(TrendData.date.desc()).first()
            
            if latest_trend:
                # Create training record
                record = {
                    'id': tech.id,
                    'name': tech.name,
                    'category': tech.category,
                    'description': tech.description or '',
                    'keywords': tech.keywords or [],
                    'first_detected': tech.first_detected,
                    'last_updated': tech.last_updated,
                    'trend_score': latest_trend.trend_score,
                    'github_stars': latest_trend.github_stars,
                    'github_forks': latest_trend.github_forks,
                    'github_issues': latest_trend.github_issues,
                    'arxiv_papers': latest_trend.arxiv_papers,
                    'patent_filings': latest_trend.patent_filings,
                    'job_postings': latest_trend.job_postings,
                    'social_mentions': latest_trend.social_mentions,
                    'momentum_score': latest_trend.momentum_score,
                    'adoption_score': latest_trend.adoption_score
                }
                
                # Get predictions for this technology
                predictions = db.query(Prediction).filter(
                    Prediction.technology_id == tech.id
                ).order_by(Prediction.prediction_date.desc()).first()
                
                if predictions:
                    record.update({
                        'adoption_probability': predictions.adoption_probability,
                        'market_impact_score': predictions.market_impact_score,
                        'risk_score': predictions.risk_score
                    })
                else:
                    # Default values if no predictions exist
                    record.update({
                        'adoption_probability': 0.5,
                        'market_impact_score': 0.5,
                        'risk_score': 0.5
                    })
                
                training_data.append(record)
        
        if not training_data:
            logger.warning("No training data found")
            return pd.DataFrame()
        
        df = pd.DataFrame(training_data)
        logger.info(f"Loaded {len(df)} training samples")
        
        return df
    
    def _save_models(self):
        """Save all trained models"""
        logger.info("Saving trained models...")
        
        try:
            # Save ensemble model
            self.ensemble_model.save_model()
            
            # Save individual models
            self.ensemble_model.trend_predictor.save_model()
            self.ensemble_model.adoption_forecaster.save_model()
            self.ensemble_model.technology_classifier.save_model()
            self.ensemble_model.sentiment_analyzer.save_model()
            
            logger.info("All models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_models(self) -> bool:
        """Load previously trained models"""
        logger.info("Loading trained models...")
        
        try:
            # Try to load ensemble model
            if self.ensemble_model.load_model():
                self.is_initialized = True
                logger.info("Models loaded successfully")
                return True
            else:
                logger.warning("No saved models found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict_technology(self, technology_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions for a single technology"""
        if not self.is_initialized:
            raise ValueError("Models must be initialized before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame([technology_data])
        
        # Make ensemble predictions
        predictions = self.ensemble_model.predict(df)
        
        # Format results
        result = {
            'technology_name': technology_data.get('name', 'Unknown'),
            'predictions': {},
            'confidence': {},
            'recommendations': []
        }
        
        if 'ensemble_predictions' in predictions:
            ensemble_preds = predictions['ensemble_predictions']
            for target, values in ensemble_preds.items():
                result['predictions'][target] = float(values[0]) if len(values) > 0 else 0.5
        
        # Calculate confidence
        if 'ensemble_predictions' in predictions:
            confidence = self.ensemble_model._calculate_model_confidence(predictions)
            result['confidence'] = confidence
        
        # Generate recommendations
        if 'ensemble_predictions' in predictions:
            recommendations = self.ensemble_model._generate_recommendations(
                predictions['ensemble_predictions']
            )
            result['recommendations'] = recommendations
        
        return result
    
    def predict_multiple_technologies(self, technologies_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions for multiple technologies"""
        if not self.is_initialized:
            raise ValueError("Models must be initialized before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame(technologies_data)
        
        # Make ensemble predictions
        predictions = self.ensemble_model.predict(df)
        
        # Format results
        results = []
        
        if 'ensemble_predictions' in predictions:
            ensemble_preds = predictions['ensemble_predictions']
            
            for i, tech_data in enumerate(technologies_data):
                result = {
                    'technology_name': tech_data.get('name', 'Unknown'),
                    'predictions': {},
                    'confidence': {},
                    'recommendations': []
                }
                
                # Add predictions
                for target, values in ensemble_preds.items():
                    result['predictions'][target] = float(values[i]) if i < len(values) else 0.5
                
                # Calculate confidence
                confidence = self.ensemble_model._calculate_model_confidence(predictions)
                result['confidence'] = confidence
                
                # Generate recommendations for this technology
                tech_predictions = {target: [values[i]] if i < len(values) else [0.5] 
                                  for target, values in ensemble_preds.items()}
                recommendations = self.ensemble_model._generate_recommendations(tech_predictions)
                result['recommendations'] = recommendations
                
                results.append(result)
        
        return results
    
    def predict_future_trends(self, technology_data: Dict[str, Any], days_ahead: int = 90) -> Dict[str, Any]:
        """Predict future trends for a technology"""
        if not self.is_initialized:
            raise ValueError("Models must be initialized before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame([technology_data])
        
        # Get future predictions
        future_predictions = self.ensemble_model.predict_technology_future(df, days_ahead)
        
        return future_predictions
    
    def classify_technology(self, name: str, description: str = "", keywords: List[str] = None) -> Dict[str, Any]:
        """Classify a technology into a category"""
        if not self.is_initialized:
            raise ValueError("Models must be initialized before making predictions")
        
        return self.ensemble_model.technology_classifier.classify_technology(name, description, keywords)
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for a list of texts"""
        if not self.is_initialized:
            raise ValueError("Models must be initialized before making predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame({'text': texts})
        
        # Analyze sentiment
        return self.ensemble_model.sentiment_analyzer.predict(df)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        if not self.is_initialized:
            return {
                'initialized': False,
                'message': 'Models not initialized'
            }
        
        # Get ensemble model summary
        summary = self.ensemble_model.get_model_performance_summary()
        
        return {
            'initialized': True,
            'ensemble_status': summary['ensemble_status'],
            'models_available': summary['models_available'],
            'overall_performance': summary['overall_performance'],
            'last_trained': self.ensemble_model.model_metadata.get('last_trained')
        }
    
    def retrain_models(self, db: Session) -> Dict[str, Any]:
        """Retrain all models with fresh data"""
        logger.info("Retraining all models...")
        
        try:
            # Load fresh data
            data = self._load_training_data(db)
            
            if data.empty:
                return {'status': 'error', 'message': 'No training data available'}
            
            # Retrain ensemble model
            training_results = self.ensemble_model.train_all_models(data)
            
            # Save updated models
            self._save_models()
            
            logger.info("Models retrained successfully")
            
            return {
                'status': 'success',
                'training_results': training_results,
                'data_samples': len(data)
            }
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_feature_importance(self, model_name: str = None) -> Dict[str, Any]:
        """Get feature importance for models"""
        if not self.is_initialized:
            return {'error': 'Models not initialized'}
        
        importance_data = {}
        
        if model_name is None or model_name == 'adoption_forecaster':
            importance_data['adoption_forecaster'] = self.ensemble_model.adoption_forecaster.get_feature_importance()
        
        if model_name is None or model_name == 'technology_classifier':
            # Technology classifier doesn't have traditional feature importance
            importance_data['technology_classifier'] = {}
        
        return importance_data
    
    def plot_model_performance(self, save_path: Optional[str] = None):
        """Plot model performance"""
        if not self.is_initialized:
            logger.warning("Models not initialized")
            return
        
        self.ensemble_model.plot_ensemble_performance(save_path)
    
    def update_prediction(self, db: Session, technology_id: int, prediction_data: Dict[str, Any]) -> bool:
        """Update prediction in database with ML model results"""
        try:
            # Create or update prediction record
            existing_prediction = db.query(Prediction).filter(
                Prediction.technology_id == technology_id
            ).order_by(Prediction.prediction_date.desc()).first()
            
            if existing_prediction:
                # Update existing prediction
                existing_prediction.adoption_probability = prediction_data.get('adoption_probability', 0.5)
                existing_prediction.market_impact_score = prediction_data.get('market_impact_score', 0.5)
                existing_prediction.risk_score = prediction_data.get('risk_score', 0.5)
                existing_prediction.confidence_interval = prediction_data.get('confidence', {}).get('overall', 0.5)
                existing_prediction.model_used = 'ensemble_ml_model'
                existing_prediction.prediction_reasoning = '; '.join(prediction_data.get('recommendations', []))
            else:
                # Create new prediction
                prediction = Prediction(
                    technology_id=technology_id,
                    target_date=datetime.utcnow(),
                    adoption_probability=prediction_data.get('adoption_probability', 0.5),
                    market_impact_score=prediction_data.get('market_impact_score', 0.5),
                    risk_score=prediction_data.get('risk_score', 0.5),
                    confidence_interval=prediction_data.get('confidence', {}).get('overall', 0.5),
                    model_used='ensemble_ml_model',
                    features_used=['trend_score', 'github_stars', 'github_forks', 'github_issues', 
                                 'arxiv_papers', 'patent_filings', 'job_postings', 'social_mentions'],
                    prediction_reasoning='; '.join(prediction_data.get('recommendations', []))
                )
                db.add(prediction)
            
            db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error updating prediction: {e}")
            db.rollback()
            return False 