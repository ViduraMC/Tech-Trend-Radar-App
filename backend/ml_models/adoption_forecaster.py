"""
Ensemble-based Adoption Forecaster for Technology Adoption Prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class AdoptionForecaster(BaseModel):
    """Ensemble model for predicting technology adoption probability and market impact"""
    
    def __init__(self):
        super().__init__("adoption_forecaster")
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Feature engineering
        self.feature_names = [
            'trend_score', 'github_stars', 'github_forks', 'github_issues',
            'arxiv_papers', 'patent_filings', 'job_postings', 'social_mentions',
            'momentum_score', 'adoption_score', 'days_since_first_detected',
            'category_encoded', 'github_star_growth_rate', 'issue_resolution_rate',
            'community_activity_score', 'market_demand_score'
        ]
        
        # Multiple targets
        self.target_names = ['adoption_probability', 'market_impact_score', 'risk_score']
        self.models = {}  # Separate model for each target
        
    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess features for adoption forecasting"""
        # Create a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Handle missing features
        missing_features = set(self.feature_names) - set(processed_data.columns)
        for feature in missing_features:
            if feature == 'category_encoded':
                processed_data[feature] = 0
            elif 'rate' in feature or 'score' in feature:
                processed_data[feature] = 0.0
            else:
                processed_data[feature] = 0
        
        # Encode categorical features
        if 'category' in processed_data.columns and 'category_encoded' not in processed_data.columns:
            le = LabelEncoder()
            processed_data['category_encoded'] = le.fit_transform(processed_data['category'])
            self.label_encoders['category'] = le
        
        # Calculate derived features
        processed_data = self._calculate_derived_features(processed_data)
        
        # Select features
        feature_data = processed_data[self.feature_names].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill').fillna(0)
        
        # Scale features
        scaled_data = self.scaler.fit_transform(feature_data)
        
        return scaled_data
    
    def _calculate_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features for better prediction"""
        # Days since first detected
        if 'first_detected' in data.columns:
            data['days_since_first_detected'] = (
                pd.to_datetime(data['first_detected']) - pd.to_datetime(data['first_detected'].min())
            ).dt.days
        else:
            data['days_since_first_detected'] = 0
        
        # GitHub star growth rate (if we have historical data)
        if 'github_stars' in data.columns:
            data['github_star_growth_rate'] = data['github_stars'].pct_change().fillna(0)
        
        # Issue resolution rate
        if 'github_issues' in data.columns and 'github_stars' in data.columns:
            data['issue_resolution_rate'] = (
                data['github_stars'] / (data['github_issues'] + 1)
            ).fillna(0)
        else:
            data['issue_resolution_rate'] = 0
        
        # Community activity score
        activity_features = ['github_stars', 'github_forks', 'social_mentions']
        available_features = [f for f in activity_features if f in data.columns]
        if available_features:
            data['community_activity_score'] = data[available_features].mean(axis=1)
        else:
            data['community_activity_score'] = 0
        
        # Market demand score (based on job postings and patents)
        demand_features = ['job_postings', 'patent_filings']
        available_demand = [f for f in demand_features if f in data.columns]
        if available_demand:
            data['market_demand_score'] = data[available_demand].mean(axis=1)
        else:
            data['market_demand_score'] = 0
        
        return data
    
    def build_ensemble_model(self, target_name: str):
        """Build ensemble model for a specific target"""
        # Base models
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        ridge = Ridge(alpha=1.0, random_state=42)
        
        # Create ensemble
        ensemble = VotingRegressor([
            ('rf', rf),
            ('gb', gb),
            ('xgb', xgb_model),
            ('ridge', ridge)
        ], weights=[0.3, 0.3, 0.3, 0.1])
        
        return ensemble
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train ensemble models for all targets"""
        logger.info("Starting adoption forecaster training...")
        
        # Preprocess features
        feature_data = self.preprocess_features(X)
        
        # Ensure y has the required columns
        if isinstance(y, pd.Series):
            # If y is a single series, assume it's adoption_probability
            y = pd.DataFrame({self.target_names[0]: y})
        
        missing_targets = set(self.target_names) - set(y.columns)
        for target in missing_targets:
            y[target] = 0.5  # Default value
        
        # Train separate model for each target
        results = {}
        
        for target_name in self.target_names:
            if target_name not in y.columns:
                logger.warning(f"Target {target_name} not found in training data")
                continue
            
            logger.info(f"Training model for {target_name}...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                feature_data, y[target_name], test_size=0.2, random_state=42
            )
            
            # Build and train model
            model = self.build_ensemble_model(target_name)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            # Store model and results
            self.models[target_name] = {
                'model': model,
                'metrics': metrics,
                'feature_importance': self._get_feature_importance(model)
            }
            
            results[target_name] = metrics
            logger.info(f"{target_name} - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")
        
        # Update metadata
        self.model_metadata.update({
            'last_trained': datetime.utcnow(),
            'training_samples': len(feature_data),
            'targets_trained': list(self.models.keys()),
            'performance_metrics': results
        })
        
        self.is_trained = True
        logger.info("Adoption forecaster training completed")
        return results
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions for all targets"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess features
        feature_data = self.preprocess_features(X)
        
        # Make predictions for each target
        predictions = {}
        
        for target_name, model_info in self.models.items():
            model = model_info['model']
            pred = model.predict(feature_data)
            
            # Ensure predictions are within valid range
            if target_name == 'risk_score':
                pred = np.clip(pred, 0.0, 1.0)
            else:
                pred = np.clip(pred, 0.0, 1.0)
            
            predictions[target_name] = pred
        
        return predictions
    
    def predict_adoption_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Predict adoption probability specifically"""
        predictions = self.predict(X)
        return predictions.get('adoption_probability', np.zeros(len(X)))
    
    def predict_market_impact(self, X: pd.DataFrame) -> np.ndarray:
        """Predict market impact score specifically"""
        predictions = self.predict(X)
        return predictions.get('market_impact_score', np.zeros(len(X)))
    
    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk score specifically"""
        predictions = self.predict(X)
        return predictions.get('risk_score', np.zeros(len(X)))
    
    def _get_feature_importance(self, model) -> Dict[str, float]:
        """Get feature importance from ensemble model"""
        # Try to get feature importance from the first model that has it
        for name, estimator in model.named_estimators_.items():
            if hasattr(estimator, 'feature_importances_'):
                importance_dict = dict(zip(self.feature_names, estimator.feature_importances_))
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for all models"""
        importance_dict = {}
        
        for target_name, model_info in self.models.items():
            importance_dict[target_name] = model_info.get('feature_importance', {})
        
        return importance_dict
    
    def plot_feature_importance(self, target_name: str = None, top_n: int = 10, save_path: Optional[str] = None):
        """Plot feature importance for specific target or all targets"""
        importance_dict = self.get_feature_importance()
        
        if target_name and target_name in importance_dict:
            # Plot for specific target
            importance = importance_dict[target_name]
            if not importance:
                logger.warning(f"No feature importance available for {target_name}")
                return
            
            top_features = dict(list(importance.items())[:top_n])
            
            plt.figure(figsize=(10, 6))
            features = list(top_features.keys())
            scores = list(top_features.values())
            
            plt.barh(range(len(features)), scores)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance Score')
            plt.title(f'Top {top_n} Feature Importance - {target_name}')
            plt.gca().invert_yaxis()
            
        else:
            # Plot for all targets
            fig, axes = plt.subplots(len(importance_dict), 1, figsize=(12, 4*len(importance_dict)))
            if len(importance_dict) == 1:
                axes = [axes]
            
            for i, (target, importance) in enumerate(importance_dict.items()):
                if not importance:
                    continue
                
                top_features = dict(list(importance.items())[:top_n])
                features = list(top_features.keys())
                scores = list(top_features.values())
                
                axes[i].barh(range(len(features)), scores)
                axes[i].set_yticks(range(len(features)))
                axes[i].set_yticklabels(features)
                axes[i].set_xlabel('Importance Score')
                axes[i].set_title(f'Feature Importance - {target}')
                axes[i].invert_yaxis()
            
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all models"""
        performance = {}
        
        for target_name, model_info in self.models.items():
            performance[target_name] = model_info.get('metrics', {})
        
        return performance
    
    def plot_performance_comparison(self, save_path: Optional[str] = None):
        """Plot performance comparison across all models"""
        performance = self.get_model_performance()
        
        if not performance:
            logger.warning("No performance data available")
            return
        
        metrics = ['r2', 'rmse', 'mae']
        targets = list(performance.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = [performance[target].get(metric, 0) for target in targets]
            
            axes[i].bar(targets, values)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + max(values) * 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Performance comparison plot saved to {save_path}")
        
        plt.show() 