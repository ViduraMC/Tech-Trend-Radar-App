"""
Base class for all machine learning models in Tech Trend Radar
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Base class for all ML models"""
    
    def __init__(self, model_name: str, model_dir: str = "models"):
        self.model_name = model_name
        self.model_dir = model_dir
        self.model = None
        self.feature_names = []
        self.target_name = ""
        self.scaler = None
        self.is_trained = False
        self.training_history = {}
        self.model_metadata = {
            "created_at": datetime.utcnow(),
            "last_trained": None,
            "version": "1.0.0",
            "performance_metrics": {}
        }
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
    
    @abstractmethod
    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess input features"""
        pass
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            filepath = os.path.join(self.model_dir, f"{self.model_name}.joblib")
        
        # Save model and metadata
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "scaler": self.scaler,
            "model_metadata": self.model_metadata,
            "is_trained": self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: Optional[str] = None) -> bool:
        """Load a trained model"""
        if filepath is None:
            filepath = os.path.join(self.model_dir, f"{self.model_name}.joblib")
        
        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return False
        
        try:
            model_data = joblib.load(filepath)
            self.model = model_data["model"]
            self.feature_names = model_data["feature_names"]
            self.target_name = model_data["target_name"]
            self.scaler = model_data["scaler"]
            self.model_metadata = model_data["model_metadata"]
            self.is_trained = model_data["is_trained"]
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }
        
        # Update metadata
        self.model_metadata["performance_metrics"] = metrics
        self.model_metadata["last_evaluated"] = datetime.utcnow()
        
        logger.info(f"Model evaluation completed: {metrics}")
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        if not self.is_trained:
            raise ValueError("Model must be trained before cross-validation")
        
        # For models that support cross_val_score
        if hasattr(self.model, 'predict'):
            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
            cv_metrics = {
                "cv_r2_mean": cv_scores.mean(),
                "cv_r2_std": cv_scores.std(),
                "cv_r2_scores": cv_scores.tolist()
            }
            
            logger.info(f"Cross-validation completed: {cv_metrics}")
            return cv_metrics
        
        return {}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available"""
        if not self.is_trained or self.model is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        elif hasattr(self.model, 'coef_'):
            importance_dict = dict(zip(self.feature_names, np.abs(self.model.coef_)))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def plot_feature_importance(self, top_n: int = 10, save_path: Optional[str] = None):
        """Plot feature importance"""
        importance = self.get_feature_importance()
        if not importance:
            logger.warning("Feature importance not available for this model")
            return
        
        # Get top N features
        top_features = dict(list(importance.items())[:top_n])
        
        plt.figure(figsize=(10, 6))
        features = list(top_features.keys())
        scores = list(top_features.values())
        
        plt.barh(range(len(features)), scores)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance - {self.model_name}')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "metadata": self.model_metadata,
            "feature_importance": self.get_feature_importance()
        }
    
    def _validate_input(self, X: pd.DataFrame) -> bool:
        """Validate input data"""
        if X.empty:
            raise ValueError("Input data is empty")
        
        if not self.is_trained:
            raise ValueError("Model is not trained")
        
        # Check if required features are present
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        return True 