"""
LSTM-based Trend Predictor for Technology Trend Scores
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
import joblib

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class TrendPredictor(BaseModel):
    """Ensemble-based model for predicting technology trend scores"""
    
    def __init__(self, sequence_length: int = 30, prediction_horizon: int = 7):
        super().__init__("trend_predictor")
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_names = [
            'trend_score', 'github_stars', 'github_forks', 'github_issues',
            'arxiv_papers', 'patent_filings', 'job_postings', 'social_mentions',
            'momentum_score', 'adoption_score'
        ]
        self.target_name = 'trend_score'
        
    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess features for LSTM input"""
        # Ensure we have the required features
        missing_features = set(self.feature_names) - set(data.columns)
        if missing_features:
            # Fill missing features with zeros
            for feature in missing_features:
                data[feature] = 0.0
        
        # Select only the features we need
        feature_data = data[self.feature_names].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill').fillna(0)
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(feature_data)
        
        return scaled_data
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.prediction_horizon + 1):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i:i+self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]) -> RandomForestRegressor:
        """Build ensemble model architecture"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        return model
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the ensemble trend predictor"""
        logger.info("Starting ensemble trend predictor training...")
        
        # Preprocess features
        feature_data = self.preprocess_features(X)
        target_data = y.values
        
        # Split data
        split_idx = int(0.8 * len(feature_data))
        X_train, X_test = feature_data[:split_idx], feature_data[split_idx:]
        y_train, y_test = target_data[:split_idx], target_data[split_idx:]
        
        if len(X_train) < 5:
            raise ValueError("Insufficient data for training. Need at least 5 samples.")
        
        # Build and train model
        self.model = self.build_model((X_train.shape[1], X_train.shape[1]))
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        # Update metadata
        self.model_metadata.update({
            'last_trained': datetime.utcnow(),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'performance_metrics': metrics
        })
        
        self.is_trained = True
        
        logger.info(f"Ensemble training completed. Test RMSE: {metrics['rmse']:.4f}")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained ensemble model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess features
        feature_data = self.preprocess_features(X)
        
        # Make prediction
        prediction = self.model.predict(feature_data)
        
        return prediction
    
    def predict_future_trends(self, historical_data: pd.DataFrame, days_ahead: int = 30) -> Dict[str, Any]:
        """Predict future trend scores for multiple days"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess historical data
        feature_data = self.preprocess_features(historical_data)
        
        if len(feature_data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
        
        predictions = []
        current_sequence = feature_data[-self.sequence_length:].copy()
        
        # Predict step by step
        for _ in range(days_ahead):
            # Reshape for prediction
            sequence = current_sequence.reshape(1, self.sequence_length, -1)
            
            # Make prediction
            pred = self.model.predict(sequence, verbose=0)
            pred_original = self.target_scaler.inverse_transform(pred)
            
            predictions.append(pred_original[0, 0])
            
            # Update sequence for next prediction (shift and add predicted value)
            # This is a simplified approach - in practice, you'd need to update all features
            new_row = current_sequence[-1].copy()
            new_row[0] = pred[0, 0]  # Update trend_score with prediction
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Create dates for predictions
        last_date = historical_data.index[-1] if hasattr(historical_data.index[-1], 'date') else datetime.now()
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        return {
            'dates': future_dates,
            'predictions': predictions,
            'confidence_interval': self._calculate_confidence_interval(predictions)
        }
    
    def _calculate_confidence_interval(self, predictions: List[float], confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval for predictions"""
        predictions_array = np.array(predictions)
        mean_pred = np.mean(predictions_array)
        std_pred = np.std(predictions_array)
        
        # Simple confidence interval calculation
        z_score = 1.96  # 95% confidence
        margin_of_error = z_score * std_pred / np.sqrt(len(predictions))
        
        return {
            'lower_bound': mean_pred - margin_of_error,
            'upper_bound': mean_pred + margin_of_error,
            'mean': mean_pred,
            'std': std_pred
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        if not self.training_history:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.training_history['loss'], label='Training Loss')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MAE
        ax2.plot(self.training_history['mae'], label='Training MAE')
        ax2.plot(self.training_history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """Save the trained model with additional LSTM-specific data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            filepath = os.path.join(self.model_dir, f"{self.model_name}.h5")
        
        # Save Keras model
        self.model.save(filepath)
        
        # Save additional metadata
        metadata_file = filepath.replace('.h5', '_metadata.joblib')
        metadata = {
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'model_metadata': self.model_metadata,
            'is_trained': self.is_trained,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'training_history': self.training_history
        }
        
        joblib.dump(metadata, metadata_file)
        logger.info(f"LSTM model saved to {filepath}")
        return filepath 