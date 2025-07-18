# 🤖 Machine Learning Models for Tech Trend Radar

This directory contains the complete machine learning system for Tech Trend Radar, providing AI-powered predictions and insights for technology trends.

## 🏗️ Architecture Overview

The ML system consists of multiple specialized models that work together:

```
ml_models/
├── base_model.py           # Base class for all ML models
├── trend_predictor.py      # LSTM-based trend forecasting
├── adoption_forecaster.py  # Ensemble model for adoption prediction
├── technology_classifier.py # NLP-based technology categorization
├── sentiment_analyzer.py   # Sentiment analysis for discussions
├── ensemble_model.py       # Combines all models
├── model_manager.py        # Coordinates all models
└── README.md              # This file
```

## 🎯 Model Types

### 1. **Trend Predictor (LSTM)**
- **Purpose**: Predicts technology trend scores over time
- **Model**: Bidirectional LSTM with attention
- **Features**: GitHub metrics, social mentions, job postings, etc.
- **Output**: Future trend scores (next 7-30 days)

### 2. **Adoption Forecaster (Ensemble)**
- **Purpose**: Predicts technology adoption probability and market impact
- **Models**: Random Forest + XGBoost + Gradient Boosting + Ridge Regression
- **Features**: 16 engineered features including growth rates and community activity
- **Output**: Adoption probability, market impact, risk score

### 3. **Technology Classifier (NLP)**
- **Purpose**: Automatically categorizes technologies
- **Model**: TF-IDF + Random Forest (with transformer fallback)
- **Features**: Technology name, description, keywords
- **Output**: Technology category (AI/ML, Cloud, Blockchain, etc.)

### 4. **Sentiment Analyzer (NLP)**
- **Purpose**: Analyzes sentiment in technology discussions
- **Model**: Transformer-based (with TextBlob fallback)
- **Features**: Social media posts, GitHub issues, discussions
- **Output**: Sentiment scores and trends

### 5. **Ensemble Model**
- **Purpose**: Combines all predictions for maximum reliability
- **Method**: Weighted averaging of individual model predictions
- **Weights**: Adoption Forecaster (40%), Trend Predictor (30%), Classifier (20%), Sentiment (10%)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
cd backend
python train_ml_models.py
```

### 3. Use in API
```python
from ml_models.model_manager import ModelManager

# Initialize
model_manager = ModelManager()
model_manager.initialize_models(db)

# Make predictions
prediction = model_manager.predict_technology(technology_data)
```

## 📊 Model Performance

### Expected Performance Metrics:
- **Trend Predictor**: RMSE < 0.15, R² > 0.7
- **Adoption Forecaster**: R² > 0.6 for adoption probability
- **Technology Classifier**: Accuracy > 85%
- **Sentiment Analyzer**: Accuracy > 80%

### Feature Importance (Top 5):
1. **GitHub Stars** - Community adoption indicator
2. **Trend Score** - Current momentum
3. **Job Postings** - Market demand
4. **Social Mentions** - Public interest
5. **Patent Filings** - Innovation activity

## 🔧 API Endpoints

### ML Model Management
```http
POST /api/predictions/ml/initialize    # Train models
GET  /api/predictions/ml/status        # Check model status
POST /api/predictions/ml/retrain       # Retrain models
POST /api/predictions/ml/predict       # Make ML prediction
GET  /api/predictions/ml/feature-importance  # Get feature importance
```

### Example Usage
```python
import requests

# Initialize models
response = requests.post('http://localhost:8000/api/predictions/ml/initialize')
print(response.json())

# Make prediction
tech_data = {
    'name': 'React',
    'category': 'Frontend',
    'trend_score': 0.85,
    'github_stars': 200000,
    'github_forks': 40000,
    'github_issues': 500,
    'social_mentions': 1500
}

response = requests.post('http://localhost:8000/api/predictions/ml/predict', json=tech_data)
prediction = response.json()
print(f"Adoption Probability: {prediction['predictions']['adoption_probability']:.2%}")
```

## 🎨 Model Visualization

### Training History
```python
# Plot training progress
model_manager.plot_model_performance()
```

### Feature Importance
```python
# Plot feature importance
model_manager.ensemble_model.adoption_forecaster.plot_feature_importance()
```

### Sentiment Trends
```python
# Plot sentiment over time
model_manager.ensemble_model.sentiment_analyzer.plot_sentiment_trends(data)
```

## 🔄 Model Lifecycle

### 1. **Training Phase**
- Load data from database
- Preprocess features
- Train individual models
- Validate performance
- Save trained models

### 2. **Prediction Phase**
- Load trained models
- Preprocess input data
- Make ensemble predictions
- Calculate confidence scores
- Generate recommendations

### 3. **Retraining Phase**
- Monitor model performance
- Collect new data
- Retrain when accuracy drops
- A/B test new models

## 📈 Data Requirements

### Minimum Data for Training:
- **Technologies**: 50+ technologies
- **Trend Data**: 30+ days of historical data
- **Features**: GitHub metrics, social mentions, job postings
- **Categories**: At least 3 technology categories

### Data Quality:
- **Completeness**: >80% feature coverage
- **Freshness**: Data updated within 24 hours
- **Accuracy**: Validated data sources

## 🛠️ Customization

### Adding New Models
```python
from .base_model import BaseModel

class CustomModel(BaseModel):
    def __init__(self):
        super().__init__("custom_model")
        # Your model initialization
    
    def train(self, X, y, **kwargs):
        # Your training logic
        pass
    
    def predict(self, X):
        # Your prediction logic
        pass
```

### Modifying Feature Engineering
```python
# In adoption_forecaster.py
def _calculate_derived_features(self, data):
    # Add your custom features
    data['custom_feature'] = your_calculation(data)
    return data
```

### Adjusting Model Weights
```python
# In ensemble_model.py
self.model_weights = {
    'trend_predictor': 0.3,      # Adjust these weights
    'adoption_forecaster': 0.4,
    'technology_classifier': 0.2,
    'sentiment_analyzer': 0.1
}
```

## 🔍 Model Interpretability

### Feature Importance
- **Random Forest**: Traditional feature importance
- **LSTM**: Attention weights
- **Ensemble**: Weighted feature importance

### Confidence Intervals
- **Prediction Confidence**: Based on model agreement
- **Model Availability**: Percentage of models available
- **Data Quality**: Based on feature completeness

### Recommendations
- **High Adoption**: "Consider early investment"
- **High Risk**: "Proceed with caution"
- **Strong Trend**: "Momentum building"
- **Low Impact**: "Limited strategic value"

## 🚨 Troubleshooting

### Common Issues:

1. **"No training data available"**
   - Ensure database has technologies and trend data
   - Run `python create_sample_data.py` first

2. **"ML models not initialized"**
   - Run `python train_ml_models.py`
   - Check model files exist in `ml_models/` directory

3. **"Insufficient data for training"**
   - Need at least 30 data points for LSTM
   - Need at least 10 data points for ensemble models

4. **"Transformer model failed to load"**
   - Falls back to TextBlob automatically
   - Check internet connection for model downloads

### Performance Optimization:
- **GPU Acceleration**: Set `CUDA_VISIBLE_DEVICES=0`
- **Batch Processing**: Use `predict_multiple_technologies()`
- **Caching**: Models are cached after first load

## 📚 Advanced Usage

### Custom Training
```python
# Train with custom parameters
training_results = model_manager.ensemble_model.train_all_models(
    data,
    epochs=200,
    batch_size=64,
    learning_rate=0.001
)
```

### Model Comparison
```python
# Compare model performances
performance = model_manager.get_model_performance_summary()
print(performance['overall_performance'])
```

### Future Predictions
```python
# Predict 90 days ahead
future = model_manager.predict_future_trends(tech_data, days_ahead=90)
print(f"Future trend: {future['future_trends']}")
```

## 🤝 Contributing

### Adding New Models:
1. Inherit from `BaseModel`
2. Implement required methods
3. Add to `EnsembleModel`
4. Update weights and documentation

### Improving Performance:
1. Add new features
2. Tune hyperparameters
3. Test with cross-validation
4. Update model weights

### Data Sources:
1. Add new data collectors
2. Update feature engineering
3. Validate data quality
4. Retrain models

---

**🎯 The ML system provides reliable, interpretable predictions for technology trends, helping users make informed decisions about technology investments and strategies.** 