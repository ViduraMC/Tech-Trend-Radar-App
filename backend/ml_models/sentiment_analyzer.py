"""
Sentiment Analyzer for Technology Discussions and Social Media
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging
import re
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
# import torch
import matplotlib.pyplot as plt
import seaborn as sns

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class SentimentAnalyzer(BaseModel):
    """Sentiment analysis model for technology discussions"""
    
    def __init__(self, use_transformer: bool = True):
        super().__init__("sentiment_analyzer")
        self.use_transformer = use_transformer
        self.transformer_pipeline = None
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
        # Text preprocessing
        self.text_features = ['text', 'description', 'title', 'content']
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.!?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess features for sentiment analysis"""
        processed_data = data.copy()
        
        # Find text columns
        text_columns = [col for col in self.text_features if col in processed_data.columns]
        
        if not text_columns:
            raise ValueError("No text columns found for sentiment analysis")
        
        # Combine text from all available columns
        combined_text = []
        for _, row in processed_data.iterrows():
            text_parts = []
            for col in text_columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    text_parts.append(str(row[col]))
            
            combined_text.append(' '.join(text_parts))
        
        # Preprocess text
        processed_text = [self.preprocess_text(text) for text in combined_text]
        
        return np.array(processed_text)
    
    def build_transformer_pipeline(self):
        """Build transformer-based sentiment analysis pipeline"""
        try:
            # For now, just use TextBlob
            logger.info("Transformer models not available, using TextBlob")
            return False
            
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}")
            logger.info("Falling back to TextBlob")
            return False
    
    def analyze_sentiment_textblob(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        
        # Get polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Convert polarity to sentiment label
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Convert polarity to confidence (0 to 1)
        confidence = abs(polarity)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    def train(self, X: pd.DataFrame, y: pd.Series = None, **kwargs) -> Dict[str, Any]:
        """Initialize sentiment analyzer (no training needed for pre-trained models)"""
        logger.info("Initializing sentiment analyzer...")
        
        if self.use_transformer:
            success = self.build_transformer_pipeline()
            if success:
                logger.info("Transformer-based sentiment analyzer loaded successfully")
            else:
                logger.info("Using TextBlob for sentiment analysis")
                self.use_transformer = False
        
        # Update metadata
        self.model_metadata.update({
            'last_trained': datetime.utcnow(),
            'model_type': 'transformer' if self.use_transformer else 'textblob',
            'sentiment_labels': self.sentiment_labels
        })
        
        self.is_trained = True
        logger.info("Sentiment analyzer initialization completed")
        
        return {'status': 'initialized', 'model_type': self.model_metadata['model_type']}
    
    def predict(self, X: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze sentiment for input texts"""
        if not self.is_trained:
            raise ValueError("Sentiment analyzer must be initialized before use")
        
        # Preprocess text
        text_data = self.preprocess_features(X)
        
        results = []
        
        for text in text_data:
            if not text.strip():
                # Empty text gets neutral sentiment
                result = {
                    'sentiment': 'neutral',
                    'confidence': 0.0,
                    'polarity': 0.0,
                    'subjectivity': 0.0
                }
            elif self.use_transformer and self.transformer_pipeline:
                # Use transformer model
                try:
                    analysis = self.transformer_pipeline(text[:512])  # Limit text length
                    result = {
                        'sentiment': analysis[0]['label'].lower(),
                        'confidence': analysis[0]['score'],
                        'polarity': self._sentiment_to_polarity(analysis[0]['label']),
                        'subjectivity': 0.5  # Default for transformer models
                    }
                except Exception as e:
                    logger.warning(f"Transformer analysis failed: {e}")
                    result = self.analyze_sentiment_textblob(text)
            else:
                # Use TextBlob
                result = self.analyze_sentiment_textblob(text)
            
            results.append(result)
        
        return results
    
    def _sentiment_to_polarity(self, sentiment_label: str) -> float:
        """Convert sentiment label to polarity score"""
        label_mapping = {
            'positive': 0.5,
            'neutral': 0.0,
            'negative': -0.5,
            'LABEL_2': 0.5,  # Positive
            'LABEL_1': 0.0,  # Neutral
            'LABEL_0': -0.5  # Negative
        }
        
        return label_mapping.get(sentiment_label.lower(), 0.0)
    
    def analyze_technology_sentiment(self, technology_name: str, texts: List[str]) -> Dict[str, Any]:
        """Analyze sentiment for a specific technology"""
        if not self.is_trained:
            raise ValueError("Sentiment analyzer must be initialized before use")
        
        # Create input data
        input_data = pd.DataFrame({'text': texts})
        
        # Analyze sentiment
        sentiment_results = self.predict(input_data)
        
        # Aggregate results
        sentiments = [result['sentiment'] for result in sentiment_results]
        confidences = [result['confidence'] for result in sentiment_results]
        polarities = [result['polarity'] for result in sentiment_results]
        
        # Calculate statistics
        sentiment_counts = pd.Series(sentiments).value_counts()
        avg_confidence = np.mean(confidences)
        avg_polarity = np.mean(polarities)
        
        # Determine overall sentiment
        if sentiment_counts.get('positive', 0) > sentiment_counts.get('negative', 0):
            overall_sentiment = 'positive'
        elif sentiment_counts.get('negative', 0) > sentiment_counts.get('positive', 0):
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'technology_name': technology_name,
            'overall_sentiment': overall_sentiment,
            'sentiment_distribution': sentiment_counts.to_dict(),
            'average_confidence': avg_confidence,
            'average_polarity': avg_polarity,
            'total_mentions': len(texts),
            'detailed_results': sentiment_results
        }
    
    def analyze_social_media_sentiment(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze sentiment for social media data"""
        if not self.is_trained:
            raise ValueError("Sentiment analyzer must be initialized before use")
        
        # Analyze sentiment
        sentiment_results = self.predict(data)
        
        # Add results to dataframe
        result_df = data.copy()
        result_df['sentiment'] = [result['sentiment'] for result in sentiment_results]
        result_df['sentiment_confidence'] = [result['confidence'] for result in sentiment_results]
        result_df['sentiment_polarity'] = [result['polarity'] for result in sentiment_results]
        result_df['sentiment_subjectivity'] = [result['subjectivity'] for result in sentiment_results]
        
        return result_df
    
    def get_sentiment_trends(self, data: pd.DataFrame, date_column: str = 'date') -> Dict[str, Any]:
        """Analyze sentiment trends over time"""
        if not self.is_trained:
            raise ValueError("Sentiment analyzer must be initialized before use")
        
        # Ensure date column exists
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found in data")
        
        # Analyze sentiment
        sentiment_results = self.predict(data)
        
        # Add sentiment to dataframe
        result_df = data.copy()
        result_df['sentiment'] = [result['sentiment'] for result in sentiment_results]
        result_df['polarity'] = [result['polarity'] for result in sentiment_results]
        
        # Convert date column to datetime
        result_df[date_column] = pd.to_datetime(result_df[date_column])
        
        # Group by date and calculate sentiment statistics
        daily_sentiment = result_df.groupby(result_df[date_column].dt.date).agg({
            'sentiment': lambda x: x.value_counts().index[0] if len(x) > 0 else 'neutral',
            'polarity': 'mean',
            'sentiment': 'count'
        }).rename(columns={'sentiment': 'mention_count'})
        
        # Calculate sentiment distribution over time
        sentiment_trends = result_df.groupby([result_df[date_column].dt.date, 'sentiment']).size().unstack(fill_value=0)
        
        return {
            'daily_sentiment': daily_sentiment.to_dict('index'),
            'sentiment_trends': sentiment_trends.to_dict(),
            'overall_statistics': {
                'total_mentions': len(result_df),
                'positive_ratio': (result_df['sentiment'] == 'positive').mean(),
                'negative_ratio': (result_df['sentiment'] == 'negative').mean(),
                'neutral_ratio': (result_df['sentiment'] == 'neutral').mean(),
                'average_polarity': result_df['polarity'].mean()
            }
        }
    
    def plot_sentiment_distribution(self, sentiment_results: List[Dict[str, Any]], save_path: Optional[str] = None):
        """Plot sentiment distribution"""
        sentiments = [result['sentiment'] for result in sentiment_results]
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        plt.figure(figsize=(10, 6))
        
        # Create pie chart
        plt.subplot(1, 2, 1)
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', colors=colors)
        plt.title('Sentiment Distribution')
        
        # Create bar chart
        plt.subplot(1, 2, 2)
        sentiment_counts.plot(kind='bar', color=colors)
        plt.title('Sentiment Counts')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Sentiment distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_sentiment_trends(self, data: pd.DataFrame, date_column: str = 'date', save_path: Optional[str] = None):
        """Plot sentiment trends over time"""
        if not self.is_trained:
            logger.warning("Sentiment analyzer must be initialized before plotting")
            return
        
        # Get sentiment trends
        trends = self.get_sentiment_trends(data, date_column)
        
        # Create time series plot
        daily_data = pd.DataFrame.from_dict(trends['daily_sentiment'], orient='index')
        daily_data.index = pd.to_datetime(daily_data.index)
        
        plt.figure(figsize=(15, 8))
        
        # Plot polarity over time
        plt.subplot(2, 1, 1)
        plt.plot(daily_data.index, daily_data['polarity'], marker='o', linewidth=2)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.title('Sentiment Polarity Over Time')
        plt.ylabel('Polarity Score')
        plt.grid(True, alpha=0.3)
        
        # Plot mention count over time
        plt.subplot(2, 1, 2)
        plt.bar(daily_data.index, daily_data['mention_count'], alpha=0.7)
        plt.title('Mention Count Over Time')
        plt.ylabel('Number of Mentions')
        plt.xlabel('Date')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Sentiment trends plot saved to {save_path}")
        
        plt.show() 