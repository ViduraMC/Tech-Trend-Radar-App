"""
Technology Classifier using NLP and Machine Learning
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import torch

from .base_model import BaseModel

logger = logging.getLogger(__name__)

class TechnologyClassifier(BaseModel):
    """NLP-based model for classifying technologies into categories"""
    
    def __init__(self, use_transformer: bool = False):
        super().__init__("technology_classifier")
        self.use_transformer = use_transformer
        self.label_encoder = LabelEncoder()
        self.vectorizer = None
        self.classifier = None
        self.transformer_pipeline = None
        
        # Technology categories
        self.categories = [
            'AI/ML', 'Blockchain', 'Cloud', 'Data', 'DevOps', 'Frontend',
            'Backend', 'Mobile', 'IoT', 'Quantum', 'AR/VR', 'Cybersecurity'
        ]
        
        # Feature names for text processing
        self.text_features = ['name', 'description', 'keywords']
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for classification"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess features for technology classification"""
        processed_data = data.copy()
        
        # Combine text features
        text_columns = [col for col in self.text_features if col in processed_data.columns]
        
        if not text_columns:
            raise ValueError("No text columns found for classification")
        
        # Preprocess and combine text
        combined_text = []
        for _, row in processed_data.iterrows():
            text_parts = []
            for col in text_columns:
                if col == 'keywords' and isinstance(row[col], list):
                    text_parts.extend(row[col])
                else:
                    text_parts.append(str(row[col]))
            
            combined_text.append(' '.join(text_parts))
        
        # Preprocess text
        processed_text = [self.preprocess_text(text) for text in combined_text]
        
        return np.array(processed_text)
    
    def build_traditional_pipeline(self):
        """Build traditional ML pipeline with TF-IDF and classifier"""
        # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        # Classifier (ensemble of multiple models)
        classifiers = [
            ('nb', MultinomialNB(alpha=0.1)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]
        
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifiers[1][1])  # Use Random Forest by default
        ])
        
        return pipeline
    
    def build_transformer_pipeline(self):
        """Build transformer-based pipeline"""
        try:
            # For now, just return traditional pipeline
            logger.info("Transformer models not available, using traditional ML pipeline")
            return self.build_traditional_pipeline()
            
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}")
            logger.info("Falling back to traditional ML pipeline")
            return self.build_traditional_pipeline()
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """Train the technology classifier"""
        logger.info("Starting technology classifier training...")
        
        # Preprocess text features
        text_data = self.preprocess_features(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            text_data, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Build pipeline
        if self.use_transformer:
            self.transformer_pipeline = self.build_transformer_pipeline()
        else:
            self.classifier = self.build_traditional_pipeline()
        
        # Train model
        if self.use_transformer and self.transformer_pipeline:
            # For transformer models, we need to handle training differently
            logger.info("Training transformer model...")
            # This is a simplified approach - in practice, you'd need more sophisticated training
            self.is_trained = True
        else:
            logger.info("Training traditional ML model...")
            self.classifier.fit(X_train, y_train)
            self.is_trained = True
        
        # Evaluate model
        if self.is_trained:
            y_pred = self.predict(X_test)
            y_pred_encoded = self.label_encoder.transform(y_pred)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred_encoded)
            report = classification_report(
                y_test, y_pred_encoded, 
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.classifier if not self.use_transformer else None,
                X_train, y_train, cv=5, scoring='accuracy'
            )
            
            metrics = {
                'accuracy': accuracy,
                'cv_accuracy_mean': cv_scores.mean() if not self.use_transformer else 0,
                'cv_accuracy_std': cv_scores.std() if not self.use_transformer else 0,
                'classification_report': report
            }
            
            # Update metadata
            self.model_metadata.update({
                'last_trained': datetime.utcnow(),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'categories': self.categories,
                'performance_metrics': metrics
            })
            
            logger.info(f"Classifier training completed. Accuracy: {accuracy:.4f}")
            return metrics
        
        return {}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict technology categories"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess text features
        text_data = self.preprocess_features(X)
        
        # Make predictions
        if self.use_transformer and self.transformer_pipeline:
            # Transformer predictions
            predictions = []
            for text in text_data:
                if text.strip():
                    result = self.transformer_pipeline(text)
                    predictions.append(result[0]['label'])
                else:
                    predictions.append(self.categories[0])  # Default category
            return np.array(predictions)
        else:
            # Traditional ML predictions
            predictions_encoded = self.classifier.predict(text_data)
            predictions = self.label_encoder.inverse_transform(predictions_encoded)
            return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability scores for each category"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess text features
        text_data = self.preprocess_features(X)
        
        # Get probability scores
        if self.use_transformer and self.transformer_pipeline:
            # Transformer doesn't provide probabilities in the same way
            # Return uniform probabilities as fallback
            n_samples = len(text_data)
            n_classes = len(self.categories)
            return np.ones((n_samples, n_classes)) / n_classes
        else:
            # Traditional ML probabilities
            return self.classifier.predict_proba(text_data)
    
    def classify_technology(self, name: str, description: str = "", keywords: List[str] = None) -> Dict[str, Any]:
        """Classify a single technology"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create input data
        input_data = pd.DataFrame([{
            'name': name,
            'description': description,
            'keywords': keywords or []
        }])
        
        # Make prediction
        prediction = self.predict(input_data)[0]
        probabilities = self.predict_proba(input_data)[0]
        
        # Create result
        result = {
            'technology_name': name,
            'predicted_category': prediction,
            'confidence': float(np.max(probabilities)),
            'category_probabilities': dict(zip(self.categories, probabilities.tolist()))
        }
        
        return result
    
    def get_category_keywords(self) -> Dict[str, List[str]]:
        """Get keywords associated with each category"""
        # This could be enhanced with actual keyword extraction from training data
        category_keywords = {
            'AI/ML': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'nlp', 'computer vision'],
            'Blockchain': ['blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'smart contracts', 'defi', 'nft'],
            'Cloud': ['cloud computing', 'aws', 'azure', 'google cloud', 'kubernetes', 'docker', 'serverless'],
            'Data': ['big data', 'data science', 'analytics', 'data engineering', 'apache spark', 'kafka'],
            'DevOps': ['devops', 'ci/cd', 'jenkins', 'github actions', 'terraform', 'ansible'],
            'Frontend': ['react', 'vue', 'angular', 'javascript', 'typescript', 'css', 'html'],
            'Backend': ['python', 'java', 'golang', 'rust', 'nodejs', 'api', 'database'],
            'Mobile': ['mobile development', 'ios', 'android', 'react native', 'flutter'],
            'IoT': ['internet of things', 'iot', 'embedded systems', 'sensors', 'edge computing'],
            'Quantum': ['quantum computing', 'quantum algorithms', 'qubits', 'quantum cryptography'],
            'AR/VR': ['augmented reality', 'virtual reality', 'mixed reality', 'unity', 'unreal engine'],
            'Cybersecurity': ['cybersecurity', 'security', 'encryption', 'authentication', 'vulnerability']
        }
        
        return category_keywords
    
    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series, save_path: Optional[str] = None):
        """Plot confusion matrix"""
        if not self.is_trained:
            logger.warning("Model must be trained before plotting confusion matrix")
            return
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title('Technology Classification Confusion Matrix')
        plt.xlabel('Predicted Category')
        plt.ylabel('Actual Category')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_category_distribution(self, predictions: np.ndarray, save_path: Optional[str] = None):
        """Plot distribution of predicted categories"""
        # Count predictions
        unique, counts = np.unique(predictions, return_counts=True)
        category_counts = dict(zip(unique, counts))
        
        # Ensure all categories are represented
        for category in self.categories:
            if category not in category_counts:
                category_counts[category] = 0
        
        # Sort by count
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Plot
        plt.figure(figsize=(12, 6))
        categories, counts = zip(*sorted_categories)
        
        plt.bar(categories, counts)
        plt.title('Distribution of Predicted Technology Categories')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            plt.text(i, count + max(counts) * 0.01, str(count), ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Category distribution plot saved to {save_path}")
        
        plt.show()
    
    def get_classification_report(self, X_test: pd.DataFrame, y_test: pd.Series) -> str:
        """Get detailed classification report"""
        if not self.is_trained:
            return "Model not trained"
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        # Generate report
        report = classification_report(
            y_test_encoded, y_pred_encoded,
            target_names=self.label_encoder.classes_,
            digits=4
        )
        
        return report 