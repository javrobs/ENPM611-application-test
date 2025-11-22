"""
Implements a runtime data model that can be used to access
the properties contained in the issues JSON.
"""

from typing import List
from enum import Enum
from datetime import datetime, timedelta
from dateutil import parser
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import json
from typing import Dict, List, Tuple


class LabelResolutionPredictor:
    """
    Machine Learning model for predicting issue resolution times
    Uses Random Forest and Gradient Boosting ensemble
    """
    
    def __init__(self):
        """Initialize the prediction models"""
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.training_metrics = {}
        
    def train(self, features: List, labels: List, feature_names: List) -> Dict:
        """
        Train the ML models
        
        Args:
            features: List of feature vectors
            labels: List of resolution times (in hours)
            feature_names: Names of features
            
        Returns:
            Dictionary containing training metrics
        """
        if len(features) < 10:
            return {
                'status': 'error',
                'message': 'Insufficient training data. Need at least 10 samples.'
            }
        
        self.feature_names = feature_names
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        
        # Train Gradient Boosting
        self.gb_model.fit(X_train, y_train)
        gb_pred = self.gb_model.predict(X_test)
        
        # Ensemble prediction (average)
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        # Calculate metrics
        self.training_metrics = {
            'status': 'success',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'random_forest': {
                'mae_hours': float(mean_absolute_error(y_test, rf_pred)),
                'mae_days': float(mean_absolute_error(y_test, rf_pred) / 24),
                'rmse_hours': float(np.sqrt(mean_squared_error(y_test, rf_pred))),
                'r2_score': float(r2_score(y_test, rf_pred))
            },
            'gradient_boosting': {
                'mae_hours': float(mean_absolute_error(y_test, gb_pred)),
                'mae_days': float(mean_absolute_error(y_test, gb_pred) / 24),
                'rmse_hours': float(np.sqrt(mean_squared_error(y_test, gb_pred))),
                'r2_score': float(r2_score(y_test, gb_pred))
            },
            'ensemble': {
                'mae_hours': float(mean_absolute_error(y_test, ensemble_pred)),
                'mae_days': float(mean_absolute_error(y_test, ensemble_pred) / 24),
                'rmse_hours': float(np.sqrt(mean_squared_error(y_test, ensemble_pred))),
                'r2_score': float(r2_score(y_test, ensemble_pred))
            },
            'feature_importance': self._get_feature_importance()
        }
        
        self.is_trained = True
        return self.training_metrics
    
    def predict(self, features: List) -> Dict:
        """
        Predict resolution time for given features
        
        Args:
            features: Feature vector for prediction
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            return {
                'status': 'error',
                'message': 'Model not trained yet'
            }
        
        # Scale features
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        rf_pred = self.rf_model.predict(X_scaled)[0]
        gb_pred = self.gb_model.predict(X_scaled)[0]
        
        # Ensemble prediction
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        return {
            'status': 'success',
            'predicted_hours': float(ensemble_pred),
            'predicted_days': float(ensemble_pred / 24),
            'confidence_interval': {
                'lower_days': float(max(0, ensemble_pred - self.training_metrics['ensemble']['mae_hours']) / 24),
                'upper_days': float((ensemble_pred + self.training_metrics['ensemble']['mae_hours']) / 24)
            },
            'model_predictions': {
                'random_forest_days': float(rf_pred / 24),
                'gradient_boosting_days': float(gb_pred / 24)
            }
        }
    
    def predict_batch(self, features_list: List[List]) -> List[Dict]:
        """
        Predict resolution times for multiple issues
        
        Args:
            features_list: List of feature vectors
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        for features in features_list:
            pred = self.predict(features)
            predictions.append(pred)
        return predictions
    
    def _get_feature_importance(self) -> Dict:
        """
        Get feature importance from Random Forest model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance_dict = {}
        importances = self.rf_model.feature_importances_
        
        for name, importance in zip(self.feature_names, importances):
            importance_dict[name] = float(importance)
        
        # Sort by importance
        sorted_importance = dict(sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return sorted_importance
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained model
        
        Returns:
            Dictionary with model information
        """
        return {
            'is_trained': self.is_trained,
            'features': self.feature_names,
            'training_metrics': self.training_metrics if self.is_trained else None,
            'model_type': 'Random Forest + Gradient Boosting Ensemble'
        }
    
    def save_model(self, filepath: str) -> bool:
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_trained:
            return False
        
        try:
            model_data = {
                'rf_model': self.rf_model,
                'gb_model': self.gb_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'training_metrics': self.training_metrics
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load trained model from disk
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_data = joblib.load(filepath)
            self.rf_model = model_data['rf_model']
            self.gb_model = model_data['gb_model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.training_metrics = model_data['training_metrics']
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False




class State(str, Enum):
    # Whether issue is open or closed
    open = 'open'
    closed = 'closed'


class Event:
    def __init__(self, jobj: any):
        self.event_type: str = None
        self.author: str = None
        self.event_date: datetime = None
        self.label: str = None
        self.comment: str = None

        if jobj is not None:
            self.from_json(jobj)

    def from_json(self, jobj: any):
        self.event_type = jobj.get('event_type')
        self.author = jobj.get('author')
        try:
            self.event_date = parser.parse(jobj.get('event_date'))
        except Exception:
            pass
        self.label = jobj.get('label')
        self.comment = jobj.get('comment')

    def is_close_event(self) -> bool:
        return self.event_type == 'closed'

    def is_comment_event(self) -> bool:
        return self.event_type == 'commented'


class Issue:
    def __init__(self, jobj: any = None):
        self.url: str = None
        self.creator: str = None
        self.labels: List[str] = []
        self.state: State = None
        self.assignees: List[str] = []
        self.title: str = None
        self.text: str = None
        self.number: int = -1
        self.created_date: datetime = None
        self.updated_date: datetime = None
        self.timeline_url: str = None
        self.events: List[Event] = []
        self.assigned_priority: str = None

        if jobj is not None:
            self.from_json(jobj)

    def from_json(self, jobj: any):
        self.url = jobj.get('url')
        self.creator = jobj.get('creator')
        self.labels = jobj.get('labels', [])
        self.state = State[jobj.get('state')]
        self.assignees = jobj.get('assignees', [])
        self.title = jobj.get('title')
        self.text = jobj.get('text')
        try:
            self.number = int(jobj.get('number', '-1'))
        except Exception:
            pass
        try:
            self.created_date = parser.parse(jobj.get('created_date'))
        except Exception:
            pass
        try:
            self.updated_date = parser.parse(jobj.get('updated_date'))
        except Exception:
            pass
        self.timeline_url = jobj.get('timeline_url')
        self.events = [Event(jevent) for jevent in jobj.get('events', [])]


    def get_labels(self) -> List[str]:
        return self.labels

    def get_creation_date(self) -> datetime:
        return self.created_date

    def get_closure_date(self) -> datetime:
        closes = [e.event_date for e in self.events if e.is_close_event()]
        return min(closes) if closes else None

    def is_closed(self) -> bool:
        return self.state == State.closed

    def get_resolution_time(self) -> timedelta:
        if self.created_date and self.get_closure_date():
            return self.get_closure_date() - self.created_date
        return None

    def get_comment_count(self) -> int:
        return sum(1 for e in self.events if e.is_comment_event())

    def get_event_count(self) -> int:
        return len(self.events)

    def get_text_content(self) -> str:
        return self.text

    def get_title(self) -> str:
        return self.title

    def set_priority(self, priority: str):
        self.assigned_priority = priority

    def get_priority(self) -> str:
        return self.assigned_priority


class Contributor:
    def __init__(self, username: str):
        self.username = username
        self.issues_created: List[Issue] = []
        self.issues_closed: List[Issue] = []
        self.comments: List[Event] = []
        self.first_activity: datetime = None
        self.last_activity: datetime = None
        
    def add_issue(self, issue: Issue):
        self.issues_created.append(issue)
        self._update_activity(issue.created_date)

    def add_comment(self, event: Event):
        self.comments.append(event)
        self._update_activity(event.event_date)

    def _update_activity(self, date: datetime):
        if not date:
            return
        if not self.first_activity or date < self.first_activity:
            self.first_activity = date
        if not self.last_activity or date > self.last_activity:
            self.last_activity = date
    
    def add_closed_issue(self, issue: Issue):
        closure_date = issue.get_closure_date()
        if closure_date:
            self.issues_closed.append(issue)
            self._update_activity(closure_date)

    def get_activity_count(self) -> int:
        return (
            len(self.issues_created)
            + len(self.issues_closed)
            + len(self.comments)
        )

    def get_activity_count_by_year(self, year: int) -> int:
        # Returns this contributor's activity count for a specific year
        # Activity = issues created + comments + issues closed in that year
        count = 0
        # Issues created
        for issue in self.issues_created:
            if issue.created_date and issue.created_date.year == year:
                count += 1
        # Comments
        for comment in self.comments:
            if comment.event_date and comment.event_date.year == year:
                count += 1
        # Issues closed
        for issue in getattr(self, "issues_closed", []):
            closure_date = issue.get_closure_date()
            if closure_date and closure_date.year == year:
                count += 1

        return count

    def get_active_years(self) -> set[int]:
        
        # Returns all years in which this contributor had any activity.
        years = set()
        years.update([i.created_date.year for i in self.issues_created if i.created_date])
        years.update([e.event_date.year for e in self.comments if e.event_date])
        years.update([
            cd.year for i in getattr(self, "issues_closed", [])
            for cd in [i.get_closure_date()] if cd
        ])
        return years

class Comment:
    def __init__(self, id: str, event_date=None, issue_id=None):
        self.id = id
        self.event_date = event_date
        self.issue_id = issue_id

    def __repr__(self):
        return f"<Comment id={self.id} date={self.event_date}>"
    

class IssuePredictionModel:
    """
    ML Model for predicting issue priority and complexity.
    Combines text features (TF-IDF) with numeric features.
    """
    
    def __init__(self):
        """Initialize the prediction model with its components."""
        self.text_vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
        self.scaler = StandardScaler()
        
        # Model for priority classification only
        self.urgency_model = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42
        )
        
        # Store for similarity search
        self.closed_issue_vectors = None
        self.closed_issue_metadata = []
        
        # Training statistics
        self.training_stats = {}
        
    def prepare_feature_matrix(self, features_list, fit=False):
        """
        Convert feature dictionaries to ML-ready matrix.
        
        Args:
            features_list (list): List of feature dictionaries
            fit (bool): Whether to fit the vectorizer/scaler
            
        Returns:
            scipy.sparse matrix: Combined feature matrix
        """
        # Extract text
        X_text = [feat['text'] for feat in features_list]
        
        # Extract numeric features
        X_numeric = []
        for feat in features_list:
            X_numeric.append([
                feat['title_len'], feat['body_len'], feat['num_code_blocks'],
                feat['has_stack_trace'], feat['num_comments'], feat['num_events'],
                feat['num_participants'], feat['num_labels'], feat['has_bug'],
                feat['has_feature'], feat['has_docs'], feat['has_critical'],
                feat['has_triage'], feat['first_response_hours']
            ])
        
        # Transform text
        if fit:
            X_text_tfidf = self.text_vectorizer.fit_transform(X_text)
        else:
            X_text_tfidf = self.text_vectorizer.transform(X_text)
        
        # Scale numeric
        if fit:
            X_numeric_scaled = self.scaler.fit_transform(X_numeric)
        else:
            X_numeric_scaled = self.scaler.transform(X_numeric)
        
        # Combine
        X_combined = hstack([X_text_tfidf, X_numeric_scaled])
        
        return X_combined, X_text_tfidf
    
    def train(self, features_list, y_urgency, closed_issues_metadata):
        """
        Train priority classification model.
        
        Args:
            features_list (list): List of feature dictionaries
            y_urgency (list): Priority categories
            closed_issues_metadata (list): Metadata for similarity search
            
        Returns:
            dict: Training statistics
        """
        print("\n✓ Preparing feature matrices...")
        X_combined, X_text_tfidf = self.prepare_feature_matrix(features_list, fit=True)
        
        # Store for similarity search
        self.closed_issue_vectors = X_text_tfidf
        self.closed_issue_metadata = closed_issues_metadata
        
        # Split data
        X_train, X_test, y_urg_train, y_urg_test = train_test_split(
            X_combined, y_urgency, test_size=0.2, random_state=42
        )
        
        # Train priority classification model
        print("\n--- Training Priority Classification Model ---")
        self.urgency_model.fit(X_train, y_urg_train)
        y_urg_pred = self.urgency_model.predict(X_test)
        
        print(f"✓ Priority Classification Performance:")
        print(classification_report(y_urg_test, y_urg_pred, zero_division=0))
        
        # Feature importance
        self._print_feature_importance()
        
        # Store stats
        self.training_stats = {
            'urgency_report': classification_report(y_urg_test, y_urg_pred, output_dict=True, zero_division=0)
        }
        
        return self.training_stats
    
    def predict(self, features, complexity_score):
        """
        Predict priority for a single issue.
        
        Args:
            features (dict): Feature dictionary
            complexity_score (int): Pre-calculated complexity score (0-100)
            
        Returns:
            dict: Predictions (priority, confidence, complexity)
        """
        X_combined, _ = self.prepare_feature_matrix([features], fit=False)
        
        # Predict priority
        pred_urgency = self.urgency_model.predict(X_combined)[0]
        urgency_probs = self.urgency_model.predict_proba(X_combined)[0]
        
        return {
            'predicted_priority': pred_urgency,
            'priority_confidence': round(max(urgency_probs) * 100, 1),
            'complexity_score': complexity_score
        }
    
    def find_similar_issues(self, issue_text, top_k=5):
        """
        Find the most similar closed issues based on text content.
        
        Args:
            issue_text (str): Combined title and body text
            top_k (int): Number of similar issues to return
            
        Returns:
            list: Similar issues with metadata
        """
        if self.closed_issue_vectors is None or len(self.closed_issue_metadata) == 0:
            return []
        
        # Vectorize the new issue
        issue_vector = self.text_vectorizer.transform([issue_text])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(issue_vector, self.closed_issue_vectors)[0]
        
        # Get top K
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_issues = []
        for idx in top_indices:
            if idx < len(self.closed_issue_metadata):
                meta = self.closed_issue_metadata[idx]
                similar_issues.append({
                    'number': meta['number'],
                    'title': meta['title'],
                    'url': meta['url'],
                    'similarity': round(similarities[idx], 3),
                    'complexity_score': meta['complexity_score'],
                    'urgency': meta['urgency'],
                    'labels': meta['labels']
                })
        
        return similar_issues
    
    def _print_feature_importance(self):
        """Print top 10 most important features."""
        print("\n--- Top 10 Most Important Features ---")
        
        feature_names = (
            self.text_vectorizer.get_feature_names_out().tolist() + 
            ['title_len', 'body_len', 'num_code_blocks', 'has_stack_trace',
             'num_comments', 'num_events', 'num_participants', 'num_labels',
             'has_bug', 'has_feature', 'has_docs', 'has_critical', 'has_triage',
             'first_response_hours']
        )
        
        importances = self.urgency_model.feature_importances_
        indices = np.argsort(importances)[-10:]
        
        for idx in reversed(indices):
            print(f"  {feature_names[idx]}: {importances[idx]:.4f}")