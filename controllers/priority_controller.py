"""
Priority Controller (Updated with Visualization)
Orchestrates the priority prediction workflow with visualizations.
"""

import json
import numpy as np
from analysis.priority_analyzer import PriorityAnalyzer
from analysis.priority_visualizer import PriorityVisualizer  # NEW IMPORT
from model import IssuePredictionModel


class PriorityController:
    """
    Controller for managing priority prediction workflow.
    Coordinates between data, analysis, ML models, and visualizations.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the controller.
        
        Args:
            data_loader: DataLoader instance with loaded issues
        """
        self.data_loader = data_loader
        self.issues = None
        self.analyzer = None
        self.model = None
        self.raw_issues = None
        self.visualizer = None  # NEW
        
    def execute_priority_workflow(self, output_file='improved_predictions.json'):
        """
        Execute the complete priority prediction workflow with visualizations.
        
        Args:
            output_file (str): Output file path for predictions
            
        Returns:
            list: Predictions for open issues
        """
        print("\n" + "="*60)
        print("PRIORITY PREDICTION WORKFLOW")
        print("="*60)
        
        # Step 1: Load and analyze data
        self.raw_issues = self.data_loader.load_json()
        self.analyzer = PriorityAnalyzer(self.raw_issues)
        
        print(f"\n✓ Loaded {len(self.raw_issues)} issues")
        print(f"  - Closed: {len(self.analyzer.closed_issues)}")
        print(f"  - Open: {len(self.analyzer.open_issues)}")
        
        # Step 2: Show statistics
        self._display_statistics()
        
        # Step 3: Train model
        self.model = IssuePredictionModel()
        self._train_model()
        
        # Step 4: Predict for open issues
        predictions = self._predict_open_issues()
        
        # Step 5: Generate visualizations (NEW)
        self._generate_visualizations(predictions)
        
        # Step 6: Save results
        self._save_predictions(predictions, output_file)
        
        return predictions
    
    def _display_statistics(self):
        """Display analysis statistics."""
        print("\n" + "="*60)
        print("DATA ANALYSIS")
        print("="*60)
        
        # Resolution time statistics
        res_stats = self.analyzer.get_resolution_statistics()
        if res_stats:
            print(f"\n✓ Resolution Time Statistics:")
            print(f"  Median: {res_stats['median_days']} days")
            print(f"  Mean: {res_stats['mean_days']} days")
            print(f"  75th percentile: {res_stats['p75_days']} days")
            print(f"  95th percentile: {res_stats['p95_days']} days")
            print(f"  Sample size: {res_stats['count']} issues")
        
        # Urgency distribution
        urg_stats = self.analyzer.get_urgency_statistics()
        if urg_stats['total'] > 0:
            print(f"\n✓ Urgency Distribution:")
            for urgency in ['Critical', 'High', 'Medium', 'Low']:
                count = urg_stats['counts'].get(urgency, 0)
                pct = urg_stats['percentages'].get(urgency, 0)
                if count > 0:
                    print(f"  {urgency}: {count} ({pct}%)")
    
    def _train_model(self):
        """Train the ML model on closed issues."""
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        X_features = []
        y_urgency = []
        metadata = []
        
        valid_count = 0
        skipped_count = 0
        
        print("\n✓ Extracting features from closed issues...")
        
        for i, issue in enumerate(self.analyzer.closed_issues, 1):
            if i % 100 == 0:
                print(f"  Processing {i}/{len(self.analyzer.closed_issues)}...", end='\r')
            
            resolution_time = self.analyzer.get_resolution_time(issue)
            
            if resolution_time is None or resolution_time <= 0:
                skipped_count += 1
                continue
            
            features = self.analyzer.extract_features(issue)
            urgency = self.analyzer.assign_urgency_category(issue, resolution_time)
            complexity = self.analyzer.calculate_complexity_score(issue)
            
            X_features.append(features)
            y_urgency.append(urgency)
            
            metadata.append({
                'number': issue['number'],
                'title': issue['title'],
                'url': issue['url'],
                'complexity_score': complexity,
                'urgency': urgency,
                'labels': issue.get('labels', [])
            })
            
            valid_count += 1
        
        print(f"\n✓ Processed {valid_count} valid issues (skipped {skipped_count})")
        
        if valid_count == 0:
            print("\n✗ Error: No valid training data!")
            return
        
        self.model.train(X_features, y_urgency, metadata)
    
    def _predict_open_issues(self):
        """Predict priority and complexity for open issues."""
        print("\n" + "="*60)
        print("PREDICTING FOR OPEN ISSUES")
        print("="*60)
        
        predictions = []
        
        print(f"\n✓ Generating predictions for {len(self.analyzer.open_issues)} open issues...")
        
        for issue in self.analyzer.open_issues:
            features = self.analyzer.extract_features(issue)
            complexity = self.analyzer.calculate_complexity_score(issue)
            prediction = self.model.predict(features, complexity) 
            
            prediction.update({
                'number': issue['number'],
                'title': issue['title'],
                'url': issue['url'],
                'labels': issue.get('labels', []),
                'num_comments': features['num_comments']
            })
            
            predictions.append(prediction)
        
        priority_rank = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        predictions.sort(
            key=lambda x: (priority_rank.get(x['predicted_priority'], 0), 
                          x['complexity_score']), 
            reverse=True
        )
        
        self._display_top_predictions(predictions)
        
        return predictions
    
    def _display_top_predictions(self, predictions, top_n=15):
        """Display top predictions."""
        print(f"\n✓ Predicted for {len(predictions)} open issues")
        print(f"\nTop {top_n} Issues by Priority and Complexity:")
        print("-" * 60)
        
        for i, pred in enumerate(predictions[:top_n], 1):
            print(f"\n{i}. [{pred['predicted_priority']}] #{pred['number']}")
            print(f"   {pred['title'][:70]}")
            print(f"   Complexity Score: {pred['complexity_score']}/100")
            print(f"   Confidence: {pred['priority_confidence']}%")
            print(f"   Current activity: {pred['num_comments']} comments")
            print(f"   URL: {pred['url']}")
    
    def _generate_visualizations(self, predictions):
        """
        Generate and save visualizations (NEW METHOD).
        
        Args:
            predictions (list): List of predictions
        """
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        self.visualizer = PriorityVisualizer(predictions)
        
        # Print summary statistics
        self.visualizer.print_summary_statistics()
        
        # Create and save visualizations
        self.visualizer.create_visualizations()
    
    def _save_predictions(self, predictions, output_file):
        """Save predictions to JSON file."""
        import os
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to {output_file}")