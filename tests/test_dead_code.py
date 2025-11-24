import unittest
import os
import tempfile
import numpy as np
from model import LabelResolutionPredictor, IssuePredictionModel


class TestLabelResolutionPredictor(unittest.TestCase):
    """Testing code that is not used by the feature runner to reach coverage and find bugs in case it gets used in the future. Important to mention as a finding"""

    def setUp(self):
        """Set up test fixtures"""
        self.predictor = LabelResolutionPredictor()
        
        # Create sample training data
        self.features = [
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7],
            [4, 5, 6, 7, 8],
            [5, 6, 7, 8, 9],
            [6, 7, 8, 9, 10],
            [7, 8, 9, 10, 11],
            [8, 9, 10, 11, 12],
            [9, 10, 11, 12, 13],
            [10, 11, 12, 13, 14]
        ]
        self.labels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.feature_names = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
        
        # Train the model
        self.predictor.train(self.features, self.labels, self.feature_names)

    def test_save_model(self):
        """Test that the model can be saved to a file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save the model
            result = self.predictor.save_model(tmp_path)
            
            # Check that save was successful
            self.assertTrue(result)
            
            # Check that file was created
            self.assertTrue(os.path.exists(tmp_path))
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_load_model(self):
        """Test that the model can be loaded from a file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save the model
            self.predictor.save_model(tmp_path)
            
            # Create a new predictor and load the model
            new_predictor = LabelResolutionPredictor()
            result = new_predictor.load_model(tmp_path)
            
            # Check that load was successful
            self.assertTrue(result)
            
            # Check that the loaded model is trained
            self.assertTrue(new_predictor.is_trained)
            
            # Check that feature names are preserved
            self.assertEqual(new_predictor.feature_names, self.feature_names)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

class TestFeatureThreeSimilarIssues(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures for IssuePredictionModel"""
        self.model = IssuePredictionModel()
        
        # Create sample training data with features
        self.features_list = [
            {
                'text': 'bug in login system authentication failed',
                'title_len': 30, 'body_len': 200, 'num_code_blocks': 1,
                'has_stack_trace': 1, 'num_comments': 5, 'num_events': 10,
                'num_participants': 3, 'num_labels': 2, 'has_bug': 1,
                'has_feature': 0, 'has_docs': 0, 'has_critical': 1,
                'has_triage': 0, 'first_response_hours': 2.5
            },
            {
                'text': 'feature request add dark mode to settings',
                'title_len': 25, 'body_len': 150, 'num_code_blocks': 0,
                'has_stack_trace': 0, 'num_comments': 3, 'num_events': 5,
                'num_participants': 2, 'num_labels': 1, 'has_bug': 0,
                'has_feature': 1, 'has_docs': 0, 'has_critical': 0,
                'has_triage': 1, 'first_response_hours': 4.0
            }
        ]
        
        self.y_urgency = ['high', 'low']
        
        self.closed_issues_metadata = [
            {
                'number': 1, 'title': 'Login bug', 'url': 'http://example.com/1',
                'complexity_score': 75, 'urgency': 'high', 'labels': ['bug', 'critical']
            },
            {
                'number': 2, 'title': 'Dark mode feature', 'url': 'http://example.com/2',
                'complexity_score': 50, 'urgency': 'low', 'labels': ['feature']
            }
        ]
        
        # Train the model
        self.model.train(self.features_list, self.y_urgency, self.closed_issues_metadata)

    def test_find_similar_issues(self):
        """Test finding similar issues based on text content"""
        # Search for an issue similar to login/authentication issues
        test_text = "user login authentication problem with password"
        
        similar_issues = self.model.find_similar_issues(test_text, top_k=2)
        
        # Verify we get results
        self.assertIsInstance(similar_issues, list)
        self.assertGreater(len(similar_issues), 0)
        self.assertLessEqual(len(similar_issues), 3)
        
        # Verify structure of returned issues
        for issue in similar_issues:
            self.assertIn('number', issue)
            self.assertIn('title', issue)
            self.assertIn('url', issue)
            self.assertIn('similarity', issue)
            self.assertIn('complexity_score', issue)
            self.assertIn('urgency', issue)
            self.assertIn('labels', issue)
            
            # Verify similarity score is between 0 and 1
            self.assertGreaterEqual(issue['similarity'], 0)
            self.assertLessEqual(issue['similarity'], 1)

    def test_find_similar_issues_empty_model(self):
        """Test find_similar_issues returns empty list when model is not trained"""
        # Create a new untrained model
        untrained_model = IssuePredictionModel()
        
        result = untrained_model.find_similar_issues("test issue text")
        
        # Should return empty list
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
