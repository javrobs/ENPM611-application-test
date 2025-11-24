"""
Unit Tests for Feature 1: Label Resolution Time Analysis
"""

import unittest
from unittest.mock import Mock, patch, mock_open
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

# Import the classes to test
from app.feature_runner import FeatureRunner
from controllers.label_resolution_controller import LabelResolutionController
from analysis.label_resolution_analyzer import LabelResolutionAnalyzer
from model import LabelResolutionPredictor
from visualization.label_resolution_visualizer import LabelResolutionVisualizer
from data_loader import DataLoader
from utils.datetime_helper import extract_day_hour

class TestFeatureRunner(unittest.TestCase):
    """Test cases for FeatureRunner - the entry point for Feature 1"""

    def setUp(self):
        """Set up test fixtures"""
        self.runner = FeatureRunner()

    def test_initialization(self):
        """Test FeatureRunner initialization"""
        self.assertIsNotNone(self.runner)
        self.assertIsNone(self.runner.config)
        self.assertIsNone(self.runner.contributors_controller)
        self.assertIsNone(self.runner.priority_controller)

    @patch('app.feature_runner.ConfigManager')
    @patch('app.feature_runner.ContributorsController')
    @patch('app.feature_runner.DataLoader')
    @patch('app.feature_runner.PriorityController')
    def test_initialize_components(self, mock_priority, mock_loader, mock_contrib, mock_config):
        """Test initialize_components method"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance

        mock_contrib_instance = Mock()
        mock_contrib.return_value = mock_contrib_instance

        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance

        mock_priority_instance = Mock()
        mock_priority.return_value = mock_priority_instance

        # Call initialize_components
        self.runner.initialize_components()

        # Verify components were initialized
        self.assertIsNotNone(self.runner.config)
        self.assertIsNotNone(self.runner.contributors_controller)
        self.assertIsNotNone(self.runner.priority_controller)

        # Verify constructor calls
        mock_config.assert_called_once_with("config.json")
        mock_contrib.assert_called_once()
        mock_loader.assert_called_once()
        mock_priority.assert_called_once_with(mock_loader_instance)

    @patch('app.feature_runner.DataLoader')
    @patch('app.feature_runner.LabelResolutionController')
    @patch('app.feature_runner.LabelResolutionVisualizer')
    @patch('app.feature_runner.ConfigManager')
    def test_run_feature_1(self, mock_config, mock_visualizer, mock_controller, mock_loader):
        """Test run_feature method with feature_number=1"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.get_data_path.return_value = 'test_data.json'
        mock_config.return_value = mock_config_instance
        self.runner.config = mock_config_instance

        mock_loader_instance = Mock()
        mock_issues = self._create_mock_issues()
        mock_loader_instance.get_issues.return_value = mock_issues
        mock_loader.return_value = mock_loader_instance

        mock_controller_instance = Mock()
        mock_results = {
            'summary_statistics': {'total_closed_issues': 10},
            'label_statistics': {'bug': {'count': 5, 'median_days': 10.5}},
            'model_performance': {'status': 'success'},
            'open_issue_predictions': []
        }
        mock_controller_instance.run_full_analysis.return_value = mock_results
        mock_controller_instance.query_label_resolution_time.return_value = {
            'status': 'success',
            'predicted_days': 10.5,
            'based_on_issues': 5,
            'confidence_range': {'min_days': 7.3, 'max_days': 13.7}
        }
        mock_controller.return_value = mock_controller_instance

        mock_visualizer_instance = Mock()
        mock_visualizer.return_value = mock_visualizer_instance

        # Run feature 1
        with patch('builtins.print'):
            self.runner.run_feature(1, label='bug')

        # Verify DataLoader was called
        mock_loader.assert_called()
        mock_loader_instance.get_issues.assert_called()

        # Verify LabelResolutionController was created with issues
        mock_controller.assert_called_once_with(mock_issues)

        # Verify run_full_analysis was called
        mock_controller_instance.run_full_analysis.assert_called_once()

        # Verify visualizer was created and used
        mock_visualizer.assert_called_once_with(mock_results)
        mock_visualizer_instance.generate_all_visualizations.assert_called_once()

        # Verify query_label_resolution_time was called with the label
        mock_controller_instance.query_label_resolution_time.assert_called_once_with('bug')

    @patch('app.feature_runner.DataLoader')
    @patch('app.feature_runner.LabelResolutionController')
    @patch('app.feature_runner.ConfigManager')
    def test_run_feature_1_without_label(self, mock_config, mock_controller, mock_loader):
        """Test run_feature method with feature_number=1 and no label parameter"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.get_data_path.return_value = 'test_data.json'
        mock_config.return_value = mock_config_instance
        self.runner.config = mock_config_instance

        mock_loader_instance = Mock()
        mock_issues = self._create_mock_issues()
        mock_loader_instance.get_issues.return_value = mock_issues
        mock_loader.return_value = mock_loader_instance

        mock_controller_instance = Mock()
        mock_results = {
            'summary_statistics': {'total_closed_issues': 10},
            'label_statistics': {'bug': {'count': 5, 'median_days': 10.5}},
            'model_performance': {'status': 'success'},
            'open_issue_predictions': []
        }
        mock_controller_instance.run_full_analysis.return_value = mock_results
        mock_controller_instance.query_label_resolution_time.return_value = {
            'status': 'success',
            'predicted_days': 10.5,
            'based_on_issues': 5,
            'confidence_range': {'min_days': 7.3, 'max_days': 13.7}
        }
        mock_controller.return_value = mock_controller_instance

        # Run feature 1 without label (None)
        with patch('builtins.print'):
            with patch('app.feature_runner.LabelResolutionVisualizer'):
                self.runner.run_feature(1, label=None)

        # Verify controller methods were still called
        mock_controller_instance.run_full_analysis.assert_called_once()
        mock_controller_instance.query_label_resolution_time.assert_called_once_with(None)

    @patch('app.feature_runner.ConfigManager')
    def test_run_feature_invalid_number(self, mock_config):
        """Test run_feature with invalid feature number"""
        mock_config_instance = Mock()
        mock_config.return_value = mock_config_instance
        self.runner.config = mock_config_instance

        # Run with invalid feature number
        with patch('builtins.print') as mock_print:
            self.runner.run_feature(999)

        # Should print error message
        mock_print.assert_called()

    def _create_mock_issues(self):
        """Helper method to create mock issues"""
        issues = []
        for i in range(5):
            issue = Mock()
            issue.number = i + 1
            issue.state = 'closed'
            issue.labels = ['bug']
            issue.created_date = datetime.now() - timedelta(days=20)
            issue.closed_at = datetime.now() - timedelta(days=5)
            issues.append(issue)
        return issues

class TestDataLoader(unittest.TestCase):
    """Test DataLoader methods used in Feature 1"""

    @patch('data_loader.open', new_callable=mock_open, read_data='[{"number": 1}]')
    @patch('data_loader.config.get_parameter')
    def test_load_json(self, mock_config, mock_file):
        """Test loading JSON data"""
        mock_config.return_value = 'test_data.json'

        loader = DataLoader()
        data = loader.load_json()

        self.assertIsInstance(data, list)
        mock_file.assert_called_once_with('test_data.json', 'r')

    @patch.object(DataLoader, 'load_json')
    @patch('data_loader.config.get_parameter')
    def test_get_issues(self, mock_config, mock_load):
        """Test get_issues returns Issue objects"""
        mock_config.return_value = 'test_data.json'
        mock_load.return_value = [
            {'number': 1, 'state': 'closed', 'labels': ['bug']}
        ]

        loader = DataLoader()
        issues = loader.get_issues()

        self.assertIsInstance(issues, list)
        self.assertGreater(len(issues), 0)

class TestLabelResolutionController(unittest.TestCase):
    """Test cases for LabelResolutionController - Feature 1 specific methods"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock issues with proper datetime objects
        self.mock_issues = self._create_mock_issues()

        # Create temporary output directory
        self.temp_dir = tempfile.mkdtemp()

        # Create controller instance
        self.controller = LabelResolutionController(self.mock_issues)
        self.controller.output_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_issues(self):
        """Create mock Issue objects for testing"""
        issues = []

        # Closed issues with different labels and resolution times
        for i in range(10):
            issue = Mock()
            issue.number = i + 1
            issue.state = 'closed'
            issue.title = f'Test Issue {i + 1}'
            issue.labels = ['bug', 'area/core'] if i % 2 == 0 else ['feature', 'enhancement']
            issue.created_date = datetime.now() - timedelta(days=30 + i*5)
            issue.closed_at = datetime.now() - timedelta(days=i*2)
            issue.created_at = issue.created_date
            issue.events = None  # No events needed since closed_at is set
            issues.append(issue)

        # Open issues
        for i in range(5):
            issue = Mock()
            issue.number = i + 11
            issue.state = 'open'
            issue.title = f'Open Issue {i + 1}'
            issue.labels = ['bug'] if i % 2 == 0 else ['feature']
            issue.created_date = datetime.now() - timedelta(days=10 + i)
            issue.closed_at = None
            issue.created_at = issue.created_date
            issue.events = None
            issues.append(issue)

        return issues

    def test_initialization(self):
        """Test controller initialization"""
        self.assertIsNotNone(self.controller.issues)
        self.assertIsInstance(self.controller.analyzer, LabelResolutionAnalyzer)
        self.assertIsInstance(self.controller.predictor, LabelResolutionPredictor)
        self.assertIsInstance(self.controller.output_dir, Path)

    def test_run_full_analysis(self):
        """Test run_full_analysis - the main method called by Feature 1"""
        # Don't mock analyzer or predictor - let them run actual logic
        
        # Mock only _predict_open_issues, save and print methods
        with patch.object(self.controller, '_predict_open_issues') as mock_pred_open:
            mock_pred_open.return_value = []

            # Mock save and print methods
            with patch.object(self.controller, '_save_results'):
                with patch.object(self.controller, '_print_summary'):
                    # Run the analysis
                    results = self.controller.run_full_analysis()

        # Assertions
        self.assertIsNotNone(results)
        self.assertIn('analysis_date', results)
        self.assertIn('summary_statistics', results)
        self.assertIn('label_statistics', results)
        self.assertIn('model_performance', results)
        self.assertIn('open_issue_predictions', results)
        
        # Verify analyzer ran successfully
        self.assertGreater(results['summary_statistics']['total_closed_issues'], 0)
        self.assertGreater(len(results['label_statistics']), 0)
        
        # Verify model was actually trained
        self.assertEqual(results['model_performance']['status'], 'success')
        self.assertIn('ensemble', results['model_performance'])
        self.assertIn('training_samples', results['model_performance'])

    def test_run_full_analysis_insufficient_samples(self):
        """Test run_full_analysis with insufficient training samples (< 10)"""
        # Create controller with only 5 closed issues
        small_issues = []
        for i in range(5):
            issue = Mock()
            issue.number = i + 1
            issue.state = 'closed'
            issue.title = f'Test Issue {i + 1}'
            issue.labels = ['bug']
            issue.created_date = datetime.now() - timedelta(days=20 + i*2)
            issue.closed_at = datetime.now() - timedelta(days=i)
            issue.created_at = issue.created_date
            issue.events = None
            small_issues.append(issue)
        
        small_controller = LabelResolutionController(small_issues)
        small_controller.output_dir = Path(self.temp_dir)

        # Mock save and print methods
        with patch.object(small_controller, '_save_results'):
            with patch.object(small_controller, '_print_summary'):
                # Run the analysis
                results = small_controller.run_full_analysis()

        # Assertions - predictor should fail due to insufficient samples
        self.assertIsNotNone(results)
        self.assertIn('model_performance', results)
        self.assertEqual(results['model_performance']['status'], 'error')
        self.assertIn('insufficient', results['model_performance']['message'].lower())
        
        # But analyzer should still work
        self.assertGreater(results['summary_statistics']['total_closed_issues'], 0)
        self.assertIn('bug', results['label_statistics'])

    def test_analyzer_statistics_accuracy(self):
        """Test that analyzer calculates statistics correctly"""
        # Run analyzer directly
        self.controller.analyzer.analyze_closed_issues()
        
        # Verify label stats were calculated
        self.assertGreater(len(self.controller.analyzer.label_stats), 0)
        
        # Check that bug label exists (5 issues have bug label)
        self.assertIn('bug', self.controller.analyzer.label_stats)
        bug_stats = self.controller.analyzer.label_stats['bug']
        
        # Verify statistics structure
        self.assertIn('count', bug_stats)
        self.assertIn('median_days', bug_stats)
        self.assertIn('mean_days', bug_stats)
        self.assertIn('std_dev_hours', bug_stats)
        
        # Verify count is correct (5 closed issues have bug label)
        self.assertEqual(bug_stats['count'], 5)
        
        # Verify resolution data was stored
        self.assertGreater(len(self.controller.analyzer.resolution_data), 0)

    def test_analyzer_feature_extraction(self):
        """Test that analyzer extracts features correctly"""
        # Run analyzer first to populate resolution_data
        self.controller.analyzer.analyze_closed_issues()
        
        # Extract features
        features, labels = self.controller.analyzer.extract_features_for_ml()
        
        # Verify features and labels
        self.assertIsInstance(features, list)
        self.assertIsInstance(labels, list)
        self.assertEqual(len(features), len(labels))
        self.assertGreater(len(features), 0)
        
        # Verify feature vector structure
        feature_names = self.controller.analyzer.get_feature_names()
        self.assertEqual(len(features[0]), len(feature_names))
        
        # Verify feature names
        expected_features = [
            'num_labels',
            'has_bug_label',
            'has_feature_label',
            'has_docs_label',
            'has_area_label',
            'day_of_week',
            'month'
        ]
        self.assertEqual(feature_names, expected_features)

    def test_analyzer_summary_statistics(self):
        """Test that analyzer calculates summary statistics correctly"""
        # Run analyzer first
        self.controller.analyzer.analyze_closed_issues()
        
        # Get summary statistics
        summary = self.controller.analyzer.get_summary_statistics()
        
        # Verify summary structure
        self.assertIn('total_closed_issues', summary)
        self.assertIn('total_unique_labels', summary)
        self.assertIn('overall_median_days', summary)
        self.assertIn('overall_mean_days', summary)
        
        # Verify values
        self.assertEqual(summary['total_closed_issues'], 10)  # 10 closed issues
        self.assertGreater(summary['total_unique_labels'], 0)
        self.assertGreater(summary['overall_median_days'], 0)

    def test_query_label_resolution_time(self):
        """Test query_label_resolution_time - called in Feature 1"""
        # Run analyzer first to populate label_stats
        self.controller.analyzer.analyze_closed_issues()

        # Query for bug label
        result = self.controller.query_label_resolution_time('bug')

        self.assertEqual(result['status'], 'success')
        self.assertIn('predicted_days', result)
        self.assertIn('based_on_issues', result)
        self.assertIn('confidence_range', result)
        self.assertEqual(result['based_on_issues'], 5)

    def test_query_label_resolution_time_unknown_label(self):
        """Test query for unknown label"""
        # Run analyzer first
        self.controller.analyzer.analyze_closed_issues()
        
        # Query for non-existent label
        result = self.controller.query_label_resolution_time('nonexistent')
        
        self.assertEqual(result['status'], 'unknown')
        self.assertIn('message', result)

    def test_predict_open_issues(self):
        """Test _predict_open_issues - internal method used in run_full_analysis"""
        # Run analyzer first
        self.controller.analyzer.analyze_closed_issues()
        
        # Train predictor
        features, labels = self.controller.analyzer.extract_features_for_ml()
        feature_names = self.controller.analyzer.get_feature_names()
        self.controller.predictor.train(features, labels, feature_names)

        # Predict open issues
        predictions = self.controller._predict_open_issues()

        # Should have predictions for open issues only (5 in our mock data)
        self.assertEqual(len(predictions), 5)
        for pred in predictions:
            self.assertIn('issue_number', pred)
            self.assertIn('title', pred)
            self.assertIn('labels', pred)
            self.assertIn('predicted_resolution_days', pred)
            self.assertIn('confidence_interval', pred)
            self.assertIn('created_at', pred)

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_results(self, mock_json_dump, mock_file):
        """Test _save_results - saves output files in Feature 1"""
        results = {
            'analysis_date': datetime.now().isoformat(),
            'summary_statistics': {'total_closed_issues': 10},
            'label_statistics': {'bug': {'count': 5}},
            'model_performance': {'status': 'success'},
            'open_issue_predictions': []
        }

        self.controller._save_results(results)

        # Verify that 3 files were written (main results, label stats, predictions)
        self.assertEqual(mock_file.call_count, 3)
        self.assertEqual(mock_json_dump.call_count, 3)

    def test_print_summary(self):
        """Test _print_summary - prints results at end of Feature 1"""
        results = {
            'summary_statistics': {
                'total_closed_issues': 10,
                'total_unique_labels': 3,
                'overall_median_days': 15.5,
                'overall_mean_days': 18.2
            },
            'label_statistics': {
                'bug': {'count': 5, 'median_days': 10.5},
                'feature': {'count': 3, 'median_days': 20.0}
            },
            'model_performance': {
                'status': 'success',
                'ensemble': {'mae_days': 5.2, 'r2_score': 0.85},
                'feature_importance': {'label_count': 0.4, 'label_complexity': 0.3}
            },
            'open_issue_predictions': [
                {'issue_number': 1, 'predicted_resolution_days': 10.0, 'labels': ['bug']}
            ]
        }

        # Should not raise exception
        with patch('builtins.print'):
            self.controller._print_summary(results)

class TestLabelResolutionVisualizer(unittest.TestCase):
    """Test cases for LabelResolutionVisualizer - Feature 1 visualization methods"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

        # Create mock results
        self.mock_results = {
            'summary_statistics': {
                'total_closed_issues': 100,
                'total_unique_labels': 10,
                'overall_median_days': 15.5,
                'overall_mean_days': 18.2
            },
            'label_statistics': {
                'bug': {'count': 50, 'median_days': 10.5, 'mean_days': 12.0, 'std_days': 3.2},
                'feature': {'count': 30, 'median_days': 25.0, 'mean_days': 27.5, 'std_days': 5.1},
                'docs': {'count': 20, 'median_days': 8.0, 'mean_days': 9.2, 'std_days': 2.1}
            },
            'model_performance': {
                'status': 'success',
                'random_forest': {'mae_days': 5.2, 'r2_score': 0.82},
                'gradient_boosting': {'mae_days': 4.8, 'r2_score': 0.85},
                'ensemble': {'mae_days': 4.5, 'r2_score': 0.87},
                'feature_importance': {'label_count': 0.35, 'label_complexity': 0.25}
            },
            'open_issue_predictions': []
        }

        self.visualizer = LabelResolutionVisualizer(self.mock_results)
        self.visualizer.output_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_all_visualizations(self, mock_close, mock_savefig):
        """Test generate_all_visualizations - main method called in Feature 1"""
        with patch('builtins.print'):
            self.visualizer.generate_all_visualizations()

        # Should generate 6 visualizations (plot_temporal_trends is not implemented)
        self.assertEqual(mock_savefig.call_count, 6)
        self.assertEqual(mock_close.call_count, 6)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_label_resolution_comparison(self, mock_close, mock_savefig):
        """Test label resolution comparison plot"""
        self.visualizer.plot_label_resolution_comparison()

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_resolution_time_distribution(self, mock_close, mock_savefig):
        """Test resolution time distribution plot"""
        self.visualizer.plot_resolution_time_distribution()

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_feature_importance(self, mock_close, mock_savefig):
        """Test feature importance plot"""
        self.visualizer.plot_feature_importance()

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_prediction_accuracy(self, mock_close, mock_savefig):
        """Test prediction accuracy plot"""
        self.visualizer.plot_prediction_accuracy()

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_label_category_analysis(self, mock_close, mock_savefig):
        """Test label category analysis plot"""
        self.visualizer.plot_label_category_analysis()

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_top_labels_comparison(self, mock_close, mock_savefig):
        """Test top labels comparison plot"""
        self.visualizer.plot_top_labels_comparison()

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

class TestFeatureRunnerIntegration(unittest.TestCase):
    """Integration tests for Feature 1 complete workflow through FeatureRunner"""

    @patch('app.feature_runner.ConfigManager')
    @patch('app.feature_runner.ContributorsController')
    @patch('app.feature_runner.DataLoader')
    @patch('app.feature_runner.PriorityController')
    @patch('app.feature_runner.LabelResolutionController')
    @patch('app.feature_runner.LabelResolutionVisualizer')
    def test_complete_feature_1_workflow(self, mock_viz, mock_controller,
                                         mock_priority, mock_loader,
                                         mock_contrib, mock_config):
        """Test the complete Feature 1 workflow from FeatureRunner"""
        # Setup mocks
        mock_config_instance = Mock()
        mock_config_instance.get_data_path.return_value = 'test_data.json'
        mock_config.return_value = mock_config_instance

        mock_loader_instance = Mock()
        mock_issues = self._create_mock_issues()
        mock_loader_instance.get_issues.return_value = mock_issues
        mock_loader.return_value = mock_loader_instance

        mock_controller_instance = Mock()
        mock_results = {
            'analysis_date': datetime.now().isoformat(),
            'summary_statistics': {
                'total_closed_issues': 20,
                'total_unique_labels': 5,
                'overall_median_days': 15.5,
                'overall_mean_days': 18.2
            },
            'label_statistics': {
                'bug': {'count': 10, 'median_days': 10.5, 'mean_days': 12.0},
                'feature': {'count': 5, 'median_days': 25.0, 'mean_days': 27.5}
            },
            'model_performance': {
                'status': 'success',
                'training_samples': 20,
                'ensemble': {'r2_score': 0.85, 'mae_days': 5.0}
            },
            'open_issue_predictions': [
                {'issue_number': 21, 'predicted_resolution_days': 12.5}
            ]
        }
        mock_controller_instance.run_full_analysis.return_value = mock_results
        mock_controller_instance.query_label_resolution_time.return_value = {
            'status': 'success',
            'predicted_days': 10.5,
            'based_on_issues': 10,
            'confidence_range': {'min_days': 7.3, 'max_days': 13.7}
        }
        mock_controller.return_value = mock_controller_instance

        mock_viz_instance = Mock()
        mock_viz.return_value = mock_viz_instance

        # Create and initialize runner
        runner = FeatureRunner()
        runner.initialize_components()

        # Run Feature 1
        with patch('builtins.print'):
            runner.run_feature(1, label='bug')

        # Verify the complete workflow
        mock_loader_instance.get_issues.assert_called()
        mock_controller.assert_called_once()
        mock_controller_instance.run_full_analysis.assert_called_once()
        mock_viz.assert_called_once_with(mock_results)
        mock_viz_instance.generate_all_visualizations.assert_called_once()
        mock_controller_instance.query_label_resolution_time.assert_called_once_with('bug')

    def _create_mock_issues(self):
        """Create comprehensive mock issues for testing"""
        issues = []

        # Closed issues
        for i in range(20):
            issue = Mock()
            issue.number = i + 1
            issue.state = 'closed'
            issue.title = f'Closed Issue {i + 1}'
            issue.labels = ['bug'] if i % 2 == 0 else ['feature']
            issue.created_date = datetime.now() - timedelta(days=60 - i*2)
            issue.closed_at = datetime.now() - timedelta(days=30 - i)
            issue.created_at = issue.created_date
            issues.append(issue)

        # Open issues
        for i in range(5):
            issue = Mock()
            issue.number = i + 21
            issue.state = 'open'
            issue.title = f'Open Issue {i + 1}'
            issue.labels = ['bug'] if i % 2 == 0 else ['feature']
            issue.created_date = datetime.now() - timedelta(days=15 + i)
            issue.closed_at = None
            issue.created_at = issue.created_date
            issues.append(issue)

        return issues

class TestDateTimeHelper(unittest.TestCase):
    """Essential test cases for datetime_helper module"""

    def test_extract_day_hour_with_datetime(self):
        """Test extract_day_hour with valid datetime"""
        dt = datetime(2025, 1, 6, 14, 30, 0)  # Monday 14:30
        result = extract_day_hour(dt)
        self.assertEqual(result, (0, 14))

    def test_extract_day_hour_with_none(self):
        """Test extract_day_hour with None"""
        result = extract_day_hour(None)
        self.assertIsNone(result)

    def test_extract_day_hour_different_days(self):
        """Test extract_day_hour for different days"""
        test_cases = [
            (datetime(2025, 1, 6, 10, 0), (0, 10)),   # Monday
            (datetime(2025, 1, 12, 23, 0), (6, 23)),  # Sunday
        ]
        for dt, expected in test_cases:
            self.assertEqual(extract_day_hour(dt), expected)

def suite():
    """Create a test suite containing all test cases"""
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFeatureRunner))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDataLoader))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLabelResolutionController))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLabelResolutionVisualizer))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFeatureRunnerIntegration))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDateTimeHelper))
    return test_suite


if __name__ == '__main__':
    # Run all tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())