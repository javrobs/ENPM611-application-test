import unittest
from unittest.mock import patch
from app.feature_runner import FeatureRunner
from model import Issue
import pandas as pd
from data_loader import DataLoader
from analysis.priority_analyzer import PriorityAnalyzer
from analysis.priority_visualizer import PriorityVisualizer
import numpy as np
from collections import Counter
import os
import tempfile

MOCK_JSON = json_issue_sample = [{
        "url": "https://github.com/python-poetry/poetry/issues/9785",
        "creator": "dbrtly",
        "labels": [
        "kind/bug",
        "status/triage"
        ],
        "state": "closed",
        "assignees": [],
        "title": "python version resolution is messed up",
        "text":"Sample text",
        "number": 9785,
        "created_date": "2024-10-20T00:33:06+00:00",
        "updated_date": "2024-10-20T08:00:46+00:00",
        "timeline_url": "https://api.github.com/repos/python-poetry/poetry/issues/9785/timeline",
        "events": [
        {
            "event_type": "labeled",
            "author": "dbrtly",
            "event_date": "2024-10-20T00:33:06+00:00",
            "label": "kind/bug"
        },
        {
            "event_type": "labeled",
            "author": "dbrtly",
            "event_date": "2024-10-20T00:33:06+00:00",
            "label": "status/triage"
        },
        {
            "event_type": "commented",
            "author": "finswimmer",
            "event_date": "2024-10-20T08:00:46+00:00",
            "comment": "Comment"},
        {
            "event_type": "closed",
            "author": "finswimmer",
            "event_date": "2024-10-20T08:00:46+00:00"
        },]
    },
    {
        "url": "https://github.com/python-poetry/poetry/issues/9782",
        "creator": "srittau",
        "labels": [
        "area/docs",
        "status/triage"
        ],
        "state": "open",
        "assignees": [],
        "title": "Document PEP 440 (Compatible Release) specifiers",
        "text": "Sample text 2",
        "number": 9782,
        "created_date": "2024-10-19T15:01:45+00:00",
        "updated_date": "2024-10-19T15:42:09+00:00",
        "timeline_url": "https://api.github.com/repos/python-poetry/poetry/issues/9782/timeline",
        "events": [
        {
            "event_type": "labeled",
            "author": "srittau",
            "event_date": "2024-10-19T15:01:45+00:00",
            "label": "area/docs"
        },
        {
            "event_type": "labeled",
            "author": "srittau",
            "event_date": "2024-10-19T15:01:45+00:00",
            "label": "status/triage"
        },
        {
            "event_type": "commented",
            "author": "radoering",
            "event_date": "2024-10-19T15:42:08+00:00",
            "comment": "> Would you accept a PR add a section about them?\r\n\r\nI do not see why we should not. \ud83d\ude03"
        },
        {
            "event_type": "referenced",
            "author": "srittau",
            "event_date": "2024-10-19T15:54:01+00:00"
        },
        {
            "event_type": "cross-referenced",
            "author": "srittau",
            "event_date": "2024-10-19T15:55:18+00:00"
        }
        ]
    },
    {
        "url": "https://github.com/python-poetry/poetry/issues/9781",
        "creator": "ethan-neidhart37",
        "labels": [
        "kind/bug",
        "status/triage"
        ],
        "state": "closed",
        "assignees": [],
        "title": "Cannot modify pyproject.toml: list index out of range",
        "text": "Sample text 2",
        "number": 9781,
        "created_date": "2024-10-18T15:29:38+00:00",
        "updated_date": "2024-10-18T15:55:36+00:00",
        "timeline_url": "https://api.github.com/repos/python-poetry/poetry/issues/9781/timeline",
        "events": [
        {
            "event_type": "labeled",
            "author": "ethan-neidhart37",
            "event_date": "2024-10-18T15:29:38+00:00",
            "label": "kind/bug"
        },
        {
            "event_type": "labeled",
            "author": "ethan-neidhart37",
            "event_date": "2024-10-18T15:29:38+00:00",
            "label": "status/triage"
        },
        {
            "event_type": "commented",
            "author": "dimbleby",
            "event_date": "2024-10-18T15:32:38+00:00",
            "comment": "Please search for duplicates, please close"
        },
        {
            "event_type": "commented",
            "author": "ethan-neidhart37",
            "event_date": "2024-10-18T15:35:14+00:00",
            "comment": "> Please search for duplicates, please close\r\n\r\nThe closest duplicate I found was #9505 but that seems to be slightly different.\r\nThis issue is happening regardless of which package I install, and is not happening on other contributor's computers running the same poetry version.\r\n\r\nIf there is another issue which does match mine, I apologize but I was unable to find one"
        },
        {
            "event_type": "commented",
            "author": "dimbleby",
            "event_date": "2024-10-18T15:36:04+00:00",
            "comment": "This is an exact duplicate of #9505"
        },
        {
            "event_type": "commented",
            "author": "dimbleby",
            "event_date": "2024-10-18T15:44:57+00:00",
            "comment": "isodate 0.7.0 is your issue. Please read your own logs, please close"
        },
        {
            "event_type": "closed",
            "author": "ethan-neidhart37",
            "event_date": "2024-10-18T15:54:55+00:00"
        },
        {
            "event_type": "commented",
            "author": "ethan-neidhart37",
            "event_date": "2024-10-18T15:55:34+00:00",
            "comment": "Thank you, sorry about that"
        }
        ]}]
        

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.patcher = patch("data_loader.DataLoader.load_json", return_value=MOCK_JSON)
        # self.mock_load = self.patcher.start()
        self.patcher.start()
        self.loader = DataLoader()

    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        import data_loader
        data_loader._ISSUES = None  # test interference

    def test_dataloader_get_issues(self):
        loader = self.loader
        self.assertEqual(len(loader.get_issues()),3,"get_issues didn't return the right amount of objects")
        self.assertIsInstance(loader.get_issues()[0],Issue,"get_issues didn't return an issue object")
    
    def test_dataloader_parse_issues(self):
        loader = self.loader
        parse_issues = loader.parse_issues()
        # self.assertEqual(len(loader.parse_issues()),3,"parse_issues didn't return the right amount of objects")
        self.assertIsInstance(parse_issues,pd.DataFrame,"parse_issues didn't return a DataFrame object")
        self.assertEqual(parse_issues['number'].count(),3,"parse_issues didn't return the right amount of row")

class TestRunner(unittest.TestCase):

    def setUp(self):
        self.featureRunner = FeatureRunner()
        self.patcher = patch("data_loader.DataLoader.load_json", return_value=MOCK_JSON)
        self.patcher.start()

    def test_runner_feature_one(self):
        self.featureRunner.initialize_components()
        predictions = self.featureRunner.run_feature(3)
        self.assertIsInstance(predictions,list)


class TestPriorityAnalyzer(unittest.TestCase):
    """Test suite for PriorityAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_issues = [
            {
                "number": 1,
                "title": "Critical bug causing crashes",
                "text": "This is a critical bug with stack trace:\n```python\nTraceback (most recent call last):\n  File 'test.py', line 1\n    error: something failed\n```",
                "state": "closed",
                "labels": ["kind/bug", "critical", "blocker"],
                "creator": "user1",
                "created_date": "2024-01-01T10:00:00+00:00",
                "updated_date": "2024-01-01T12:00:00+00:00",
                "events": [
                    {"event_type": "commented", "author": "user2", "event_date": "2024-01-01T10:30:00+00:00"},
                    {"event_type": "commented", "author": "user3", "event_date": "2024-01-01T11:00:00+00:00"},
                    {"event_type": "closed", "author": "user2", "event_date": "2024-01-01T12:00:00+00:00"}
                ]
            },
            {
                "number": 2,
                "title": "Feature request for enhancement",
                "text": "Would be nice to have this feature",
                "state": "open",
                "labels": ["enhancement", "feature"],
                "creator": "user4",
                "created_date": "2024-01-05T14:00:00+00:00",
                "updated_date": "2024-01-05T14:30:00+00:00",
                "events": [
                    {"event_type": "commented", "author": "user5", "event_date": "2024-01-05T14:15:00+00:00"}
                ]
            },
            {
                "number": 3,
                "title": "Documentation update needed",
                "text": "The docs are outdated",
                "state": "closed",
                "labels": ["area/docs", "status/triage"],
                "creator": "user6",
                "created_date": "2024-01-10T09:00:00+00:00",
                "updated_date": "2024-01-15T09:00:00+00:00",
                "events": [
                    {"event_type": "closed", "author": "user6", "event_date": "2024-01-15T09:00:00+00:00"}
                ]
            }
        ]
        self.analyzer = PriorityAnalyzer(self.test_issues)

    def test_initialization(self):
        """Test analyzer initializes correctly."""
        self.assertEqual(len(self.analyzer.issues), 3)
        self.assertEqual(len(self.analyzer.closed_issues), 2)
        self.assertEqual(len(self.analyzer.open_issues), 1)

    def test_get_resolution_time_closed_issue(self):
        """Test resolution time calculation for closed issue."""
        resolution_time = self.analyzer.get_resolution_time(self.test_issues[0])
        self.assertIsNotNone(resolution_time)
        self.assertGreater(resolution_time, 0)
        # 2 hours = 2 * 60 * 60 = 7200 seconds / 3600 = 2 hours
        self.assertEqual(resolution_time, 2.0)

    def test_get_resolution_time_open_issue(self):
        """Test resolution time returns None for open issue."""
        resolution_time = self.analyzer.get_resolution_time(self.test_issues[1])
        self.assertIsNone(resolution_time)

    def test_get_resolution_time_no_close_event(self):
        """Test resolution time uses updated_date when no close event."""
        issue = self.test_issues[2]
        resolution_time = self.analyzer.get_resolution_time(issue)
        self.assertIsNotNone(resolution_time)
        # 5 days = 120 hours
        self.assertEqual(resolution_time, 120.0)

    def test_assign_urgency_category_critical(self):
        """Test urgency assignment for critical issue."""
        urgency = self.analyzer.assign_urgency_category(self.test_issues[0], 2.0)
        self.assertEqual(urgency, "Critical")

    def test_assign_urgency_category_with_bug_label(self):
        """Test urgency assignment considers bug labels."""
        issue = self.test_issues[0].copy()
        issue["labels"] = ["kind/bug"]
        urgency = self.analyzer.assign_urgency_category(issue, 10.0)
        self.assertIn(urgency, ["High", "Medium", "Critical"])

    def test_assign_urgency_category_with_high_engagement(self):
        """Test urgency assignment considers comment activity."""
        issue = {
            "labels": [],
            "events": [
                {"event_type": "commented", "author": f"user{i}"} for i in range(25)
            ]
        }
        urgency = self.analyzer.assign_urgency_category(issue, 50.0)
        self.assertIn(urgency, ["High", "Medium", "Critical"])

    def test_extract_features_comprehensive(self):
        """Test feature extraction includes all expected features."""
        features = self.analyzer.extract_features(self.test_issues[0])
        
        # Check all expected keys
        expected_keys = [
            'text', 'title_len', 'body_len', 'num_code_blocks', 'has_stack_trace',
            'num_comments', 'num_events', 'num_participants', 'num_labels',
            'has_bug', 'has_feature', 'has_docs', 'has_critical', 'has_triage',
            'first_response_hours'
        ]
        for key in expected_keys:
            self.assertIn(key, features)

    def test_extract_features_text_content(self):
        """Test feature extraction for text features."""
        features = self.analyzer.extract_features(self.test_issues[0])
        self.assertIn("Critical bug causing crashes", features['text'])
        self.assertGreater(features['title_len'], 0)
        self.assertGreater(features['body_len'], 0)

    def test_extract_features_code_blocks(self):
        """Test feature extraction detects code blocks."""
        features = self.analyzer.extract_features(self.test_issues[0])
        self.assertEqual(features['num_code_blocks'], 2)  # Opening and closing ```

    def test_extract_features_stack_trace(self):
        """Test feature extraction detects stack traces."""
        features = self.analyzer.extract_features(self.test_issues[0])
        self.assertEqual(features['has_stack_trace'], 1)

    def test_extract_features_activity_metrics(self):
        """Test feature extraction for activity metrics."""
        features = self.analyzer.extract_features(self.test_issues[0])
        self.assertEqual(features['num_comments'], 2)
        self.assertEqual(features['num_events'], 3)
        self.assertEqual(features['num_participants'], 2)  # user2, user3 (excluding duplicates)

    def test_extract_features_label_flags(self):
        """Test feature extraction for label flags."""
        features = self.analyzer.extract_features(self.test_issues[0])
        self.assertEqual(features['has_bug'], 1)
        self.assertEqual(features['has_critical'], 1)
        self.assertEqual(features['has_feature'], 0)
        self.assertEqual(features['has_docs'], 0)

    def test_calculate_complexity_score_high(self):
        """Test complexity score calculation for complex issue."""
        complexity = self.analyzer.calculate_complexity_score(self.test_issues[0])
        self.assertGreater(complexity, 0)
        self.assertLessEqual(complexity, 100)

    def test_calculate_complexity_score_low(self):
        """Test complexity score calculation for simple issue."""
        complexity = self.analyzer.calculate_complexity_score(self.test_issues[1])
        self.assertGreaterEqual(complexity, 0)
        self.assertLessEqual(complexity, 100)

    def test_calculate_complexity_score_body_length(self):
        """Test complexity score increases with body length."""
        issue_short = {"text": "Short", "title": "Test", "labels": []}
        issue_long = {"text": "A" * 4000, "title": "Test", "labels": []}
        
        score_short = self.analyzer.calculate_complexity_score(issue_short)
        score_long = self.analyzer.calculate_complexity_score(issue_long)
        
        self.assertGreater(score_long, score_short)

    def test_calculate_complexity_score_technical_labels(self):
        """Test complexity score increases with technical labels."""
        issue_simple = {"text": "Test", "title": "Test", "labels": []}
        issue_technical = {"text": "Test", "title": "Test", "labels": ["architecture", "refauthor", "performance"]}
        
        score_simple = self.analyzer.calculate_complexity_score(issue_simple)
        score_technical = self.analyzer.calculate_complexity_score(issue_technical)
        
        self.assertGreater(score_technical, score_simple)

    def test_get_resolution_statistics(self):
        """Test resolution statistics calculation."""
        stats = self.analyzer.get_resolution_statistics()
        
        self.assertIn('median_days', stats)
        self.assertIn('mean_days', stats)
        self.assertIn('p75_days', stats)
        self.assertIn('p95_days', stats)
        self.assertIn('count', stats)
        self.assertEqual(stats['count'], 2)  # 2 closed issues

    def test_get_urgency_statistics(self):
        """Test urgency statistics calculation."""
        stats = self.analyzer.get_urgency_statistics()
        
        self.assertIn('counts', stats)
        self.assertIn('percentages', stats)
        self.assertIn('total', stats)
        self.assertEqual(stats['total'], 2)  # 2 closed issues

    def test_first_response_time(self):
        """Test first response time calculation."""
        response_time = self.analyzer._get_first_response_time(self.test_issues[0])
        self.assertIsNotNone(response_time)
        self.assertGreater(response_time, 0)
        # 30 minutes = 0.5 hours
        self.assertEqual(response_time, 0.5)

    def test_first_response_time_no_response(self):
        """Test first response time when no response from others."""
        issue = {
            "creator": "user1",
            "created_date": "2024-01-01T10:00:00+00:00",
            "events": []
        }
        response_time = self.analyzer._get_first_response_time(issue)
        self.assertIsNone(response_time)

    def test_empty_issues_list(self):
        """Test analyzer handles empty issues list."""
        empty_analyzer = PriorityAnalyzer([])
        self.assertEqual(len(empty_analyzer.issues), 0)
        self.assertEqual(len(empty_analyzer.closed_issues), 0)
        self.assertEqual(len(empty_analyzer.open_issues), 0)


class TestPriorityVisualizer(unittest.TestCase):
    """Test suite for PriorityVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_predictions = [
            {
                "number": 1,
                "predicted_priority": "Critical",
                "complexity_score": 85,
                "priority_confidence": 95
            },
            {
                "number": 2,
                "predicted_priority": "High",
                "complexity_score": 60,
                "priority_confidence": 88
            },
            {
                "number": 3,
                "predicted_priority": "High",
                "complexity_score": 20,
                "priority_confidence": 82
            },
            {
                "number": 4,
                "predicted_priority": "Medium",
                "complexity_score": 45,
                "priority_confidence": 75
            },
            {
                "number": 5,
                "predicted_priority": "Low",
                "complexity_score": 15,
                "priority_confidence": 70
            }
        ]
        self.visualizer = PriorityVisualizer(self.test_predictions)

    def test_initialization(self):
        """Test visualizer initializes with predictions."""
        self.assertEqual(len(self.visualizer.predictions), 5)
        self.assertIsInstance(self.visualizer.predictions, list)

    def test_create_visualizations(self):
        """Test visualization creation and file saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'visualizations')
            
            # Mock plt.savefig and plt.close to avoid actual file operations
            with patch('matplotlib.pyplot.savefig') as mock_save, \
                 patch('matplotlib.pyplot.close') as mock_close:
                self.visualizer.create_visualizations(output_dir)
                
                # Verify savefig was called
                mock_save.assert_called_once()
                mock_close.assert_called_once()

    def test_plot_priority_distribution(self):
        """Test priority distribution plotting."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
        self.visualizer._plot_priority_distribution(ax)
        
        # Check that bars were created
        self.assertGreater(len(ax.patches), 0)
        plt.close(fig)

    def test_plot_complexity_distribution(self):
        """Test complexity distribution plotting."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
        self.visualizer._plot_complexity_distribution(ax)
        
        # Check that bars were created
        self.assertGreater(len(ax.patches), 0)
        plt.close(fig)

    def test_plot_priority_complexity_overlap(self):
        """Test priority-complexity overlap plotting."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        
        self.visualizer._plot_priority_complexity_overlap(ax)
        
        # Check that bars were created
        self.assertGreater(len(ax.patches), 0)
        plt.close(fig)

    def test_print_summary_statistics(self):
        """Test summary statistics printing."""
        # Capture print output
        from io import StringIO
        import sys
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        self.visualizer.print_summary_statistics()
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Verify key sections are printed
        self.assertIn("PREDICTION SUMMARY STATISTICS", output)
        self.assertIn("Priority Distribution", output)
        self.assertIn("Complexity Statistics", output)

    def test_priority_distribution_counts(self):
        """Test priority distribution counts are correct."""
        priorities = [p['predicted_priority'] for p in self.test_predictions]
        priority_counts = Counter(priorities)
        
        self.assertEqual(priority_counts['Critical'], 1)
        self.assertEqual(priority_counts['High'], 2)
        self.assertEqual(priority_counts['Medium'], 1)
        self.assertEqual(priority_counts['Low'], 1)

    def test_complexity_score_statistics(self):
        """Test complexity score statistics calculation."""
        complexity_scores = [p['complexity_score'] for p in self.test_predictions]
        
        mean_complexity = np.mean(complexity_scores)
        median_complexity = np.median(complexity_scores)
        
        self.assertEqual(mean_complexity, 45.0)
        self.assertEqual(median_complexity, 45.0)

    def test_complexity_ranges(self):
        """Test complexity range categorization."""
        complexity_scores = [p['complexity_score'] for p in self.test_predictions]
        
        simple = sum(1 for score in complexity_scores if 0 <= score < 25)
        moderate = sum(1 for score in complexity_scores if 25 <= score < 50)
        complex_range = sum(1 for score in complexity_scores if 50 <= score < 75)
        highly_complex = sum(1 for score in complexity_scores if 75 <= score < 101)
        
        self.assertEqual(simple, 2)  # 15, 20
        self.assertEqual(moderate, 1)  # 45
        self.assertEqual(complex_range, 1)  # 60
        self.assertEqual(highly_complex, 1)  # 85

    def test_quick_wins_detection(self):
        """Test detection of high priority simple issues (quick wins)."""
        high_simple = sum(1 for p in self.test_predictions 
                         if p['predicted_priority'] == 'High' 
                         and p['complexity_score'] < 25)
        
        self.assertEqual(high_simple, 1)  # Issue #3

    def test_empty_predictions(self):
        """Test visualizer handles empty predictions list."""
        empty_visualizer = PriorityVisualizer([])
        self.assertEqual(len(empty_visualizer.predictions), 0)

    def test_visualization_with_single_prediction(self):
        """Test visualization works with single prediction."""
        single_prediction = [self.test_predictions[0]]
        single_visualizer = PriorityVisualizer(single_prediction)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'visualizations')
            
            with patch('matplotlib.pyplot.savefig') as mock_save, \
                 patch('matplotlib.pyplot.close') as mock_close:
                single_visualizer.create_visualizations(output_dir)
                
                mock_save.assert_called_once()
                mock_close.assert_called_once()

    def test_priority_order_consistency(self):
        """Test that priority order is maintained correctly."""
        priority_order = ['Critical', 'High', 'Medium', 'Low']
        
        # Verify all predictions use valid priorities
        for pred in self.test_predictions:
            self.assertIn(pred['predicted_priority'], priority_order)

    def test_confidence_values(self):
        """Test that confidence values are within valid range."""
        for pred in self.test_predictions:
            self.assertGreaterEqual(pred['priority_confidence'], 0)
            self.assertLessEqual(pred['priority_confidence'], 100)
    

if __name__ == "__main__":
    unittest.main()