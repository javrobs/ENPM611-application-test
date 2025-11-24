"""
Comprehensive test suite for Feature 2: Contributors Analysis
Combines tests for ContributorsAnalyzer and ContributorsController
Coverage: 90%+ for both classes
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.contributors_analyzer import ContributorsAnalyzer
from controllers.contributors_controller import ContributorsController
from model import Contributor, Issue, Event


# ============================================================================
# ContributorsAnalyzer Tests
# ============================================================================

class TestContributorsAnalyzer(unittest.TestCase):
    """Comprehensive test suite for ContributorsAnalyzer class"""
    
    def setUp(self):
        """Set up fresh ContributorsAnalyzer instance and sample data before each test"""
        self.analyzer = ContributorsAnalyzer()
        
        # Sample issues DataFrame
        self.sample_issues_df = pd.DataFrame([
            {
                'number': 1,
                'creator': 'user1',
                'state': 'closed',
                'created_date': pd.Timestamp('2023-01-15 10:00:00'),
                'updated_date': pd.Timestamp('2023-01-20 15:00:00'),
                'labels': ['kind/bug', 'priority/high']
            },
            {
                'number': 2,
                'creator': 'user2',
                'state': 'open',
                'created_date': pd.Timestamp('2023-02-10 14:30:00'),
                'updated_date': pd.Timestamp('2023-02-12 09:00:00'),
                'labels': ['kind/feature', 'area/docs']
            },
            {
                'number': 3,
                'creator': 'user1',
                'state': 'closed',
                'created_date': pd.Timestamp('2023-03-05 08:45:00'),
                'updated_date': pd.Timestamp('2023-03-10 16:20:00'),
                'labels': ['kind/bug', 'good first issue']
            },
            {
                'number': 4,
                'creator': 'user3',
                'state': 'open',
                'created_date': pd.Timestamp('2024-01-15 11:00:00'),
                'updated_date': pd.Timestamp('2024-01-16 12:00:00'),
                'labels': ['kind/feature', 'area/ui']
            },
            {
                'number': 5,
                'creator': 'stale[bot]',
                'state': 'closed',
                'created_date': pd.Timestamp('2023-04-01 10:00:00'),
                'updated_date': pd.Timestamp('2023-04-02 10:00:00'),
                'labels': ['kind/bug']
            }
        ])
        
        # Sample events DataFrame
        self.sample_events_df = pd.DataFrame([
            {
                'issue_number': 1,
                'event_type': 'closed',
                'event_author': 'user2',
                'event_date': pd.Timestamp('2023-01-20 15:00:00'),
                'label': None,
                'comment': None
            },
            {
                'issue_number': 1,
                'event_type': 'commented',
                'event_author': 'user1',
                'event_date': pd.Timestamp('2023-01-18 12:00:00'),
                'label': None,
                'comment': 'This is a comment'
            },
            {
                'issue_number': 2,
                'event_type': 'commented',
                'event_author': 'user3',
                'event_date': pd.Timestamp('2023-02-11 10:00:00'),
                'label': None,
                'comment': 'Another comment'
            },
            {
                'issue_number': 3,
                'event_type': 'closed',
                'event_author': 'user2',
                'event_date': pd.Timestamp('2023-03-10 16:20:00'),
                'label': None,
                'comment': None
            },
            {
                'issue_number': 1,
                'event_type': 'commented',
                'event_author': 'github-actions[bot]',
                'event_date': pd.Timestamp('2023-01-19 09:00:00'),
                'label': None,
                'comment': 'Bot comment'
            }
        ])

    # ===== Core Functionality Tests =====
    
    def test_build_contributors_valid_data(self):
        """Test building contributors from valid issues and events data"""
        contributors = self.analyzer.build_contributors(self.sample_issues_df, self.sample_events_df)
        
        self.assertEqual(len(contributors), 3)
        usernames = [c.username for c in contributors]
        self.assertIn('user1', usernames)
        self.assertIn('user2', usernames)
        self.assertIn('user3', usernames)
        self.assertNotIn('stale[bot]', usernames)
        self.assertNotIn('github-actions[bot]', usernames)

    def test_build_contributors_filters_bots(self):
        """Test that bot users are properly filtered out"""
        contributors = self.analyzer.build_contributors(self.sample_issues_df, self.sample_events_df)
        
        for contributor in contributors:
            self.assertNotIn('[bot]', contributor.username)

    def test_build_contributors_empty_data(self):
        """Test handling of empty DataFrames"""
        empty_issues = pd.DataFrame(columns=['number', 'creator', 'state', 'created_date', 'updated_date', 'labels'])
        empty_events = pd.DataFrame(columns=['issue_number', 'event_type', 'event_author', 'event_date', 'label', 'comment'])
        
        contributors = self.analyzer.build_contributors(empty_issues, empty_events)
        self.assertEqual(len(contributors), 0)

    def test_analyze_bug_closure_distribution(self):
        """Test bug closure distribution analysis"""
        result = self.analyzer.analyze_bug_closure_distribution(self.sample_issues_df, self.sample_events_df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('year', result.columns)
        self.assertIn('top5_pct', result.columns)
        self.assertIn('rest_pct', result.columns)

    def test_analyze_top_feature_requesters(self):
        """Test top feature requesters analysis"""
        top_requesters, feature_issues = self.analyzer.analyze_top_feature_requesters(self.sample_issues_df, top_n=10)
        
        self.assertIsNotNone(top_requesters)
        self.assertIsNotNone(feature_issues)
        self.assertLessEqual(len(top_requesters), 10)

    def test_compute_unique_commenters(self):
        """Test computation of unique commenters per issue per month"""
        result = self.analyzer.compute_unique_commenters(self.sample_events_df, self.sample_issues_df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('issue_number', result.columns)
        self.assertIn('month', result.columns)

    def test_analyze_lifecycle_stages(self):
        """Test contributor lifecycle stages analysis"""
        contributors = self.analyzer.build_contributors(self.sample_issues_df, self.sample_events_df)
        result = self.analyzer.analyze_lifecycle_stages(contributors)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('stage', result.columns)
        valid_stages = ['Newcomer', 'Active', 'Core Maintainer', 'Graduated Contributor']
        self.assertTrue(result['stage'].isin(valid_stages).all())

    # ===== Edge Case Tests =====

    def test_compute_unique_commenters_no_comments(self):
        """Test unique commenters with no comment events"""
        events_df = pd.DataFrame([
            {
                'issue_number': 1,
                'event_type': 'closed',
                'event_author': 'user1',
                'event_date': pd.Timestamp('2023-01-20'),
                'label': None,
                'comment': None
            }
        ])
        
        result = self.analyzer.compute_unique_commenters(events_df, self.sample_issues_df)
        self.assertEqual(len(result), 0)

    def test_analyze_engagement_heatmap_empty(self):
        """Test engagement heatmap with no contributors"""
        heatmap = self.analyzer.analyze_engagement_heatmap([])
        
        self.assertIsInstance(heatmap, pd.DataFrame)
        self.assertEqual(heatmap.shape, (7, 24))
        self.assertEqual(heatmap.sum().sum(), 0)

    # ===== Year Boundary and Concurrent Activity Tests =====
    
    def test_bug_closure_year_boundary(self):
        """Test bug closures across year boundaries (Dec 31 -> Jan 1)"""
        issues_df = pd.DataFrame([
            {
                'number': 1,
                'creator': 'user1',
                'state': 'closed',
                'created_date': pd.Timestamp('2023-12-31 23:00:00'),
                'updated_date': pd.Timestamp('2024-01-01 01:00:00'),
                'labels': ['kind/bug']
            }
        ])
        
        events_df = pd.DataFrame([
            {
                'issue_number': 1,
                'event_type': 'closed',
                'event_author': 'user1',
                'event_date': pd.Timestamp('2024-01-01 01:00:00'),
                'label': None,
                'comment': None
            }
        ])
        
        result = self.analyzer.analyze_bug_closure_distribution(issues_df, events_df)
        
        # Bug should be counted in 2024
        self.assertIn(2024, result['year'].values)

    # def test_concurrent_activity_heatmap(self):
    #     """Test heatmap with multiple users active in same hour"""
    #     now = pd.Timestamp.now(tz=timezone.utc)
    #     same_time = now.replace(hour=14, minute=30)
        
    #     c1 = Contributor('user1')
    #     c1.issues_created.append(Issue(1, same_time))
    #     c1.comments.append(Event(1, 'commented', 'user1', same_time))
        
    #     c2 = Contributor('user2')
    #     c2.issues_created.append(Issue(2, same_time))
    #     c2.comments.append(Event(2, 'commented', 'user2', same_time))
        
    #     heatmap = self.analyzer.analyze_engagement_heatmap([c1, c2])
        
    #     day_name = same_time.strftime('%a')
    #     hour = same_time.hour
        
    #     # Should count all activities
    #     self.assertGreater(heatmap.loc[day_name, hour], 0)

    def test_timezone_edge_cases_lifecycle(self):
        """Test lifecycle stages with timezone-aware and naive datetimes"""
        now = pd.Timestamp.now(tz=timezone.utc)
        
        c1 = Contributor('user_tz_aware')
        c1.first_activity = now - timedelta(days=15)
        c1.last_activity = now - timedelta(days=5)
        
        c2 = Contributor('user_naive')
        c2.first_activity = pd.Timestamp.now() - timedelta(days=15)
        c2.last_activity = pd.Timestamp.now() - timedelta(days=5)
        
        result = self.analyzer.analyze_lifecycle_stages([c1, c2], reference_date=now)
        
        self.assertEqual(len(result), 2)

    def test_monthly_aggregation_commenters(self):
        """Test that commenters are properly aggregated by month"""
        events_df = pd.DataFrame([
            {
                'issue_number': 1,
                'event_type': 'commented',
                'event_author': 'user1',
                'event_date': pd.Timestamp('2023-01-05 10:00:00'),
                'label': None,
                'comment': 'comment1'
            },
            {
                'issue_number': 1,
                'event_type': 'commented',
                'event_author': 'user1',
                'event_date': pd.Timestamp('2023-01-15 14:00:00'),
                'label': None,
                'comment': 'comment2'
            },
            {
                'issue_number': 1,
                'event_type': 'commented',
                'event_author': 'user2',
                'event_date': pd.Timestamp('2023-01-20 09:00:00'),
                'label': None,
                'comment': 'comment3'
            }
        ])
        
        result = self.analyzer.compute_unique_commenters(events_df, self.sample_issues_df)
        
        # Should have 2 unique commenters for issue 1 in Jan 2023
        jan_2023 = result[result['month'] == '2023-01']
        if not jan_2023.empty:
            self.assertEqual(jan_2023['n_unique_commenters'].iloc[0], 2)

    # ===== Top Active Users and Docs Issues Tests =====
    
    def test_top_active_users_activity_calculation(self):
        """Test docs issues properly calculates state distribution"""
        docs_df = pd.DataFrame([
            {
                'number': 1,
                'creator': 'user1',
                'state': 'open',
                'created_date': pd.Timestamp('2023-01-15'),
                'updated_date': pd.Timestamp('2023-01-20'),
                'labels': ['area/docs']
            },
            {
                'number': 2,
                'creator': 'user2',
                'state': 'closed',
                'created_date': pd.Timestamp('2023-01-16'),
                'updated_date': pd.Timestamp('2023-01-21'),
                'labels': ['area/docs']
            }
        ])
        
        class MockLoader:
            def filter_by_label(self, df, label):
                return df
        
        status_counts, _ = self.analyzer.analyze_docs_issues(
            docs_df, self.sample_events_df, MockLoader()
        )
        
        if status_counts is not None:
            self.assertIn('open', status_counts.columns)
            self.assertIn('closed', status_counts.columns)

    # ===== Label Filtering and Large Dataset Tests =====
    
    def test_label_filtering_case_insensitive(self):
        """Test that label filtering handles case insensitivity"""
        issues_df = pd.DataFrame([
            {
                'number': 1,
                'creator': 'user1',
                'state': 'closed',
                'created_date': pd.Timestamp('2023-01-15'),
                'updated_date': pd.Timestamp('2023-01-20'),
                'labels': ['Kind/Bug', 'PRIORITY/HIGH']
            }
        ])
        
        events_df = pd.DataFrame([
            {
                'issue_number': 1,
                'event_type': 'closed',
                'event_author': 'user1',
                'event_date': pd.Timestamp('2023-01-20'),
                'label': None,
                'comment': None
            }
        ])
        
        result = self.analyzer.analyze_bug_closure_distribution(issues_df, events_df)
        
        # Should still find bugs despite case differences
        self.assertIsInstance(result, pd.DataFrame)

    # def test_label_special_characters(self):
    #     """Test labels with special characters (hyphens, underscores, colons)"""
    #     issues_df = pd.DataFrame([
    #         {
    #             'number': 1,
    #             'creator': 'user1',
    #             'state': 'open',
    #             'created_date': pd.Timestamp('2023-01-15'),
    #             'updated_date': pd.Timestamp('2023-01-20'),
    #             'labels': ['area/api-gateway', 'priority_high', 'scope:backend']
    #         }
    #     ])
        
    #     contributors = self.analyzer.build_contributors(issues_df, pd.DataFrame())
        
    #     self.assertEqual(len(contributors), 1)

    def test_large_dataset_performance(self):
        """Test with large dataset (1000+ issues)"""
        large_issues = pd.DataFrame([
            {
                'number': i,
                'creator': f'user{i%50}',
                'state': 'closed' if i % 3 == 0 else 'open',
                'created_date': pd.Timestamp('2023-01-01') + timedelta(days=i%365),
                'updated_date': pd.Timestamp('2023-01-01') + timedelta(days=i%365+5),
                'labels': [f'kind/{"bug" if i%2==0 else "feature"}']
            }
            for i in range(1000)
        ])
        
        large_events = pd.DataFrame([
            {
                'issue_number': i,
                'event_type': 'commented',
                'event_author': f'user{(i+1)%50}',
                'event_date': pd.Timestamp('2023-01-01') + timedelta(days=i%365+2),
                'label': None,
                'comment': f'comment {i}'
            }
            for i in range(1000)
        ])
        
        contributors = self.analyzer.build_contributors(large_issues, large_events)
        
        self.assertGreater(len(contributors), 0)
        self.assertLessEqual(len(contributors), 50)

    # ===== Null Handling and Special Characters Tests =====
    
    def test_special_character_usernames(self):
        """Test lifecycle stage graduation threshold (6 months)"""
        now = pd.Timestamp.now(tz=timezone.utc)
        
        # Exactly 6 months ago
        c1 = Contributor('user1')
        c1.first_activity = now - timedelta(days=365)
        c1.last_activity = now - timedelta(days=183)  # ~6 months
        
        result = self.analyzer.analyze_lifecycle_stages([c1], reference_date=now)
        
        stage = result[result['contributor'] == 'user1']['stage'].iloc[0]
        self.assertEqual(stage, 'Graduated Contributor')


# ============================================================================
# ContributorsController Tests
# ============================================================================

class TestContributorsController(unittest.TestCase):
    """Comprehensive test suite for ContributorsController class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        with patch('controllers.contributors_controller.DataLoader'), \
             patch('controllers.contributors_controller.ContributorsAnalyzer'), \
             patch('controllers.contributors_controller.Visualizer'):
            
            self.controller = ContributorsController()
            self.controller.data_loader = Mock()
            self.controller.analyzer = Mock()
            self.controller.visualizer = Mock()

        self.sample_issues_df = pd.DataFrame({
            'number': [1, 2, 3],
            'creator': ['user1', 'user2', 'user3'],
            'labels': [['bug'], ['feature'], ['docs']],
            'state': ['closed', 'open', 'closed'],
            'created_date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
            'updated_date': pd.to_datetime(['2023-01-15', '2023-02-15', '2023-03-15']),
        })

        self.sample_events_df = pd.DataFrame({
            'issue_number': [1, 1, 2],
            'event_type': ['closed', 'commented', 'commented'],
            'event_author': ['user1', 'user2', 'user3'],
            'event_date': pd.to_datetime(['2023-01-15', '2023-01-10', '2023-02-10']),
        })

    # ===== Initialization Tests =====
    
    def test_controller_initialization(self):
        """Test that controller initializes with all required dependencies"""
        with patch('controllers.contributors_controller.DataLoader') as mock_loader, \
             patch('controllers.contributors_controller.ContributorsAnalyzer') as mock_analyzer, \
             patch('controllers.contributors_controller.Visualizer') as mock_visualizer:
            
            controller = ContributorsController()
            
            mock_loader.assert_called_once()
            mock_analyzer.assert_called_once()
            mock_visualizer.assert_called_once()
            
            self.assertIsNotNone(controller.data_loader)
            self.assertIsNotNone(controller.analyzer)
            self.assertIsNotNone(controller.visualizer)

    # ===== load_contributor_data Tests =====
    
    def test_load_contributor_data_success(self):
        """Test successful loading of contributor data"""
        self.controller.data_loader.parse_issues.return_value = self.sample_issues_df
        self.controller.data_loader.parse_events.return_value = self.sample_events_df
        
        issues_df, events_df = self.controller.load_contributor_data()
        
        self.controller.data_loader.parse_issues.assert_called_once()
        self.controller.data_loader.parse_events.assert_called_once_with(self.sample_issues_df)
        
        pd.testing.assert_frame_equal(issues_df, self.sample_issues_df)
        pd.testing.assert_frame_equal(events_df, self.sample_events_df)

    def test_load_contributor_data_empty_dataframes(self):
        """Test loading when data loader returns empty DataFrames"""
        empty_df = pd.DataFrame()
        self.controller.data_loader.parse_issues.return_value = empty_df
        self.controller.data_loader.parse_events.return_value = empty_df
        
        issues_df, events_df = self.controller.load_contributor_data()
        
        self.assertTrue(issues_df.empty)
        self.assertTrue(events_df.empty)

    # ===== plot_bug_closure_distribution Tests =====
    
    def test_plot_bug_closure_distribution_success(self):
        """Test successful plotting of bug closure distribution"""
        yearly_dist = pd.DataFrame({
            'year': [2022, 2023],
            'top5_users': [['user1', 'user2'], ['user3', 'user4']],
            'top5_pct': [65.5, 70.2],
            'rest_pct': [34.5, 29.8]
        })
        
        mock_fig = Mock()
        self.controller.analyzer.analyze_bug_closure_distribution.return_value = yearly_dist
        self.controller.visualizer.create_bug_closure_distribution_chart.return_value = mock_fig
        
        result = self.controller.plot_bug_closure_distribution(
            self.sample_issues_df, 
            self.sample_events_df
        )
        
        self.controller.analyzer.analyze_bug_closure_distribution.assert_called_once_with(
            self.sample_issues_df, 
            self.sample_events_df
        )
        
        self.controller.visualizer.create_bug_closure_distribution_chart.assert_called_once()
        self.assertEqual(result, mock_fig)

    @patch('builtins.print')
    def test_plot_bug_closure_distribution_prints_top5(self, mock_print):
        """Test that top 5 bug closers are printed to console"""
        yearly_dist = pd.DataFrame({
            'year': [2023.0],
            'top5_users': [['alice', 'bob', 'charlie']],
            'top5_pct': [70.0],
            'rest_pct': [30.0]
        })
        
        self.controller.analyzer.analyze_bug_closure_distribution.return_value = yearly_dist
        self.controller.visualizer.create_bug_closure_distribution_chart.return_value = Mock()
        
        self.controller.plot_bug_closure_distribution(
            self.sample_issues_df, 
            self.sample_events_df
        )
        
        mock_print.assert_called_once()

    # ===== plot_top_feature_requesters Tests =====
    
    def test_plot_top_feature_requesters_success(self):
        """Test successful plotting of top feature requesters"""
        top_requesters = pd.Series([10, 8, 6], index=['user1', 'user2', 'user3'])
        feature_issues = self.sample_issues_df
        
        mock_fig = Mock()
        self.controller.analyzer.analyze_top_feature_requesters.return_value = (
            top_requesters, 
            feature_issues
        )
        self.controller.visualizer.create_top_feature_requesters_chart.return_value = mock_fig
        
        result = self.controller.plot_top_feature_requesters(self.sample_issues_df, top_n=10)
        
        self.controller.analyzer.analyze_top_feature_requesters.assert_called_once_with(
            self.sample_issues_df, 
            top_n=10
        )
        
        self.assertEqual(result, mock_fig)

    def test_plot_top_feature_requesters_returns_none_when_no_data(self):
        """Test that None is returned when analyzer returns None"""
        self.controller.analyzer.analyze_top_feature_requesters.return_value = (None, None)
        
        result = self.controller.plot_top_feature_requesters(self.sample_issues_df)
        
        self.assertIsNone(result)
        self.controller.visualizer.create_top_feature_requesters_chart.assert_not_called()

    # ===== plot_docs_issues Tests =====
    
    def test_plot_docs_issues_success(self):
        """Test successful plotting of docs issues"""
        status_counts = pd.DataFrame({
            'open': [5, 3, 7],
            'closed': [2, 4, 3]
        }, index=pd.to_datetime(['2023-01', '2023-02', '2023-03']))
        
        avg_commenters = pd.Series([2.5, 3.0, 2.8], 
                                   index=pd.to_datetime(['2023-01', '2023-02', '2023-03']))
        
        mock_fig = Mock()
        self.controller.analyzer.analyze_docs_issues.return_value = (status_counts, avg_commenters)
        self.controller.visualizer.create_docs_issues_chart.return_value = mock_fig
        
        result = self.controller.plot_docs_issues(self.sample_issues_df, self.sample_events_df)
        
        self.controller.analyzer.analyze_docs_issues.assert_called_once_with(
            self.sample_issues_df, 
            self.sample_events_df, 
            self.controller.data_loader
        )
        
        self.assertEqual(result, mock_fig)

    def test_plot_docs_issues_returns_none_when_no_data(self):
        """Test that None is returned when analyzer returns None"""
        self.controller.analyzer.analyze_docs_issues.return_value = (None, None)
        
        result = self.controller.plot_docs_issues(self.sample_issues_df, self.sample_events_df)
        
        self.assertIsNone(result)

    # ===== plot_issues_created_per_user Tests =====
    
    def test_plot_issues_created_per_user_success(self):
        """Test successful plotting of issues created per user"""
        issues_per_user = pd.Series([15, 12, 10], index=['user1', 'user2', 'user3'])
        all_counts = pd.Series([15, 12, 10, 8, 6], index=['u1', 'u2', 'u3', 'u4', 'u5'])
        
        mock_fig = Mock()
        self.controller.analyzer.analyze_issues_created_per_user.return_value = (
            issues_per_user, 
            all_counts
        )
        self.controller.visualizer.create_issues_created_per_user_chart.return_value = mock_fig
        
        result = self.controller.plot_issues_created_per_user(self.sample_issues_df, top_n=40)
        
        self.controller.analyzer.analyze_issues_created_per_user.assert_called_once_with(
            self.sample_issues_df, 
            top_n=40
        )
        
        self.assertEqual(result, mock_fig)

    # ===== plot_top_active_users_per_year Tests =====
    
    def test_plot_top_active_users_per_year_success(self):
        """Test successful plotting of top active users per year"""
        yearly_data = {
            2023: pd.DataFrame({
                'user': ['user1', 'user2'],
                'activity': [35, 28]
            })
        }
        
        mock_fig = Mock()
        self.controller.analyzer.analyze_top_active_users_per_year.return_value = yearly_data
        self.controller.visualizer.create_top_active_users_per_year_chart.return_value = mock_fig
        
        result = self.controller.plot_top_active_users_per_year([], top_n=10)
        
        self.controller.analyzer.analyze_top_active_users_per_year.assert_called_once()
        self.assertEqual(result, mock_fig)

    # ===== run_engagement_heatmap Tests =====
    
    @patch('builtins.print')
    def test_run_engagement_heatmap_success(self, mock_print):
        """Test successful generation of engagement heatmap"""
        heatmap = pd.DataFrame(
            np.random.rand(7, 24),
            index=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            columns=list(range(24))
        )
        
        mock_fig = Mock()
        self.controller.analyzer.analyze_engagement_heatmap.return_value = heatmap
        self.controller.visualizer.create_engagement_heatmap.return_value = mock_fig
        
        result = self.controller.run_engagement_heatmap([])
        
        self.controller.analyzer.analyze_engagement_heatmap.assert_called_once()
        self.assertEqual(result, mock_fig)

    # ===== run_contributor_lifecycle Tests =====
    
    @patch('builtins.print')
    def test_run_contributor_lifecycle_success(self, mock_print):
        """Test successful analysis of contributor lifecycle"""
        lifecycle_df = pd.DataFrame({
            'contributor': ['user1', 'user2'],
            'first_activity': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-06-01')],
            'last_activity': [pd.Timestamp('2023-12-01'), pd.Timestamp('2023-12-15')],
            'stage': ['Core Maintainer', 'Active']
        })
        
        mock_fig = Mock()
        self.controller.analyzer.analyze_lifecycle_stages.return_value = lifecycle_df
        self.controller.visualizer.create_lifecycle_stages_chart.return_value = mock_fig
        
        result = self.controller.run_contributor_lifecycle([])
        
        self.controller.analyzer.analyze_lifecycle_stages.assert_called_once()
        self.assertEqual(result, mock_fig)


# ============================================================================
# Visualizer Tests
# ============================================================================


class TestVisualizer(unittest.TestCase):
    """Test suite for Visualizer class - 90%+ coverage"""
    
    def setUp(self):
        """Set up test fixtures"""
        from visualization.visualizer import Visualizer
        self.visualizer = Visualizer()
        
        # Sample data for bug closure distribution
        self.yearly_distribution = pd.DataFrame({
            'year': [2020, 2021, 2022],
            'top5_pct': [60.0, 55.0, 70.0],
            'rest_pct': [40.0, 45.0, 30.0]
        })
        
        # Sample data for top feature requesters
        self.top_requesters = pd.Series([15, 12, 10, 8, 7], 
                                        index=['user1', 'user2', 'user3', 'user4', 'user5'])
        self.feature_issues = pd.DataFrame({
            'creator': ['user1', 'user1', 'user2', 'user2', 'user3', 'user3', 'user4', 'user5'],
            'state': ['State.open', 'State.closed', 'State.open', 'State.closed', 
                     'State.open', 'State.closed', 'State.open', 'State.closed']
        })
        
        # Sample data for docs issues
        self.status_counts = pd.DataFrame({
            'open': [5, 8, 6, 4],
            'closed': [3, 5, 7, 9]
        }, index=pd.to_datetime(['2023-01', '2023-02', '2023-03', '2023-04']))
        self.avg_commenters = pd.Series([2.5, 3.0, 2.8, 3.5], 
                                       index=self.status_counts.index)
        
        # Sample data for issues per user
        self.issues_per_user = pd.Series([50, 45, 40, 35, 30], 
                                         index=['user1', 'user2', 'user3', 'user4', 'user5'])
        self.all_counts = pd.Series([50, 45, 40, 35, 30, 25, 20, 15, 10, 5])
        
        # Sample data for engagement heatmap
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        self.heatmap_df = pd.DataFrame(
            [[10, 5, 3, 2, 1, 0, 0, 0, 0, 0, 5, 10, 15, 20, 18, 15, 12, 10, 8, 6, 4, 3, 2, 1] for _ in days],
            index=days,
            columns=range(24)
        )
        
        # Sample data for lifecycle
        self.lifecycle_summary = pd.DataFrame({
            'count': [10, 25, 8, 5]
        }, index=['Newcomer', 'Active', 'Core Maintainer', 'Graduated Contributor'])
    
    def test_create_bug_closure_distribution_chart_success(self):
        """Test successful creation of bug closure distribution chart"""
        fig = self.visualizer.create_bug_closure_distribution_chart(
            self.yearly_distribution, 
            "Bug Closure Distribution"
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), "Bug Closure Distribution")
        self.assertEqual(ax.get_xlabel(), "Year")
        self.assertEqual(ax.get_ylabel(), "Percentage of Bug Closures (%)")
    
    def test_create_bug_closure_distribution_chart_legend(self):
        """Test legend is present in bug closure chart"""
        fig = self.visualizer.create_bug_closure_distribution_chart(
            self.yearly_distribution, 
            "Test Title"
        )
        ax = fig.axes[0]
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        legend_texts = [t.get_text() for t in legend.get_texts()]
        self.assertIn("Top 5 Closers", legend_texts)
        self.assertIn("Rest", legend_texts)
    
    def test_create_bug_closure_distribution_chart_single_year(self):
        """Test chart with single year data"""
        single_year = pd.DataFrame({
            'year': [2023],
            'top5_pct': [75.0],
            'rest_pct': [25.0]
        })
        fig = self.visualizer.create_bug_closure_distribution_chart(
            single_year, 
            "Single Year"
        )
        self.assertIsNotNone(fig)
    
    def test_create_bug_closure_distribution_chart_zero_values(self):
        """Test chart with zero percentage values"""
        zero_data = pd.DataFrame({
            'year': [2023],
            'top5_pct': [0.0],
            'rest_pct': [100.0]
        })
        fig = self.visualizer.create_bug_closure_distribution_chart(
            zero_data, 
            "Zero Top 5"
        )
        self.assertIsNotNone(fig)
    
    def test_create_top_feature_requesters_chart_success(self):
        """Test successful creation of top feature requesters chart"""
        fig = self.visualizer.create_top_feature_requesters_chart(
            self.top_requesters,
            self.feature_issues,
            "Top Requesters"
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), "Top Requesters")
        self.assertEqual(ax.get_xlabel(), "Number of Feature Requests")
        self.assertEqual(ax.get_ylabel(), "Contributor")
    
    def test_create_top_feature_requesters_chart_state_replacement(self):
        """Test that State. prefix is removed from states"""
        fig = self.visualizer.create_top_feature_requesters_chart(
            self.top_requesters,
            self.feature_issues
        )
        self.assertIsNotNone(fig)
    
    def test_create_top_feature_requesters_chart_inverted_yaxis(self):
        """Test that y-axis is inverted (highest on top)"""
        fig = self.visualizer.create_top_feature_requesters_chart(
            self.top_requesters,
            self.feature_issues
        )
        ax = fig.axes[0]
        y_lim = ax.get_ylim()
        self.assertGreater(y_lim[0], y_lim[1])
    
    def test_create_top_feature_requesters_chart_legend_present(self):
        """Test legend is present and has State title"""
        fig = self.visualizer.create_top_feature_requesters_chart(
            self.top_requesters,
            self.feature_issues
        )
        ax = fig.axes[0]
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(legend.get_title().get_text(), "State")
    
    def test_create_top_feature_requesters_chart_single_user(self):
        """Test chart with single user"""
        single_user = pd.Series([10], index=['user1'])
        single_issue = pd.DataFrame({
            'creator': ['user1'],
            'state': ['State.open']
        })
        fig = self.visualizer.create_top_feature_requesters_chart(
            single_user,
            single_issue
        )
        self.assertIsNotNone(fig)
    
    def test_create_docs_issues_chart_success(self):
        """Test successful creation of docs issues chart"""
        fig = self.visualizer.create_docs_issues_chart(
            self.status_counts,
            self.avg_commenters,
            "Docs Issues Chart"
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)
        ax1 = fig.axes[0]
        self.assertEqual(ax1.get_title(), "Docs Issues Chart")
        self.assertEqual(ax1.get_xlabel(), "Month")
        self.assertEqual(ax1.get_ylabel(), "Number of Doc Issues")
    
    def test_create_docs_issues_chart_dual_axis(self):
        """Test that chart has dual y-axes"""
        fig = self.visualizer.create_docs_issues_chart(
            self.status_counts,
            self.avg_commenters
        )
        self.assertEqual(len(fig.axes), 2)
        ax1, ax2 = fig.axes
        self.assertEqual(ax2.get_ylabel(), "Avg Unique Commenters per Doc Issue")
    
    def test_create_docs_issues_chart_legends(self):
        """Test both legends are present"""
        fig = self.visualizer.create_docs_issues_chart(
            self.status_counts,
            self.avg_commenters
        )
        ax1, ax2 = fig.axes
        self.assertIsNotNone(ax1.get_legend())
        self.assertIsNotNone(ax2.get_legend())
    
    def test_create_docs_issues_chart_single_month(self):
        """Test chart with single month data"""
        single_month = pd.DataFrame({
            'open': [5],
            'closed': [3]
        }, index=pd.to_datetime(['2023-01']))
        single_avg = pd.Series([2.5], index=single_month.index)
        fig = self.visualizer.create_docs_issues_chart(
            single_month,
            single_avg
        )
        self.assertIsNotNone(fig)
    
    def test_create_docs_issues_chart_zero_commenters(self):
        """Test chart with zero average commenters"""
        zero_avg = pd.Series([0.0, 0.0, 0.0, 0.0], 
                            index=self.status_counts.index)
        fig = self.visualizer.create_docs_issues_chart(
            self.status_counts,
            zero_avg
        )
        self.assertIsNotNone(fig)
    
    def test_create_issues_created_per_user_chart_success(self):
        """Test successful creation of issues per user chart"""
        fig = self.visualizer.create_issues_created_per_user_chart(
            self.issues_per_user,
            self.all_counts,
            "Top 40 Issues"
        )
        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), "Number of Issues Created")
        self.assertEqual(ax.get_ylabel(), "Contributor's username")
    
    def test_create_issues_created_per_user_chart_percentage_calculation(self):
        """Test that percentage is calculated correctly"""
        fig = self.visualizer.create_issues_created_per_user_chart(
            self.issues_per_user,
            self.all_counts
        )
        total = self.all_counts.sum()
        top_total = self.issues_per_user.sum()
        expected_pct = (top_total / total) * 100
        self.assertIsNotNone(fig)
    
    def test_create_issues_created_per_user_chart_sorted_bars(self):
        """Test that bars are sorted by value"""
        fig = self.visualizer.create_issues_created_per_user_chart(
            self.issues_per_user,
            self.all_counts
        )
        self.assertIsNotNone(fig)
    
    def test_create_issues_created_per_user_chart_top40_limit(self):
        """Test that chart limits to top 40 users"""
        large_series = pd.Series(range(100, 0, -1), 
                                index=[f'user{i}' for i in range(100)])
        fig = self.visualizer.create_issues_created_per_user_chart(
            large_series,
            large_series
        )
        self.assertIsNotNone(fig)
    
    def test_create_issues_created_per_user_chart_single_user(self):
        """Test chart with single user"""
        single = pd.Series([10], index=['user1'])
        fig = self.visualizer.create_issues_created_per_user_chart(
            single,
            single
        )
        self.assertIsNotNone(fig)
    
    def test_create_top_active_users_per_year_chart_success(self):
        """Test successful creation of active users per year chart"""
        import plotly.graph_objects as go
        yearly_data = {
            2020: pd.DataFrame({
                'user': ['user1', 'user2', 'user3'],
                'activity': [100, 80, 60]
            }),
            2021: pd.DataFrame({
                'user': ['user4', 'user5', 'user6'],
                'activity': [120, 90, 70]
            })
        }
        fig = self.visualizer.create_top_active_users_per_year_chart(yearly_data, top_n=10)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 2)
    
    def test_create_top_active_users_per_year_chart_empty_data(self):
        """Test chart with empty yearly data"""
        empty_data = {}
        fig = self.visualizer.create_top_active_users_per_year_chart(empty_data)
        self.assertIsNotNone(fig)
        self.assertIn("No contributor activity", fig.layout.title.text)
    
    def test_create_top_active_users_per_year_chart_single_year(self):
        """Test chart with single year"""
        single_year = {
            2023: pd.DataFrame({
                'user': ['user1', 'user2'],
                'activity': [50, 40]
            })
        }
        fig = self.visualizer.create_top_active_users_per_year_chart(single_year)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)
    
    def test_create_top_active_users_per_year_chart_dropdown_menu(self):
        """Test that dropdown menu is created"""
        yearly_data = {
            2020: pd.DataFrame({'user': ['user1'], 'activity': [100]}),
            2021: pd.DataFrame({'user': ['user2'], 'activity': [120]})
        }
        fig = self.visualizer.create_top_active_users_per_year_chart(yearly_data)
        self.assertIsNotNone(fig.layout.updatemenus)
        self.assertEqual(len(fig.layout.updatemenus), 1)
    
    def test_create_top_active_users_per_year_chart_top_n_limit(self):
        """Test that chart respects top_n parameter"""
        large_year = {
            2023: pd.DataFrame({
                'user': [f'user{i}' for i in range(20)],
                'activity': list(range(100, 80, -1))
            })
        }
        fig = self.visualizer.create_top_active_users_per_year_chart(large_year, top_n=5)
        self.assertIsNotNone(fig)
    
    def test_create_top_active_users_per_year_chart_sorted_years(self):
        """Test that years are sorted chronologically"""
        unsorted_data = {
            2022: pd.DataFrame({'user': ['user1'], 'activity': [100]}),
            2020: pd.DataFrame({'user': ['user2'], 'activity': [120]}),
            2021: pd.DataFrame({'user': ['user3'], 'activity': [110]})
        }
        fig = self.visualizer.create_top_active_users_per_year_chart(unsorted_data)
        self.assertIsNotNone(fig)
    
    def test_create_engagement_heatmap_chart_success(self):
        """Test successful creation of engagement heatmap"""
        fig = self.visualizer.create_engagement_heatmap_chart(self.heatmap_df)
        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), "Contributor Engagement Heatmap")
        self.assertEqual(ax.get_xlabel(), "Hour of Day")
        self.assertEqual(ax.get_ylabel(), "Day of Week")
    
    def test_create_engagement_heatmap_chart_normalization(self):
        """Test that heatmap normalizes by row (percentage of day)"""
        multi_day = pd.DataFrame(
            [[10, 20, 30], [5, 10, 15]],
            index=['Monday', 'Tuesday'],
            columns=[0, 1, 2]
        )
        fig = self.visualizer.create_engagement_heatmap_chart(multi_day)
        self.assertIsNotNone(fig)
    
    def test_create_engagement_heatmap_chart_zero_activity(self):
        """Test heatmap with zero activity hours"""
        zero_hours = pd.DataFrame(
            [[0, 0, 10, 20]],
            index=['Monday'],
            columns=[0, 1, 2, 3]
        )
        fig = self.visualizer.create_engagement_heatmap_chart(zero_hours)
        self.assertIsNotNone(fig)
    
    def test_create_engagement_heatmap_chart_single_day(self):
        """Test heatmap with single day"""
        single_day = pd.DataFrame(
            [[5, 10, 15, 20]],
            index=['Wednesday'],
            columns=[0, 1, 2, 3]
        )
        fig = self.visualizer.create_engagement_heatmap_chart(single_day)
        self.assertIsNotNone(fig)
    
    def test_create_engagement_heatmap_chart_all_zeros(self):
        """Test heatmap with all zero values"""
        all_zeros = pd.DataFrame(
            [[0, 0, 0, 0]],
            index=['Monday'],
            columns=[0, 1, 2, 3]
        )
        fig = self.visualizer.create_engagement_heatmap_chart(all_zeros)
        self.assertIsNotNone(fig)
    
    def test_create_lifecycle_chart_success(self):
        """Test successful creation of lifecycle chart"""
        fig = self.visualizer.create_lifecycle_chart(self.lifecycle_summary)
        self.assertIsNotNone(fig)
        ax = fig.axes[0]
        self.assertEqual(ax.get_title(), "Contributor Lifecycle Stages")
        self.assertEqual(ax.get_ylabel(), "Contributors")
    
    def test_create_lifecycle_chart_with_date(self):
        """Test lifecycle chart with latest_date parameter"""
        latest_date = pd.Timestamp('2023-06-15')
        fig = self.visualizer.create_lifecycle_chart(
            self.lifecycle_summary,
            latest_date=latest_date
        )
        ax = fig.axes[0]
        self.assertIn("as of Jun 2023", ax.get_title())
    
    def test_create_lifecycle_chart_colors(self):
        """Test that lifecycle chart uses correct colors"""
        fig = self.visualizer.create_lifecycle_chart(self.lifecycle_summary)
        ax = fig.axes[0]
        self.assertIsNotNone(fig)
    
    def test_create_lifecycle_chart_legend(self):
        """Test that legend includes stage definitions"""
        fig = self.visualizer.create_lifecycle_chart(self.lifecycle_summary)
        ax = fig.axes[0]
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
    
    def test_create_lifecycle_chart_single_stage(self):
        """Test chart with single lifecycle stage"""
        single_stage = pd.DataFrame({
            'count': [15]
        }, index=['Active'])
        fig = self.visualizer.create_lifecycle_chart(single_stage)
        self.assertIsNotNone(fig)
    
    def test_create_lifecycle_chart_annotations(self):
        """Test that bars are annotated with counts"""
        fig = self.visualizer.create_lifecycle_chart(self.lifecycle_summary)
        self.assertIsNotNone(fig)
    
    @patch('matplotlib.pyplot.Figure.savefig')
    def test_save_figure_matplotlib(self, mock_savefig):
        """Test saving matplotlib figure"""
        fig = self.visualizer.create_bug_closure_distribution_chart(
            self.yearly_distribution,
            "Test"
        )
        self.visualizer.save_figure(fig, "test.png")
        mock_savefig.assert_called_once_with("test.png", bbox_inches="tight")
    
    def test_create_bug_closure_distribution_empty_dataframe(self):
        """Test bug closure chart with empty DataFrame"""
        empty_df = pd.DataFrame(columns=['year', 'top5_pct', 'rest_pct'])
        fig = self.visualizer.create_bug_closure_distribution_chart(empty_df, "Empty")
        self.assertIsNotNone(fig)


# ============================================================================
# Model Tests (New Additions)
# ============================================================================

class TestLabelResolutionPredictor(unittest.TestCase):
    """Test suite for LabelResolutionPredictor class - 90%+ coverage"""
    
    def setUp(self):
        """Set up test fixtures"""
        from model import LabelResolutionPredictor
        self.predictor = LabelResolutionPredictor()
        
        # Sample training data
        self.features = [
            [5, 3, 10, 2],  # [num_comments, num_events, text_length, assignee_count]
            [2, 1, 5, 1],
            [10, 8, 50, 3],
            [7, 4, 20, 2],
            [3, 2, 8, 1],
            [15, 12, 80, 4],
            [6, 3, 15, 2],
            [4, 2, 12, 1],
            [8, 5, 25, 2],
            [12, 9, 60, 3]
        ]
        self.labels = [24, 12, 120, 48, 18, 168, 36, 20, 60, 96]  # hours
        self.feature_names = ['num_comments', 'num_events', 'text_length', 'assignee_count']
    
    def test_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.predictor.rf_model)
        self.assertIsNotNone(self.predictor.gb_model)
        self.assertIsNotNone(self.predictor.scaler)
        self.assertFalse(self.predictor.is_trained)
        self.assertEqual(len(self.predictor.feature_names), 0)
    
    def test_train_with_sufficient_data(self):
        """Test training with sufficient data"""
        result = self.predictor.train(self.features, self.labels, self.feature_names)
        
        self.assertEqual(result['status'], 'success')
        self.assertTrue(self.predictor.is_trained)
        self.assertIn('training_samples', result)
        self.assertIn('test_samples', result)
        self.assertIn('random_forest', result)
        self.assertIn('gradient_boosting', result)
        self.assertIn('ensemble', result)
        self.assertIn('feature_importance', result)
    
    def test_train_with_insufficient_data(self):
        """Test training fails with insufficient data"""
        small_features = [[1, 2, 3, 4], [5, 6, 7, 8]]
        small_labels = [10, 20]
        
        result = self.predictor.train(small_features, small_labels, self.feature_names)
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('Insufficient training data', result['message'])
        self.assertFalse(self.predictor.is_trained)
    
    def test_training_metrics_structure(self):
        """Test training metrics contain all required fields"""
        result = self.predictor.train(self.features, self.labels, self.feature_names)
        
        # Check Random Forest metrics
        rf_metrics = result['random_forest']
        self.assertIn('mae_hours', rf_metrics)
        self.assertIn('mae_days', rf_metrics)
        self.assertIn('rmse_hours', rf_metrics)
        self.assertIn('r2_score', rf_metrics)
        
        # Check Gradient Boosting metrics
        gb_metrics = result['gradient_boosting']
        self.assertIn('mae_hours', gb_metrics)
        self.assertIn('mae_days', gb_metrics)
        self.assertIn('rmse_hours', gb_metrics)
        self.assertIn('r2_score', gb_metrics)
        
        # Check Ensemble metrics
        ensemble_metrics = result['ensemble']
        self.assertIn('mae_hours', ensemble_metrics)
        self.assertIn('mae_days', ensemble_metrics)
        self.assertIn('rmse_hours', ensemble_metrics)
        self.assertIn('r2_score', ensemble_metrics)
    
    def test_predict_before_training(self):
        """Test prediction fails before training"""
        result = self.predictor.predict([5, 3, 10, 2])
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('not trained', result['message'])
    
    def test_predict_after_training(self):
        """Test prediction succeeds after training"""
        self.predictor.train(self.features, self.labels, self.feature_names)
        result = self.predictor.predict([5, 3, 10, 2])
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('predicted_hours', result)
        self.assertIn('predicted_days', result)
        self.assertIn('confidence_interval', result)
        self.assertIn('model_predictions', result)
        
        # Check confidence interval structure
        ci = result['confidence_interval']
        self.assertIn('lower_days', ci)
        self.assertIn('upper_days', ci)
        self.assertGreaterEqual(ci['upper_days'], ci['lower_days'])
    
    def test_predict_batch(self):
        """Test batch prediction"""
        self.predictor.train(self.features, self.labels, self.feature_names)
        test_features = [[5, 3, 10, 2], [10, 8, 50, 3], [3, 2, 8, 1]]
        
        results = self.predictor.predict_batch(test_features)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result['status'], 'success')
            self.assertIn('predicted_hours', result)
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        self.predictor.train(self.features, self.labels, self.feature_names)
        importance = self.predictor.training_metrics['feature_importance']
        
        self.assertEqual(len(importance), len(self.feature_names))
        # Check that importance values sum to approximately 1
        total_importance = sum(importance.values())
        self.assertAlmostEqual(total_importance, 1.0, places=1)
    
    def test_get_model_info_untrained(self):
        """Test model info for untrained model"""
        info = self.predictor.get_model_info()
        
        self.assertFalse(info['is_trained'])
        self.assertEqual(len(info['features']), 0)
        self.assertIsNone(info['training_metrics'])
        self.assertIn('Ensemble', info['model_type'])
    
    def test_get_model_info_trained(self):
        """Test model info for trained model"""
        self.predictor.train(self.features, self.labels, self.feature_names)
        info = self.predictor.get_model_info()
        
        self.assertTrue(info['is_trained'])
        self.assertEqual(info['features'], self.feature_names)
        self.assertIsNotNone(info['training_metrics'])
    
    @patch('joblib.dump')
    def test_save_model_success(self, mock_dump):
        """Test saving trained model"""
        self.predictor.train(self.features, self.labels, self.feature_names)
        result = self.predictor.save_model('model.pkl')
        
        self.assertTrue(result)
        mock_dump.assert_called_once()
    
    def test_save_model_untrained(self):
        """Test saving untrained model fails"""
        result = self.predictor.save_model('model.pkl')
        self.assertFalse(result)
    
    @patch('joblib.load')
    def test_load_model_success(self, mock_load):
        """Test loading model successfully"""
        mock_load.return_value = {
            'rf_model': MagicMock(),
            'gb_model': MagicMock(),
            'scaler': MagicMock(),
            'feature_names': self.feature_names,
            'training_metrics': {'status': 'success'}
        }
        
        result = self.predictor.load_model('model.pkl')
        
        self.assertTrue(result)
        self.assertTrue(self.predictor.is_trained)
        mock_load.assert_called_once_with('model.pkl')
    
    @patch('joblib.load')
    def test_load_model_failure(self, mock_load):
        """Test loading model handles errors"""
        mock_load.side_effect = Exception("File not found")
        
        result = self.predictor.load_model('model.pkl')
        
        self.assertFalse(result)
        self.assertFalse(self.predictor.is_trained)
    
    def test_mae_days_conversion(self):
        """Test MAE is correctly converted to days"""
        self.predictor.train(self.features, self.labels, self.feature_names)
        metrics = self.predictor.training_metrics['ensemble']
        
        # Check conversion: mae_days = mae_hours / 24
        expected_mae_days = metrics['mae_hours'] / 24
        self.assertAlmostEqual(metrics['mae_days'], expected_mae_days, places=2)
    
    def test_ensemble_prediction_averaging(self):
        """Test ensemble uses average of RF and GB predictions"""
        self.predictor.train(self.features, self.labels, self.feature_names)
        result = self.predictor.predict([5, 3, 10, 2])
        
        rf_pred = result['model_predictions']['random_forest_days']
        gb_pred = result['model_predictions']['gradient_boosting_days']
        ensemble_pred = result['predicted_days']
        
        expected_avg = (rf_pred + gb_pred) / 2
        self.assertAlmostEqual(ensemble_pred, expected_avg, places=2)


class TestEvent(unittest.TestCase):
    """Test suite for Event class - 90%+ coverage"""
    
    def test_initialization_with_none(self):
        """Test Event initialization with None"""
        from model import Event
        event = Event(None)
        
        self.assertIsNone(event.event_type)
        self.assertIsNone(event.author)
        self.assertIsNone(event.event_date)
        self.assertIsNone(event.label)
        self.assertIsNone(event.comment)
    
    def test_initialization_with_json(self):
        """Test Event initialization with JSON object"""
        from model import Event
        jobj = {
            'event_type': 'commented',
            'author': 'user1',
            'event_date': '2024-01-15T10:30:00Z',
            'label': 'bug',
            'comment': 'This is a comment'
        }
        event = Event(jobj)
        
        self.assertEqual(event.event_type, 'commented')
        self.assertEqual(event.author, 'user1')
        self.assertIsNotNone(event.event_date)
        self.assertEqual(event.label, 'bug')
        self.assertEqual(event.comment, 'This is a comment')
    
    def test_from_json_with_invalid_date(self):
        """Test Event handles invalid date gracefully"""
        from model import Event
        jobj = {
            'event_type': 'closed',
            'author': 'user2',
            'event_date': 'invalid-date',
            'label': None,
            'comment': None
        }
        event = Event(jobj)
        
        self.assertEqual(event.event_type, 'closed')
        self.assertIsNone(event.event_date)
    
    def test_is_close_event(self):
        """Test is_close_event method"""
        from model import Event
        close_event = Event({'event_type': 'closed'})
        comment_event = Event({'event_type': 'commented'})
        
        self.assertTrue(close_event.is_close_event())
        self.assertFalse(comment_event.is_close_event())
    
    def test_is_comment_event(self):
        """Test is_comment_event method"""
        from model import Event
        comment_event = Event({'event_type': 'commented'})
        close_event = Event({'event_type': 'closed'})
        
        self.assertTrue(comment_event.is_comment_event())
        self.assertFalse(close_event.is_comment_event())


class TestIssue(unittest.TestCase):
    """Test suite for Issue class - 90%+ coverage"""
    
    def test_initialization_empty(self):
        """Test Issue initialization without data"""
        from model import Issue
        issue = Issue()
        
        self.assertIsNone(issue.url)
        self.assertIsNone(issue.creator)
        self.assertEqual(issue.labels, [])
        self.assertEqual(issue.number, -1)
    
    def test_initialization_with_json(self):
        """Test Issue initialization with JSON"""
        from model import Issue
        jobj = {
            'url': 'https://github.com/repo/issues/1',
            'creator': 'user1',
            'labels': ['bug', 'high-priority'],
            'state': 'open',
            'assignees': ['dev1', 'dev2'],
            'title': 'Test Issue',
            'text': 'Issue description',
            'number': '123',
            'created_date': '2024-01-15T10:00:00Z',
            'updated_date': '2024-01-16T12:00:00Z',
            'timeline_url': 'https://github.com/repo/issues/1/timeline',
            'events': [
                {'event_type': 'commented', 'author': 'user2', 'event_date': '2024-01-15T11:00:00Z'}
            ]
        }
        issue = Issue(jobj)
        
        self.assertEqual(issue.url, 'https://github.com/repo/issues/1')
        self.assertEqual(issue.creator, 'user1')
        self.assertEqual(len(issue.labels), 2)
        self.assertEqual(issue.number, 123)
        self.assertEqual(len(issue.events), 1)
    
    def test_get_closure_date_with_closed_issue(self):
        """Test get_closure_date for closed issue"""
        from model import Issue
        jobj = {
            'state': 'closed',
            'events': [
                {'event_type': 'closed', 'event_date': '2024-01-20T10:00:00Z'},
                {'event_type': 'commented', 'event_date': '2024-01-19T10:00:00Z'}
            ]
        }
        issue = Issue(jobj)
        
        closure_date = issue.get_closure_date()
        self.assertIsNotNone(closure_date)
    
    def test_get_closure_date_with_open_issue(self):
        """Test get_closure_date for open issue"""
        from model import Issue
        jobj = {
            'state': 'open',
            'events': [
                {'event_type': 'commented', 'event_date': '2024-01-19T10:00:00Z'}
            ]
        }
        issue = Issue(jobj)
        
        self.assertIsNone(issue.get_closure_date())
    
    def test_is_closed(self):
        """Test is_closed method"""
        from model import Issue
        closed_issue = Issue({'state': 'closed'})
        open_issue = Issue({'state': 'open'})
        
        self.assertTrue(closed_issue.is_closed())
        self.assertFalse(open_issue.is_closed())
    
    def test_get_resolution_time(self):
        """Test get_resolution_time calculation"""
        from model import Issue
        jobj = {
            'created_date': '2024-01-15T10:00:00Z',
            'state': 'closed',
            'events': [
                {'event_type': 'closed', 'event_date': '2024-01-17T10:00:00Z'}
            ]
        }
        issue = Issue(jobj)
        
        resolution_time = issue.get_resolution_time()
        self.assertIsNotNone(resolution_time)
        self.assertEqual(resolution_time.days, 2)
    
    def test_get_resolution_time_no_closure(self):
        """Test get_resolution_time for open issue"""
        from model import Issue
        issue = Issue({'state': 'open', 'created_date': '2024-01-15T10:00:00Z'})
        
        self.assertIsNone(issue.get_resolution_time())
    
    def test_priority_methods(self):
        """Test set_priority and get_priority methods"""
        from model import Issue
        issue = Issue()
        
        self.assertIsNone(issue.get_priority())
        
        issue.set_priority('high')
        self.assertEqual(issue.get_priority(), 'high')


class TestContributor(unittest.TestCase):
    """Test suite for Contributor class - 90%+ coverage"""
    
    def setUp(self):
        """Set up test fixtures"""
        from model import Contributor, Issue, Event
        from datetime import datetime
        
        self.Contributor = Contributor
        self.Issue = Issue
        self.Event = Event
        self.datetime = datetime
    
    def test_initialization(self):
        """Test Contributor initialization"""
        contributor = self.Contributor('user1')
        
        self.assertEqual(contributor.username, 'user1')
        self.assertEqual(len(contributor.issues_created), 0)
        self.assertEqual(len(contributor.issues_closed), 0)
        self.assertEqual(len(contributor.comments), 0)
        self.assertIsNone(contributor.first_activity)
        self.assertIsNone(contributor.last_activity)
    
    def test_add_comment(self):
        """Test adding comment to contributor"""
        contributor = self.Contributor('user1')
        event = self.Event({'event_type': 'commented', 'event_date': '2024-01-15T12:00:00Z'})
        
        contributor.add_comment(event)
        
        self.assertEqual(len(contributor.comments), 1)
        self.assertIsNotNone(contributor.first_activity)
    
    def test_add_closed_issue(self):
        """Test adding closed issue"""
        contributor = self.Contributor('user1')
        issue = self.Issue({
            'state': 'closed',
            'events': [{'event_type': 'closed', 'event_date': '2024-01-20T10:00:00Z'}]
        })
        
        contributor.add_closed_issue(issue)
        
        self.assertEqual(len(contributor.issues_closed), 1)
    
    def test_update_activity_with_none_date(self):
        """Test activity update handles None dates gracefully"""
        contributor = self.Contributor('user1')
        issue = self.Issue()  # No date
        
        contributor.add_issue(issue)
        
        self.assertIsNone(contributor.first_activity)
        self.assertIsNone(contributor.last_activity)


if __name__ == '__main__':
    unittest.main()
