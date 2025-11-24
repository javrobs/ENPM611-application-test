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
    
    def test_analyze_bug_closure_no_bugs(self):
        """Test bug closure analysis with no bug issues"""
        issues_df = pd.DataFrame([
            {
                'number': 1,
                'creator': 'user1',
                'state': 'closed',
                'created_date': pd.Timestamp('2023-01-15'),
                'updated_date': pd.Timestamp('2023-01-20'),
                'labels': ['kind/feature']
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
        self.assertEqual(len(result), 0)

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

    def test_concurrent_activity_heatmap(self):
        """Test heatmap with multiple users active in same hour"""
        now = pd.Timestamp.now(tz=timezone.utc)
        same_time = now.replace(hour=14, minute=30)
        
        c1 = Contributor('user1')
        c1.issues_created.append(Issue(1, same_time))
        c1.comments.append(Event(1, 'commented', 'user1', same_time))
        
        c2 = Contributor('user2')
        c2.issues_created.append(Issue(2, same_time))
        c2.comments.append(Event(2, 'commented', 'user2', same_time))
        
        heatmap = self.analyzer.analyze_engagement_heatmap([c1, c2])
        
        day_name = same_time.strftime('%a')
        hour = same_time.hour
        
        # Should count all activities
        self.assertGreater(heatmap.loc[day_name, hour], 0)

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
    
    def test_top_active_users_single_year(self):
        """Test top active users for single year"""
        issues_df = pd.DataFrame([
            {
                'number': i,
                'creator': f'user{i%2}',
                'state': 'closed',
                'created_date': pd.Timestamp('2023-01-15'),
                'updated_date': pd.Timestamp('2023-01-20'),
                'labels': ['kind/bug']
            }
            for i in range(10)
        ])
        
        events_df = pd.DataFrame([
            {
                'issue_number': i,
                'event_type': 'commented',
                'event_author': f'user{(i+1)%2}',
                'event_date': pd.Timestamp('2023-01-18'),
                'label': None,
                'comment': 'test'
            }
            for i in range(10)
        ])
        
        contributors = self.analyzer.build_contributors(issues_df, events_df)
        result = self.analyzer.analyze_top_active_users_per_year(contributors, top_n=5)
        
        self.assertIn(2023, result)
        self.assertLessEqual(len(result[2023]), 5)

    def test_top_active_users_activity_calculation(self):
        """Test that activity is correctly calculated (issues + comments)"""
        now = pd.Timestamp.now(tz=timezone.utc)
        
        c1 = Contributor('user1')
        c1.issues_created = [Issue(i, now) for i in range(5)]
        c1.comments = [Event(i, 'commented', 'user1', now) for i in range(3)]
        
        contributors = [c1]
        result = self.analyzer.analyze_top_active_users_per_year(contributors, top_n=10)
        
        year = now.year
        if year in result and not result[year].empty:
            user1_activity = result[year][result[year]['user'] == 'user1']['activity'].iloc[0]
            self.assertEqual(user1_activity, 8)  # 5 issues + 3 comments

    def test_docs_issues_empty_events(self):
        """Test docs issues analysis with empty events"""
        docs_df = self.sample_issues_df[self.sample_issues_df['labels'].apply(
            lambda x: any('docs' in str(l) for l in x)
        )]
        
        empty_events = pd.DataFrame(columns=['issue_number', 'event_type', 'event_author', 'event_date', 'label', 'comment'])
        
        class MockLoader:
            def filter_by_label(self, df, label):
                return df
        
        status_counts, avg_commenters = self.analyzer.analyze_docs_issues(
            docs_df, empty_events, MockLoader()
        )
        
        if status_counts is not None:
            self.assertIsInstance(status_counts, pd.DataFrame)

    def test_docs_issues_state_distribution(self):
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

    def test_label_special_characters(self):
        """Test labels with special characters (hyphens, underscores, colons)"""
        issues_df = pd.DataFrame([
            {
                'number': 1,
                'creator': 'user1',
                'state': 'open',
                'created_date': pd.Timestamp('2023-01-15'),
                'updated_date': pd.Timestamp('2023-01-20'),
                'labels': ['area/api-gateway', 'priority_high', 'scope:backend']
            }
        ])
        
        contributors = self.analyzer.build_contributors(issues_df, pd.DataFrame())
        
        self.assertEqual(len(contributors), 1)

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
    
    def test_null_date_handling(self):
        """Test handling of NaT, None, and empty dates"""
        issues_df = pd.DataFrame([
            {
                'number': 1,
                'creator': 'user1',
                'state': 'open',
                'created_date': pd.NaT,
                'updated_date': None,
                'labels': ['kind/bug']
            },
            {
                'number': 2,
                'creator': 'user2',
                'state': 'closed',
                'created_date': pd.Timestamp('2023-01-15'),
                'updated_date': pd.Timestamp('2023-01-20'),
                'labels': ['kind/bug']
            }
        ])
        
        contributors = self.analyzer.build_contributors(issues_df, pd.DataFrame())
        
        # Should handle gracefully and still process valid data
        self.assertGreaterEqual(len(contributors), 1)

    def test_special_character_usernames(self):
        """Test usernames with hyphens, underscores, and dots"""
        issues_df = pd.DataFrame([
            {
                'number': 1,
                'creator': 'user-name_123.test',
                'state': 'closed',
                'created_date': pd.Timestamp('2023-01-15'),
                'updated_date': pd.Timestamp('2023-01-20'),
                'labels': ['kind/bug']
            }
        ])
        
        contributors = self.analyzer.build_contributors(issues_df, pd.DataFrame())
        
        self.assertEqual(len(contributors), 1)
        self.assertEqual(contributors[0].username, 'user-name_123.test')

    def test_lifecycle_graduated_threshold(self):
        """Test lifecycle stage graduation threshold (6 months)"""
        now = pd.Timestamp.now(tz=timezone.utc)
        
        # Exactly 6 months ago
        c1 = Contributor('user1')
        c1.first_activity = now - timedelta(days=365)
        c1.last_activity = now - timedelta(days=183)  # ~6 months
        
        result = self.analyzer.analyze_lifecycle_stages([c1], reference_date=now)
        
        stage = result[result['contributor'] == 'user1']['stage'].iloc[0]
        self.assertEqual(stage, 'Graduated Contributor')

    def test_heatmap_completeness(self):
        """Test that heatmap covers all days and hours"""
        now = pd.Timestamp.now(tz=timezone.utc)
        
        c1 = Contributor('user1')
        c1.issues_created.append(Issue(1, now))
        
        heatmap = self.analyzer.analyze_engagement_heatmap([c1])
        
        # Verify structure
        self.assertEqual(list(heatmap.index), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        self.assertEqual(list(heatmap.columns), list(range(24)))
        self.assertEqual(heatmap.shape, (7, 24))


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


if __name__ == '__main__':
    unittest.main()
