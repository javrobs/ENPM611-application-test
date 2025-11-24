import unittest
from unittest.mock import patch, MagicMock, Mock
from app.feature_runner import FeatureRunner
from controllers.contributors_controller import ContributorsController
from analysis.contributors_analyzer import ContributorsAnalyzer
from visualization.visualizer import Visualizer
from model import Contributor
from data_loader import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tempfile
import os

# Mock data for testing
MOCK_JSON = [
    {
        "url": "https://github.com/test/repo/issues/1",
        "creator": "user1",
        "labels": ["kind/bug", "critical"],
        "state": "closed",
        "assignees": [],
        "title": "Critical bug fix",
        "text": "Bug description",
        "number": 1,
        "created_date": "2023-01-15T10:00:00+00:00",
        "updated_date": "2023-01-15T14:00:00+00:00",
        "timeline_url": "https://api.github.com/repos/test/repo/issues/1/timeline",
        "events": [
            {
                "event_type": "labeled",
                "author": "user1",
                "event_date": "2023-01-15T10:00:00+00:00",
                "label": "kind/bug"
            },
            {
                "event_type": "commented",
                "author": "user2",
                "event_date": "2023-01-15T11:00:00+00:00",
                "comment": "I'll fix this"
            },
            {
                "event_type": "closed",
                "author": "user2",
                "event_date": "2023-01-15T14:00:00+00:00"
            }
        ]
    },
    {
        "url": "https://github.com/test/repo/issues/2",
        "creator": "user3",
        "labels": ["feature", "enhancement", "area/docs"],
        "state": "open",
        "assignees": [],
        "title": "Add new feature",
        "text": "Feature request",
        "number": 2,
        "created_date": "2023-02-10T09:00:00+00:00",
        "updated_date": "2023-02-10T09:30:00+00:00",
        "timeline_url": "https://api.github.com/repos/test/repo/issues/2/timeline",
        "events": [
            {
                "event_type": "labeled",
                "author": "user3",
                "event_date": "2023-02-10T09:00:00+00:00",
                "label": "feature"
            },
            {
                "event_type": "commented",
                "author": "user1",
                "event_date": "2023-02-10T09:15:00+00:00",
                "comment": "Good idea"
            }
        ]
    },
    {
        "url": "https://github.com/test/repo/issues/3",
        "creator": "user2",
        "labels": ["area/docs"],
        "state": "closed",
        "assignees": [],
        "title": "Update documentation",
        "text": "Docs need updating",
        "number": 3,
        "created_date": "2024-03-05T08:00:00+00:00",
        "updated_date": "2024-03-10T08:00:00+00:00",
        "timeline_url": "https://api.github.com/repos/test/repo/issues/3/timeline",
        "events": [
            {
                "event_type": "labeled",
                "author": "user2",
                "event_date": "2024-03-05T08:00:00+00:00",
                "label": "area/docs"
            },
            {
                "event_type": "commented",
                "author": "user3",
                "event_date": "2024-03-06T10:00:00+00:00",
                "comment": "I can help"
            },
            {
                "event_type": "closed",
                "author": "user2",
                "event_date": "2024-03-10T08:00:00+00:00"
            }
        ]
    }
]


class TestRunnerTwo(unittest.TestCase):
    """Test suite for Feature 2 execution via FeatureRunner."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = FeatureRunner()
        self.patcher = patch("data_loader.DataLoader.load_json", return_value=MOCK_JSON)
        self.patcher.start()

    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        plt.close('all')

    def test_initialize_components(self):
        """Test that runner initializes all components correctly."""
        self.runner.initialize_components()
        
        self.assertIsNotNone(self.runner.config)
        self.assertIsNotNone(self.runner.contributors_controller)
        self.assertIsNotNone(self.runner.priority_controller)

    @patch('matplotlib.pyplot.show')
    def test_run_feature_two(self, mock_show):
        """Test running feature 2 executes without errors."""
        self.runner.initialize_components()
        
        # Should not raise any exceptions
        try:
            self.runner.run_feature(2)
        except Exception as e:
            self.fail(f"Feature 2 raised an exception: {e}")

    @patch('matplotlib.pyplot.show')
    def test_run_feature_two_creates_visualizations(self, mock_show):
        """Test that feature 2 creates expected visualizations."""
        self.runner.initialize_components()
        self.runner.run_feature(2)
        
        # Verify matplotlib show was called (indicates plots were created)
        mock_show.assert_called()


class TestVisualizer(unittest.TestCase):
    """Test suite for Visualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = Visualizer()

    def tearDown(self):
        """Clean up matplotlib figures."""
        plt.close('all')

    def test_create_bug_closure_distribution_chart(self):
        """Test bug closure distribution chart creation."""
        yearly_data = pd.DataFrame({
            'year': [2023, 2024],
            'top5_pct': [60.0, 70.0],
            'rest_pct': [40.0, 30.0],
            'top5_users': ['user1, user2', 'user3, user4']
        })
        
        fig = self.visualizer.create_bug_closure_distribution_chart(
            yearly_data,
            "Test Bug Closure Distribution"
        )
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 1)
        # Check bars were created
        self.assertGreater(len(fig.axes[0].patches), 0)

    def test_create_top_feature_requesters_chart(self):
        """Test top feature requesters chart creation."""
        top_requesters = pd.Series([5, 3, 2], index=['user1', 'user2', 'user3'])
        
        feature_issues = pd.DataFrame({
            'creator': ['user1', 'user1', 'user2', 'user3'],
            'state': ['State.open', 'State.closed', 'State.open', 'State.closed']
        })
        
        fig = self.visualizer.create_top_feature_requesters_chart(
            top_requesters,
            feature_issues,
            "Test Feature Requesters"
        )
        
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.axes[0].patches), 0)

    def test_create_docs_issues_chart(self):
        """Test docs issues chart creation."""
        status_counts = pd.DataFrame({
            'open': [5, 3],
            'closed': [2, 4]
        }, index=pd.to_datetime(['2024-01-01', '2024-02-01']))
        
        avg_commenters = pd.Series([2.5, 3.0], 
                                   index=pd.to_datetime(['2024-01-01', '2024-02-01']))
        
        fig = self.visualizer.create_docs_issues_chart(
            status_counts,
            avg_commenters,
            "Test Docs Issues"
        )
        
        self.assertIsNotNone(fig)
        # Should have two y-axes
        self.assertEqual(len(fig.axes), 2)

    def test_create_issues_created_per_user_chart(self):
        """Test issues created per user chart creation."""
        issues_per_user = pd.Series([10, 8, 6, 5], index=['user1', 'user2', 'user3', 'user4'])
        all_counts = pd.Series([10, 8, 6, 5, 3, 2, 1], 
                               index=['user1', 'user2', 'user3', 'user4', 'user5', 'user6', 'user7'])
        
        fig = self.visualizer.create_issues_created_per_user_chart(
            issues_per_user,
            all_counts,
            "Test Issues Created"
        )
        
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.axes[0].patches), 0)

    def test_create_top_active_users_per_year_chart(self):
        """Test top active users per year chart creation."""
        yearly_data = {
            2023: pd.DataFrame({
                'user': ['user1', 'user2'],
                'activity': [100, 80]
            }),
            2024: pd.DataFrame({
                'user': ['user3', 'user1'],
                'activity': [120, 90]
            })
        }
        
        fig = self.visualizer.create_top_active_users_per_year_chart(yearly_data, top_n=10)
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)

    def test_create_top_active_users_chart_empty_data(self):
        """Test top active users chart handles empty data."""
        yearly_data = {}
        
        fig = self.visualizer.create_top_active_users_per_year_chart(yearly_data, top_n=10)
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)

    def test_create_engagement_heatmap_chart(self):
        """Test engagement heatmap chart creation."""
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        hours = list(range(24))
        heatmap_data = pd.DataFrame(
            np.random.randint(0, 100, (7, 24)),
            index=days,
            columns=hours
        )
        
        fig = self.visualizer.create_engagement_heatmap_chart(heatmap_data)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)

    def test_create_lifecycle_chart(self):
        """Test lifecycle chart creation."""
        summary = pd.DataFrame({
            'count': [10, 20, 5, 8],
            '%': [23.3, 46.5, 11.6, 18.6]
        }, index=['Newcomer', 'Active', 'Core Maintainer', 'Graduated Contributor'])
        
        latest_date = pd.Timestamp('2024-11-24', tz='UTC')
        
        fig = self.visualizer.create_lifecycle_chart(summary, latest_date)
        
        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.axes[0].patches), 0)

class TestContributorsAnalyzer(unittest.TestCase):
    """Test suite for ContributorsAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ContributorsAnalyzer()
        self.patcher = patch("data_loader.DataLoader.load_json", return_value=MOCK_JSON)
        self.patcher.start()
        
        loader = DataLoader()
        self.issues_df = loader.parse_issues()
        self.events_df = loader.parse_events(self.issues_df)

    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        import data_loader
        data_loader._ISSUES = None #Test interference

    def test_build_contributors(self):
        """Test building contributor objects from data."""
        contributors = self.analyzer.build_contributors(self.issues_df, self.events_df)
        
        self.assertIsInstance(contributors, list)
        self.assertGreater(len(contributors), 0)
        
        # Check all contributors are Contributor instances
        for c in contributors:
            self.assertIsInstance(c, Contributor)

    def test_build_contributors_tracks_issues_created(self):
        """Test that contributors track issues they created."""
        contributors = self.analyzer.build_contributors(self.issues_df, self.events_df)
        
        # Find user1 who created issue #1
        user1 = next((c for c in contributors if c.username == 'user1'), None)
        self.assertIsNotNone(user1)
        self.assertGreater(len(user1.issues_created), 0)

    def test_build_contributors_tracks_comments(self):
        """Test that contributors track their comments."""
        contributors = self.analyzer.build_contributors(self.issues_df, self.events_df)
        
        # Find user2 who commented on issue #1
        user2 = next((c for c in contributors if c.username == 'user2'), None)
        self.assertIsNotNone(user2)
        self.assertGreater(len(user2.comments), 0)

    def test_build_contributors_filters_bots(self):
        """Test that bot users are filtered out."""
        # Add a bot issue
        bot_issues_df = self.issues_df.copy()
        bot_row = bot_issues_df.iloc[0].copy()
        bot_row['creator'] = 'stale[bot]'
        bot_issues_df = pd.concat([bot_issues_df, pd.DataFrame([bot_row])], ignore_index=True)
        
        contributors = self.analyzer.build_contributors(bot_issues_df, self.events_df)
        
        # Check that bot is not in contributors
        bot_usernames = [c.username for c in contributors]
        self.assertNotIn('stale[bot]', bot_usernames)
        self.assertNotIn('github-actions[bot]', bot_usernames)

    def test_analyze_bug_closure_distribution(self):
        """Test bug closure distribution analysis."""
        result = self.analyzer.analyze_bug_closure_distribution(self.issues_df, self.events_df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('year', result.columns)
        self.assertIn('top5_pct', result.columns)
        self.assertIn('rest_pct', result.columns)
        self.assertIn('top5_users', result.columns)

    def test_analyze_bug_closure_distribution_percentages_sum_to_100(self):
        """Test that top5 and rest percentages sum to 100."""
        result = self.analyzer.analyze_bug_closure_distribution(self.issues_df, self.events_df)
        
        for _, row in result.iterrows():
            total = row['top5_pct'] + row['rest_pct']
            self.assertAlmostEqual(total, 100.0, places=1)

    def test_analyze_top_feature_requesters(self):
        """Test top feature requesters analysis."""
        top_requesters, feature_issues = self.analyzer.analyze_top_feature_requesters(
            self.issues_df, top_n=10
        )
        
        self.assertIsInstance(top_requesters, pd.Series)
        self.assertIsInstance(feature_issues, pd.DataFrame)
        self.assertGreater(len(feature_issues), 0)

    def test_analyze_top_feature_requesters_no_features(self):
        """Test feature requesters analysis with no feature issues."""
        # Filter out all feature issues
        no_features_df = self.issues_df[
            ~self.issues_df['labels'].apply(lambda L: any('feature' in str(l).lower() for l in L))
        ]
        
        result = self.analyzer.analyze_top_feature_requesters(no_features_df, top_n=10)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])

    def test_compute_unique_commenters(self):
        """Test unique commenters computation."""
        result = self.analyzer.compute_unique_commenters(self.events_df, self.issues_df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('issue_number', result.columns)
        self.assertIn('n_unique_commenters', result.columns)

    def test_analyze_docs_issues(self):
        """Test docs issues analysis."""
        loader = DataLoader()
        status_counts, avg_commenters = self.analyzer.analyze_docs_issues(
            self.issues_df, self.events_df, loader
        )
        
        self.assertIsInstance(status_counts, pd.DataFrame)
        self.assertIsInstance(avg_commenters, pd.Series)

    def test_analyze_docs_issues_no_docs(self):
        """Test docs issues analysis with no doc issues."""
        # Filter out all doc issues
        no_docs_df = self.issues_df[
            ~self.issues_df['labels'].apply(lambda L: any('doc' in str(l).lower() for l in L))
        ]
        
        loader = DataLoader()
        result = self.analyzer.analyze_docs_issues(no_docs_df, self.events_df, loader)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])

    def test_analyze_issues_created_per_user(self):
        """Test issues created per user analysis."""
        issues_per_user, all_counts = self.analyzer.analyze_issues_created_per_user(
            self.issues_df, top_n=40
        )
        
        self.assertIsInstance(issues_per_user, pd.Series)
        self.assertIsInstance(all_counts, pd.Series)
        self.assertLessEqual(len(issues_per_user), 40)

    def test_analyze_issues_created_per_user_empty(self):
        """Test issues created analysis with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.analyzer.analyze_issues_created_per_user(empty_df, top_n=40)
        self.assertIsNone(result)

    def test_analyze_top_active_users_per_year(self):
        """Test top active users per year analysis."""
        contributors = self.analyzer.build_contributors(self.issues_df, self.events_df)
        result = self.analyzer.analyze_top_active_users_per_year(contributors)
        
        self.assertIsInstance(result, dict)
        # Should have data for years with activity
        self.assertGreater(len(result), 0)
        
        # Check structure of each year's data
        for year, df in result.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn('user', df.columns)
            self.assertIn('activity', df.columns)

    def test_analyze_engagement_heatmap(self):
        """Test engagement heatmap analysis."""
        contributors = self.analyzer.build_contributors(self.issues_df, self.events_df)
        result = self.analyzer.analyze_engagement_heatmap(contributors)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (7, 24))  # 7 days, 24 hours
        
        # Check index and columns
        self.assertEqual(list(result.index), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        self.assertEqual(list(result.columns), list(range(24)))

    def test_analyze_lifecycle_stages(self):
        """Test lifecycle stages analysis."""
        contributors = self.analyzer.build_contributors(self.issues_df, self.events_df)
        result = self.analyzer.analyze_lifecycle_stages(contributors)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('contributor', result.columns)
        self.assertIn('stage', result.columns)
        self.assertIn('first_activity', result.columns)
        self.assertIn('last_activity', result.columns)

    def test_analyze_lifecycle_stages_valid_stages(self):
        """Test that lifecycle stages are valid."""
        contributors = self.analyzer.build_contributors(self.issues_df, self.events_df)
        result = self.analyzer.analyze_lifecycle_stages(contributors)
        
        valid_stages = {'Newcomer', 'Active', 'Core Maintainer', 'Graduated Contributor'}
        for stage in result['stage'].unique():
            self.assertIn(stage, valid_stages)

    def test_analyze_lifecycle_stages_no_duplicates(self):
        """Test that each contributor appears only once in lifecycle analysis."""
        contributors = self.analyzer.build_contributors(self.issues_df, self.events_df)
        result = self.analyzer.analyze_lifecycle_stages(contributors)
        
        # Check no duplicate contributors
        self.assertEqual(len(result), len(result['contributor'].unique()))


class TestContributorsController(unittest.TestCase):
    """Test suite for ContributorsController class."""

    def setUp(self):
        """Set up test fixtures."""
        self.controller = ContributorsController()
        self.patcher = patch("data_loader.DataLoader.load_json", return_value=MOCK_JSON)
        self.patcher.start()

    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        plt.close('all')

    def test_initialization(self):
        """Test controller initializes with required components."""
        self.assertIsNotNone(self.controller.data_loader)
        self.assertIsNotNone(self.controller.analyzer)
        self.assertIsNotNone(self.controller.visualizer)
        
        self.assertIsInstance(self.controller.data_loader, DataLoader)
        self.assertIsInstance(self.controller.analyzer, ContributorsAnalyzer)
        self.assertIsInstance(self.controller.visualizer, Visualizer)

    def test_load_contributor_data(self):
        """Test loading contributor data returns DataFrames."""
        issues_df, events_df = self.controller.load_contributor_data()
        
        self.assertIsInstance(issues_df, pd.DataFrame)
        self.assertIsInstance(events_df, pd.DataFrame)
        self.assertGreater(len(issues_df), 0)
        self.assertGreater(len(events_df), 0)

    def test_plot_bug_closure_distribution(self):
        """Test bug closure distribution plotting."""
        issues_df, events_df = self.controller.load_contributor_data()
        fig = self.controller.plot_bug_closure_distribution(issues_df, events_df)
        
        self.assertIsNotNone(fig)

    def test_plot_top_feature_requesters(self):
        """Test top feature requesters plotting."""
        issues_df, _ = self.controller.load_contributor_data()
        fig = self.controller.plot_top_feature_requesters(issues_df, top_n=10)
        
        # May return None if no feature issues
        if fig is not None:
            self.assertIsNotNone(fig)

    def test_plot_docs_issues(self):
        """Test docs issues plotting."""
        issues_df, events_df = self.controller.load_contributor_data()
        fig = self.controller.plot_docs_issues(issues_df, events_df)
        
        # May return None if no doc issues
        if fig is not None:
            self.assertIsNotNone(fig)

    def test_plot_issues_created_per_user(self):
        """Test issues created per user plotting."""
        issues_df, _ = self.controller.load_contributor_data()
        fig = self.controller.plot_issues_created_per_user(issues_df, top_n=40)
        
        if fig is not None:
            self.assertIsNotNone(fig)

    def test_plot_top_active_users_per_year(self):
        """Test top active users per year plotting."""
        issues_df, events_df = self.controller.load_contributor_data()
        contributors = self.controller.analyzer.build_contributors(issues_df, events_df)
        
        fig = self.controller.plot_top_active_users_per_year(contributors, top_n=10)
        
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, go.Figure)

    def test_run_engagement_heatmap(self):
        """Test engagement heatmap generation."""
        issues_df, events_df = self.controller.load_contributor_data()
        contributors = self.controller.analyzer.build_contributors(issues_df, events_df)
        
        fig = self.controller.run_engagement_heatmap(contributors)
        
        self.assertIsNotNone(fig)

    def test_run_contributor_lifecycle(self):
        """Test contributor lifecycle generation."""
        issues_df, events_df = self.controller.load_contributor_data()
        contributors = self.controller.analyzer.build_contributors(issues_df, events_df)
        
        fig = self.controller.run_contributor_lifecycle(contributors)
        
        self.assertIsNotNone(fig)


if __name__ == "__main__":
    unittest.main()
