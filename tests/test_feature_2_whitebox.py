import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from analysis.contributors_analyzer import ContributorsAnalyzer
from model import Contributor, Issue, Event


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

    def test_build_contributors_valid_data(self):
        """Test building contributors from valid issues and events data"""
        contributors = self.analyzer.build_contributors(self.sample_issues_df, self.sample_events_df)
        
        # Should have 3 contributors (excluding bots)
        self.assertEqual(len(contributors), 3)
        
        # Verify usernames
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
            self.assertNotIn(contributor.username, ['stale[bot]', 'github-actions[bot]'])

    def test_build_contributors_empty_data(self):
        """Test handling of empty DataFrames"""
        empty_issues = pd.DataFrame(columns=['number', 'creator', 'state', 'created_date', 'updated_date', 'labels'])
        empty_events = pd.DataFrame(columns=['issue_number', 'event_type', 'event_author', 'event_date', 'label', 'comment'])
        
        contributors = self.analyzer.build_contributors(empty_issues, empty_events)
        
        self.assertEqual(len(contributors), 0)

    def test_build_contributors_tracks_issues(self):
        """Test that contributors correctly track their created issues"""
        contributors = self.analyzer.build_contributors(self.sample_issues_df, self.sample_events_df)
        
        user1 = next(c for c in contributors if c.username == 'user1')
        self.assertEqual(len(user1.issues_created), 2)  # user1 created issues 1 and 3

    def test_build_contributors_tracks_comments(self):
        """Test that contributors correctly track their comments"""
        contributors = self.analyzer.build_contributors(self.sample_issues_df, self.sample_events_df)
        
        user1 = next(c for c in contributors if c.username == 'user1')
        user3 = next(c for c in contributors if c.username == 'user3')
        
        self.assertGreaterEqual(len(user1.comments), 1)
        self.assertGreaterEqual(len(user3.comments), 1)

    def test_analyze_bug_closure_distribution(self):
        """Test bug closure distribution analysis"""
        result = self.analyzer.analyze_bug_closure_distribution(self.sample_issues_df, self.sample_events_df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('year', result.columns)
        self.assertIn('top5_pct', result.columns)
        self.assertIn('rest_pct', result.columns)
        self.assertIn('top5_users', result.columns)
        
        # Percentages should sum to 100 for each year
        for _, row in result.iterrows():
            total = row['top5_pct'] + row['rest_pct']
            self.assertLess(abs(total - 100), 0.01)  # Allow small floating point error

    def test_analyze_bug_closure_no_bugs(self):
        """Test bug closure analysis with no bug issues"""
        issues_df = pd.DataFrame([
            {
                'number': 1,
                'creator': 'user1',
                'state': 'closed',
                'created_date': pd.Timestamp('2023-01-15'),
                'updated_date': pd.Timestamp('2023-01-20'),
                'labels': ['kind/feature']  # No bug label
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

    def test_analyze_top_feature_requesters(self):
        """Test top feature requesters analysis"""
        top_requesters, feature_issues = self.analyzer.analyze_top_feature_requesters(self.sample_issues_df, top_n=10)
        
        self.assertIsNotNone(top_requesters)
        self.assertIsNotNone(feature_issues)
        self.assertIsInstance(top_requesters, pd.Series)
        self.assertLessEqual(len(top_requesters), 10)

    def test_analyze_top_feature_requesters_no_features(self):
        """Test feature requesters analysis with no feature issues"""
        issues_df = pd.DataFrame([
            {
                'number': 1,
                'creator': 'user1',
                'state': 'closed',
                'created_date': pd.Timestamp('2023-01-15'),
                'updated_date': pd.Timestamp('2023-01-20'),
                'labels': ['kind/bug']  # No feature label
            }
        ])
        
        top_requesters, feature_issues = self.analyzer.analyze_top_feature_requesters(issues_df)
        self.assertIsNone(top_requesters)
        self.assertIsNone(feature_issues)

    def test_compute_unique_commenters(self):
        """Test computation of unique commenters per issue per month"""
        result = self.analyzer.compute_unique_commenters(self.sample_events_df, self.sample_issues_df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('issue_number', result.columns)
        self.assertIn('month', result.columns)
        self.assertIn('n_unique_commenters', result.columns)

    def test_compute_unique_commenters_no_comments(self):
        """Test unique commenters computation with no comment events"""
        events_df = pd.DataFrame([
            {
                'issue_number': 1,
                'event_type': 'closed',  # Not a comment
                'event_author': 'user1',
                'event_date': pd.Timestamp('2023-01-20'),
                'label': None,
                'comment': None
            }
        ])
        
        result = self.analyzer.compute_unique_commenters(events_df, self.sample_issues_df)
        self.assertEqual(len(result), 0)

    def test_analyze_docs_issues(self):
        """Test documentation issues analysis"""
        # Create mock data loader
        class MockLoader:
            def filter_by_label(self, df, label):
                return df[df['labels'].apply(lambda L: any('doc' in str(l).lower() for l in L))]
        
        mock_loader = MockLoader()
        
        docs_issues_df = pd.DataFrame([
            {
                'number': 1,
                'creator': 'user1',
                'state': 'closed',
                'created_date': pd.Timestamp('2023-01-15'),
                'updated_date': pd.Timestamp('2023-01-20'),
                'labels': ['area/docs']
            },
            {
                'number': 2,
                'creator': 'user2',
                'state': 'open',
                'created_date': pd.Timestamp('2023-02-10'),
                'updated_date': pd.Timestamp('2023-02-12'),
                'labels': ['area/docs']
            }
        ])
        
        status_counts, avg_commenters = self.analyzer.analyze_docs_issues(docs_issues_df, self.sample_events_df, mock_loader)
        
        if status_counts is not None:
            self.assertIsInstance(status_counts, pd.DataFrame)
            self.assertIsInstance(avg_commenters, pd.Series)

    def test_analyze_issues_created_per_user(self):
        """Test issues created per user analysis"""
        issues_per_user, all_counts = self.analyzer.analyze_issues_created_per_user(self.sample_issues_df, top_n=40)
        
        self.assertIsNotNone(issues_per_user)
        self.assertIsNotNone(all_counts)
        self.assertIsInstance(issues_per_user, pd.Series)
        self.assertIsInstance(all_counts, pd.Series)
        self.assertLessEqual(len(issues_per_user), 40)

    def test_analyze_issues_created_per_user_empty(self):
        """Test issues created per user with empty DataFrame"""
        empty_df = pd.DataFrame(columns=['number', 'creator', 'state', 'created_date', 'updated_date', 'labels'])
        result = self.analyzer.analyze_issues_created_per_user(empty_df)
        self.assertIsNone(result)

    def test_analyze_top_active_users_per_year(self):
        """Test top active users per year analysis"""
        contributors = self.analyzer.build_contributors(self.sample_issues_df, self.sample_events_df)
        result = self.analyzer.analyze_top_active_users_per_year(contributors)
        
        self.assertIsInstance(result, dict)
        for year, df in result.items():
            self.assertIsInstance(year, int)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn('user', df.columns)
            self.assertIn('activity', df.columns)

    def test_analyze_engagement_heatmap(self):
        """Test engagement heatmap generation"""
        contributors = self.analyzer.build_contributors(self.sample_issues_df, self.sample_events_df)
        heatmap = self.analyzer.analyze_engagement_heatmap(contributors)
        
        self.assertIsInstance(heatmap, pd.DataFrame)
        self.assertEqual(heatmap.shape, (7, 24))  # 7 days, 24 hours
        self.assertEqual(list(heatmap.index), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        self.assertEqual(list(heatmap.columns), list(range(24)))

    def test_analyze_engagement_heatmap_empty(self):
        """Test engagement heatmap with no contributors"""
        heatmap = self.analyzer.analyze_engagement_heatmap([])
        
        self.assertIsInstance(heatmap, pd.DataFrame)
        self.assertEqual(heatmap.shape, (7, 24))
        self.assertEqual(heatmap.sum().sum(), 0)  # All zeros

    def test_analyze_lifecycle_stages(self):
        """Test contributor lifecycle stages analysis"""
        contributors = self.analyzer.build_contributors(self.sample_issues_df, self.sample_events_df)
        result = self.analyzer.analyze_lifecycle_stages(contributors)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('contributor', result.columns)
        self.assertIn('first_activity', result.columns)
        self.assertIn('last_activity', result.columns)
        self.assertIn('stage', result.columns)
        
        # Verify stage values are valid
        valid_stages = ['Newcomer', 'Active', 'Core Maintainer', 'Graduated Contributor']
        self.assertTrue(result['stage'].isin(valid_stages).all())

    def test_analyze_lifecycle_stages_custom_reference(self):
        """Test lifecycle stages with custom reference date"""
        contributors = self.analyzer.build_contributors(self.sample_issues_df, self.sample_events_df)
        reference_date = pd.Timestamp('2024-12-31', tz=timezone.utc)
        
        result = self.analyzer.analyze_lifecycle_stages(contributors, reference_date=reference_date)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('stage', result.columns)

    def test_analyze_lifecycle_stages_categories(self):
        """Test that lifecycle stages correctly categorize contributors"""
        # Create contributors with specific activity patterns
        now = pd.Timestamp.now(tz=timezone.utc)
        
        # Newcomer: first activity within last 30 days
        newcomer = Contributor('newcomer')
        newcomer.first_activity = now - timedelta(days=15)
        newcomer.last_activity = now - timedelta(days=10)
        
        # Core Maintainer: active for more than 1 year
        maintainer = Contributor('maintainer')
        maintainer.first_activity = now - timedelta(days=400)
        maintainer.last_activity = now - timedelta(days=5)
        
        # Graduated: inactive for 6+ months
        graduated = Contributor('graduated')
        graduated.first_activity = now - timedelta(days=300)
        graduated.last_activity = now - timedelta(days=200)
        
        # Active: regular contributor
        active = Contributor('active')
        active.first_activity = now - timedelta(days=100)
        active.last_activity = now - timedelta(days=20)
        
        contributors = [newcomer, maintainer, graduated, active]
        result = self.analyzer.analyze_lifecycle_stages(contributors, reference_date=now)
        
        stages = result.set_index('contributor')['stage'].to_dict()
        self.assertEqual(stages['newcomer'], 'Newcomer')
        self.assertEqual(stages['maintainer'], 'Core Maintainer')
        self.assertEqual(stages['graduated'], 'Graduated Contributor')
        self.assertEqual(stages['active'], 'Active')

    def test_analyze_lifecycle_stages_duplicates(self):
        """Test that duplicate contributors are handled correctly"""
        now = pd.Timestamp.now(tz=timezone.utc)
        
        # Create duplicate contributors
        c1 = Contributor('user1')
        c1.first_activity = now - timedelta(days=100)
        c1.last_activity = now - timedelta(days=50)
        
        c2 = Contributor('user1')  # Same username
        c2.first_activity = now - timedelta(days=80)
        c2.last_activity = now - timedelta(days=30)
        
        result = self.analyzer.analyze_lifecycle_stages([c1, c2])
        
        # Should have only one entry for user1
        self.assertEqual(len(result[result['contributor'] == 'user1']), 1)

    def test_analyze_bug_closure_distribution_multiple_years(self):
        """Test bug closure distribution across multiple years"""
        issues_df = pd.DataFrame([
            {'number': i, 'creator': f'user{i%3}', 'state': 'closed', 
             'created_date': pd.Timestamp(f'202{i%3}-01-15'),
             'updated_date': pd.Timestamp(f'202{i%3}-01-20'),
             'labels': ['kind/bug']}
            for i in range(10)
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': i, 'event_type': 'closed', 'event_author': f'user{(i+1)%3}',
             'event_date': pd.Timestamp(f'202{i%3}-01-20'), 'label': None, 'comment': None}
            for i in range(10)
        ])
        
        result = self.analyzer.analyze_bug_closure_distribution(issues_df, events_df)
        
        self.assertGreater(len(result), 0)
        self.assertGreater(result['year'].nunique(), 1)

    def test_single_contributor(self):
        """Test analysis with only one contributor"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['kind/bug']}
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user1',
             'event_date': pd.Timestamp('2023-01-18'), 'label': None, 'comment': 'test'}
        ])
        
        contributors = self.analyzer.build_contributors(issues_df, events_df)
        self.assertEqual(len(contributors), 1)
        self.assertEqual(contributors[0].username, 'user1')

    def test_date_parsing_robustness(self):
        """Test that date parsing handles various formats correctly"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': '2023-01-15T10:00:00Z',  # String format
             'updated_date': '2023-01-20T15:00:00Z',
             'labels': ['kind/bug']}
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user1',
             'event_date': '2023-01-18T12:00:00Z', 'label': None, 'comment': 'test'}
        ])
        
        # Should not raise an exception
        contributors = self.analyzer.build_contributors(issues_df, events_df)
        self.assertEqual(len(contributors), 1)

    def test_top_n_parameter_respected(self):
        """Test that top_n parameter correctly limits results"""
        issues_df = pd.DataFrame([
            {'number': i, 'creator': f'user{i}', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['kind/feature']}
            for i in range(20)
        ])
        
        top_requesters, _ = self.analyzer.analyze_top_feature_requesters(issues_df, top_n=5)
        self.assertLessEqual(len(top_requesters), 5)

    def test_year_boundary_bug_closure(self):
        """Test bug closure across year boundaries (Dec 31 to Jan 1)"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-12-31 23:00:00'),
             'updated_date': pd.Timestamp('2024-01-01 01:00:00'),
             'labels': ['kind/bug']},
            {'number': 2, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-12-30 10:00:00'),
             'updated_date': pd.Timestamp('2024-01-02 15:00:00'),
             'labels': ['kind/bug']}
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'closed', 'event_author': 'user1',
             'event_date': pd.Timestamp('2024-01-01 01:00:00'), 'label': None, 'comment': None},
            {'issue_number': 2, 'event_type': 'closed', 'event_author': 'user1',
             'event_date': pd.Timestamp('2024-01-02 15:00:00'), 'label': None, 'comment': None}
        ])
        
        result = self.analyzer.analyze_bug_closure_distribution(issues_df, events_df)
        
        # Both closures should be attributed to 2024
        self.assertGreater(len(result), 0)
        self.assertIn(2024, result['year'].values)

    def test_concurrent_user_activity_same_time(self):
        """Test multiple users active in the same hour/day"""
        issues_df = pd.DataFrame([
            {'number': i, 'creator': f'user{i}', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15 14:30:00'),  # Same day/hour
             'updated_date': pd.Timestamp('2023-01-15 14:45:00'),
             'labels': ['kind/bug']}
            for i in range(5)
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': i, 'event_type': 'commented', 'event_author': f'user{i}',
             'event_date': pd.Timestamp('2023-01-15 14:35:00'), 'label': None, 'comment': 'concurrent'}
            for i in range(5)
        ])
        
        contributors = self.analyzer.build_contributors(issues_df, events_df)
        heatmap = self.analyzer.analyze_engagement_heatmap(contributors)
        
        # Sunday (6), hour 14 should have aggregated count from all users
        self.assertGreater(heatmap.loc['Sun', 14], 0)

    def test_lifecycle_stages_timezone_edge_cases(self):
        """Test lifecycle stages with timezone-aware and naive datetimes"""
        now_utc = pd.Timestamp.now(tz=timezone.utc)
        now_naive = pd.Timestamp.now()
        
        c1 = Contributor('user_utc')
        c1.first_activity = now_utc - timedelta(days=15)
        c1.last_activity = now_utc - timedelta(days=5)
        
        c2 = Contributor('user_naive')
        c2.first_activity = now_naive - timedelta(days=15)
        c2.last_activity = now_naive - timedelta(days=5)
        
        # Should handle both without errors
        result = self.analyzer.analyze_lifecycle_stages([c1, c2])
        self.assertEqual(len(result), 2)

    def test_monthly_aggregation_unique_commenters(self):
        """Test that unique commenters are correctly counted per month"""
        events_df = pd.DataFrame([
            # Same user commenting multiple times in same month
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user1',
             'event_date': pd.Timestamp('2023-01-05'), 'label': None, 'comment': 'comment1'},
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user1',
             'event_date': pd.Timestamp('2023-01-10'), 'label': None, 'comment': 'comment2'},
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user2',
             'event_date': pd.Timestamp('2023-01-15'), 'label': None, 'comment': 'comment3'},
        ])
        
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'open',
             'created_date': pd.Timestamp('2023-01-01'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['kind/bug']}
        ])
        
        result = self.analyzer.compute_unique_commenters(events_df, issues_df)
        
        # Should count user1 only once in January despite multiple comments
        jan_row = result[result['month'] == '2023-01']
        if not jan_row.empty:
            self.assertEqual(jan_row.iloc[0]['n_unique_commenters'], 2)

    def test_top_active_users_single_year(self):
        """Test top active users when data exists for only one year"""
        issues_df = pd.DataFrame([
            {'number': i, 'creator': f'user{i%3}', 'state': 'closed',
             'created_date': pd.Timestamp('2023-06-15'),
             'updated_date': pd.Timestamp('2023-06-20'),
             'labels': ['kind/bug']}
            for i in range(10)
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': i, 'event_type': 'commented', 'event_author': f'user{i%3}',
             'event_date': pd.Timestamp('2023-06-18'), 'label': None, 'comment': 'test'}
            for i in range(10)
        ])
        
        contributors = self.analyzer.build_contributors(issues_df, events_df)
        result = self.analyzer.analyze_top_active_users_per_year(contributors)
        
        self.assertEqual(len(result), 1)
        self.assertIn(2023, result)

    def test_top_active_users_activity_calculation(self):
        """Test that activity is correctly calculated (issues + comments)"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['kind/bug']},
            {'number': 2, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-16'),
             'updated_date': pd.Timestamp('2023-01-21'),
             'labels': ['kind/bug']}
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user1',
             'event_date': pd.Timestamp('2023-01-18'), 'label': None, 'comment': 'test1'},
            {'issue_number': 2, 'event_type': 'commented', 'event_author': 'user1',
             'event_date': pd.Timestamp('2023-01-19'), 'label': None, 'comment': 'test2'},
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user1',
             'event_date': pd.Timestamp('2023-01-19'), 'label': None, 'comment': 'test3'}
        ])
        
        contributors = self.analyzer.build_contributors(issues_df, events_df)
        result = self.analyzer.analyze_top_active_users_per_year(contributors)
        
        # user1: 2 issues + 3 comments = 5 total activity
        user1_activity = result[2023][result[2023]['user'] == 'user1']['activity'].values[0]
        self.assertEqual(user1_activity, 5)

    def test_docs_issues_empty_events(self):
        """Test docs issues analysis with empty events DataFrame"""
        class MockLoader:
            def filter_by_label(self, df, label):
                return df[df['labels'].apply(lambda L: any('doc' in str(l).lower() for l in L))]
        
        mock_loader = MockLoader()
        
        docs_issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['area/docs']}
        ])
        
        empty_events = pd.DataFrame(columns=['issue_number', 'event_type', 'event_author', 'event_date', 'label', 'comment'])
        
        status_counts, avg_commenters = self.analyzer.analyze_docs_issues(docs_issues_df, empty_events, mock_loader)
        
        # Should handle empty events gracefully
        self.assertIsNotNone(status_counts)

    def test_docs_issues_state_distribution(self):
        """Test that docs issues correctly calculates state distribution"""
        class MockLoader:
            def filter_by_label(self, df, label):
                return df[df['labels'].apply(lambda L: any('doc' in str(l).lower() for l in L))]
        
        mock_loader = MockLoader()
        
        docs_issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['area/docs']},
            {'number': 2, 'creator': 'user2', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-16'),
             'updated_date': pd.Timestamp('2023-01-21'),
             'labels': ['area/docs']},
            {'number': 3, 'creator': 'user3', 'state': 'open',
             'created_date': pd.Timestamp('2023-01-17'),
             'updated_date': pd.Timestamp('2023-01-22'),
             'labels': ['area/docs']}
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user1',
             'event_date': pd.Timestamp('2023-01-18'), 'label': None, 'comment': 'test'}
        ])
        
        status_counts, _ = self.analyzer.analyze_docs_issues(docs_issues_df, events_df, mock_loader)
        
        if status_counts is not None and 'state' in status_counts.columns:
            closed_count = status_counts[status_counts['state'] == 'closed']['count'].sum()
            open_count = status_counts[status_counts['state'] == 'open']['count'].sum()
            self.assertEqual(closed_count, 2)
            self.assertEqual(open_count, 1)

    def test_label_filtering_case_insensitive(self):
        """Test that label filtering is case-insensitive"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['Kind/Bug', 'PRIORITY/HIGH']},  # Mixed case
            {'number': 2, 'creator': 'user2', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-16'),
             'updated_date': pd.Timestamp('2023-01-21'),
             'labels': ['kind/bug', 'priority/low']}  # Lowercase
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'closed', 'event_author': 'user1',
             'event_date': pd.Timestamp('2023-01-20'), 'label': None, 'comment': None},
            {'issue_number': 2, 'event_type': 'closed', 'event_author': 'user2',
             'event_date': pd.Timestamp('2023-01-21'), 'label': None, 'comment': None}
        ])
        
        result = self.analyzer.analyze_bug_closure_distribution(issues_df, events_df)
        
        # Should find both bugs regardless of label case
        self.assertGreater(len(result), 0)

    def test_label_special_characters(self):
        """Test handling of labels with special characters"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['kind/bug', 'area/api-client', 'priority:high', 'good_first_issue']},
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'closed', 'event_author': 'user1',
             'event_date': pd.Timestamp('2023-01-20'), 'label': None, 'comment': None}
        ])
        
        # Should not raise exceptions with special characters
        result = self.analyzer.analyze_bug_closure_distribution(issues_df, events_df)
        contributors = self.analyzer.build_contributors(issues_df, events_df)
        
        self.assertIsNotNone(result)
        self.assertGreater(len(contributors), 0)

    def test_large_dataset_performance(self):
        """Test analysis with large dataset (1000+ issues)"""
        large_issues_df = pd.DataFrame([
            {'number': i, 'creator': f'user{i%100}', 'state': 'closed' if i % 2 == 0 else 'open',
             'created_date': pd.Timestamp(f'2023-{(i%12)+1:02d}-15'),
             'updated_date': pd.Timestamp(f'2023-{(i%12)+1:02d}-20'),
             'labels': ['kind/bug' if i % 3 == 0 else 'kind/feature']}
            for i in range(1000)
        ])
        
        large_events_df = pd.DataFrame([
            {'issue_number': i, 'event_type': 'commented', 'event_author': f'user{(i+1)%100}',
             'event_date': pd.Timestamp(f'2023-{(i%12)+1:02d}-18'), 'label': None, 'comment': f'comment{i}'}
            for i in range(1000)
        ])
        
        # Should complete without errors or excessive time
        contributors = self.analyzer.build_contributors(large_issues_df, large_events_df)
        self.assertGreater(len(contributors), 0)
        
        result = self.analyzer.analyze_bug_closure_distribution(large_issues_df, large_events_df)
        self.assertIsNotNone(result)

    def test_null_missing_dates_handling(self):
        """Test handling of NaT, None, and empty date values"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.NaT,  # NaT value
             'labels': ['kind/bug']},
            {'number': 2, 'creator': 'user2', 'state': 'open',
             'created_date': None,  # None value
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['kind/feature']},
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user1',
             'event_date': pd.NaT, 'label': None, 'comment': 'test'},
            {'issue_number': 2, 'event_type': 'commented', 'event_author': 'user2',
             'event_date': '', 'label': None, 'comment': 'test2'}  # Empty string
        ])
        
        # Should handle gracefully without crashes
        contributors = self.analyzer.build_contributors(issues_df, events_df)
        self.assertIsNotNone(contributors)

    def test_username_special_characters(self):
        """Test handling of usernames with special characters"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user-name_123', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['kind/bug']},
            {'number': 2, 'creator': 'user.name', 'state': 'open',
             'created_date': pd.Timestamp('2023-01-16'),
             'updated_date': pd.Timestamp('2023-01-21'),
             'labels': ['kind/feature']},
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user-name_123',
             'event_date': pd.Timestamp('2023-01-18'), 'label': None, 'comment': 'test'}
        ])
        
        contributors = self.analyzer.build_contributors(issues_df, events_df)
        
        usernames = [c.username for c in contributors]
        self.assertIn('user-name_123', usernames)
        self.assertIn('user.name', usernames)

    def test_lifecycle_graduated_threshold_precision(self):
        """Test the exact 6-month graduated contributor boundary"""
        now = pd.Timestamp.now(tz=timezone.utc)
        
        # Exactly 6 months inactive (should be graduated)
        c1 = Contributor('exactly_6mo')
        c1.first_activity = now - timedelta(days=365)
        c1.last_activity = now - timedelta(days=183)  # 6 months + 1 day
        
        # Just under 6 months inactive (should be active)
        c2 = Contributor('under_6mo')
        c2.first_activity = now - timedelta(days=365)
        c2.last_activity = now - timedelta(days=179)  # Just under 6 months
        
        result = self.analyzer.analyze_lifecycle_stages([c1, c2], reference_date=now)
        
        stages = result.set_index('contributor')['stage'].to_dict()
        self.assertEqual(stages['exactly_6mo'], 'Graduated Contributor')
        self.assertIn(stages['under_6mo'], ['Active', 'Core Maintainer'])

    def test_engagement_heatmap_completeness(self):
        """Test that heatmap includes all days and hours even with sparse data"""
        # Create data only for a few specific hours/days
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-02 10:00:00'),  # Monday 10 AM
             'updated_date': pd.Timestamp('2023-01-02 11:00:00'),
             'labels': ['kind/bug']}
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user1',
             'event_date': pd.Timestamp('2023-01-02 10:30:00'), 'label': None, 'comment': 'test'}
        ])
        
        contributors = self.analyzer.build_contributors(issues_df, events_df)
        heatmap = self.analyzer.analyze_engagement_heatmap(contributors)
        
        # Should still have all 7 days and 24 hours
        self.assertEqual(heatmap.shape, (7, 24))
        self.assertEqual(len(heatmap.index), 7)
        self.assertEqual(len(heatmap.columns), 24)
        
        # Most cells should be 0, but at least one should have activity
        self.assertGreater(heatmap.sum().sum(), 0)


if __name__ == '__main__':
    unittest.main()
