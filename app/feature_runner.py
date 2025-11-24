import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from analysis.contributors_analyzer import ContributorsAnalyzer
from model import Contributor, Issue, Event

class TestContributorsAnalyzer:
    """Comprehensive test suite for ContributorsAnalyzer class"""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture that provides a fresh ContributorsAnalyzer instance"""
        return ContributorsAnalyzer()
    
    @pytest.fixture
    def sample_issues_df(self):
        """Fixture providing sample issues DataFrame with diverse test data"""
        return pd.DataFrame([
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
    
    @pytest.fixture
    def sample_events_df(self):
        """Fixture providing sample events DataFrame"""
        return pd.DataFrame([
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

    def test_build_contributors_valid_data(self, analyzer, sample_issues_df, sample_events_df):
        """Test building contributors from valid issues and events data"""
        contributors = analyzer.build_contributors(sample_issues_df, sample_events_df)
        
        # Should have 3 contributors (excluding bots)
        assert len(contributors) == 3
        
        # Verify usernames
        usernames = [c.username for c in contributors]
        assert 'user1' in usernames
        assert 'user2' in usernames
        assert 'user3' in usernames
        assert 'stale[bot]' not in usernames
        assert 'github-actions[bot]' not in usernames

    def test_build_contributors_filters_bots(self, analyzer, sample_issues_df, sample_events_df):
        """Test that bot users are properly filtered out"""
        contributors = analyzer.build_contributors(sample_issues_df, sample_events_df)
        
        for contributor in contributors:
            assert '[bot]' not in contributor.username
            assert contributor.username not in ['stale[bot]', 'github-actions[bot]']

    def test_build_contributors_empty_data(self, analyzer):
        """Test handling of empty DataFrames"""
        empty_issues = pd.DataFrame(columns=['number', 'creator', 'state', 'created_date', 'updated_date', 'labels'])
        empty_events = pd.DataFrame(columns=['issue_number', 'event_type', 'event_author', 'event_date', 'label', 'comment'])
        
        contributors = analyzer.build_contributors(empty_issues, empty_events)
        
        assert len(contributors) == 0

    def test_build_contributors_tracks_issues(self, analyzer, sample_issues_df, sample_events_df):
        """Test that contributors correctly track their created issues"""
        contributors = analyzer.build_contributors(sample_issues_df, sample_events_df)
        
        user1 = next(c for c in contributors if c.username == 'user1')
        assert len(user1.issues_created) == 2  # user1 created issues 1 and 3

    def test_build_contributors_tracks_comments(self, analyzer, sample_issues_df, sample_events_df):
        """Test that contributors correctly track their comments"""
        contributors = analyzer.build_contributors(sample_issues_df, sample_events_df)
        
        user1 = next(c for c in contributors if c.username == 'user1')
        user3 = next(c for c in contributors if c.username == 'user3')
        
        assert len(user1.comments) >= 1
        assert len(user3.comments) >= 1

    def test_analyze_bug_closure_distribution(self, analyzer, sample_issues_df, sample_events_df):
        """Test bug closure distribution analysis"""
        result = analyzer.analyze_bug_closure_distribution(sample_issues_df, sample_events_df)
        
        assert isinstance(result, pd.DataFrame)
        assert 'year' in result.columns
        assert 'top5_pct' in result.columns
        assert 'rest_pct' in result.columns
        assert 'top5_users' in result.columns
        
        # Percentages should sum to 100 for each year
        for _, row in result.iterrows():
            total = row['top5_pct'] + row['rest_pct']
            assert abs(total - 100) < 0.01  # Allow small floating point error

    def test_analyze_bug_closure_no_bugs(self, analyzer):
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
        
        result = analyzer.analyze_bug_closure_distribution(issues_df, events_df)
        assert len(result) == 0

    def test_analyze_top_feature_requesters(self, analyzer, sample_issues_df):
        """Test top feature requesters analysis"""
        top_requesters, feature_issues = analyzer.analyze_top_feature_requesters(sample_issues_df, top_n=10)
        
        assert top_requesters is not None
        assert feature_issues is not None
        assert isinstance(top_requesters, pd.Series)
        assert len(top_requesters) <= 10

    def test_analyze_top_feature_requesters_no_features(self, analyzer):
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
        
        top_requesters, feature_issues = analyzer.analyze_top_feature_requesters(issues_df)
        assert top_requesters is None
        assert feature_issues is None

    def test_compute_unique_commenters(self, analyzer, sample_issues_df, sample_events_df):
        """Test computation of unique commenters per issue per month"""
        result = analyzer.compute_unique_commenters(sample_events_df, sample_issues_df)
        
        assert isinstance(result, pd.DataFrame)
        assert 'issue_number' in result.columns
        assert 'month' in result.columns
        assert 'n_unique_commenters' in result.columns

    def test_compute_unique_commenters_no_comments(self, analyzer, sample_issues_df):
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
        
        result = analyzer.compute_unique_commenters(events_df, sample_issues_df)
        assert len(result) == 0

    def test_analyze_docs_issues(self, analyzer, sample_events_df):
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
        
        status_counts, avg_commenters = analyzer.analyze_docs_issues(docs_issues_df, sample_events_df, mock_loader)
        
        if status_counts is not None:
            assert isinstance(status_counts, pd.DataFrame)
            assert isinstance(avg_commenters, pd.Series)

    def test_analyze_issues_created_per_user(self, analyzer, sample_issues_df):
        """Test issues created per user analysis"""
        issues_per_user, all_counts = analyzer.analyze_issues_created_per_user(sample_issues_df, top_n=40)
        
        assert issues_per_user is not None
        assert all_counts is not None
        assert isinstance(issues_per_user, pd.Series)
        assert isinstance(all_counts, pd.Series)
        assert len(issues_per_user) <= 40

    def test_analyze_issues_created_per_user_empty(self, analyzer):
        """Test issues created per user with empty DataFrame"""
        empty_df = pd.DataFrame(columns=['number', 'creator', 'state', 'created_date', 'updated_date', 'labels'])
        result = analyzer.analyze_issues_created_per_user(empty_df)
        assert result is None

    def test_analyze_top_active_users_per_year(self, analyzer, sample_issues_df, sample_events_df):
        """Test top active users per year analysis"""
        contributors = analyzer.build_contributors(sample_issues_df, sample_events_df)
        result = analyzer.analyze_top_active_users_per_year(contributors)
        
        assert isinstance(result, dict)
        for year, df in result.items():
            assert isinstance(year, int)
            assert isinstance(df, pd.DataFrame)
            assert 'user' in df.columns
            assert 'activity' in df.columns

    def test_analyze_engagement_heatmap(self, analyzer, sample_issues_df, sample_events_df):
        """Test engagement heatmap generation"""
        contributors = analyzer.build_contributors(sample_issues_df, sample_events_df)
        heatmap = analyzer.analyze_engagement_heatmap(contributors)
        
        assert isinstance(heatmap, pd.DataFrame)
        assert heatmap.shape == (7, 24)  # 7 days, 24 hours
        assert list(heatmap.index) == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        assert list(heatmap.columns) == list(range(24))

    def test_analyze_engagement_heatmap_empty(self, analyzer):
        """Test engagement heatmap with no contributors"""
        heatmap = analyzer.analyze_engagement_heatmap([])
        
        assert isinstance(heatmap, pd.DataFrame)
        assert heatmap.shape == (7, 24)
        assert heatmap.sum().sum() == 0  # All zeros

    def test_analyze_lifecycle_stages(self, analyzer, sample_issues_df, sample_events_df):
        """Test contributor lifecycle stages analysis"""
        contributors = analyzer.build_contributors(sample_issues_df, sample_events_df)
        result = analyzer.analyze_lifecycle_stages(contributors)
        
        assert isinstance(result, pd.DataFrame)
        assert 'contributor' in result.columns
        assert 'first_activity' in result.columns
        assert 'last_activity' in result.columns
        assert 'stage' in result.columns
        
        # Verify stage values are valid
        valid_stages = ['Newcomer', 'Active', 'Core Maintainer', 'Graduated Contributor']
        assert result['stage'].isin(valid_stages).all()

    def test_analyze_lifecycle_stages_custom_reference(self, analyzer, sample_issues_df, sample_events_df):
        """Test lifecycle stages with custom reference date"""
        contributors = analyzer.build_contributors(sample_issues_df, sample_events_df)
        reference_date = pd.Timestamp('2024-12-31', tz=timezone.utc)
        
        result = analyzer.analyze_lifecycle_stages(contributors, reference_date=reference_date)
        
        assert isinstance(result, pd.DataFrame)
        assert 'stage' in result.columns

    def test_analyze_lifecycle_stages_categories(self, analyzer):
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
        result = analyzer.analyze_lifecycle_stages(contributors, reference_date=now)
        
        stages = result.set_index('contributor')['stage'].to_dict()
        assert stages['newcomer'] == 'Newcomer'
        assert stages['maintainer'] == 'Core Maintainer'
        assert stages['graduated'] == 'Graduated Contributor'
        assert stages['active'] == 'Active'

    def test_analyze_lifecycle_stages_duplicates(self, analyzer):
        """Test that duplicate contributors are handled correctly"""
        now = pd.Timestamp.now(tz=timezone.utc)
        
        # Create duplicate contributors
        c1 = Contributor('user1')
        c1.first_activity = now - timedelta(days=100)
        c1.last_activity = now - timedelta(days=50)
        
        c2 = Contributor('user1')  # Same username
        c2.first_activity = now - timedelta(days=80)
        c2.last_activity = now - timedelta(days=30)
        
        result = analyzer.analyze_lifecycle_stages([c1, c2])
        
        # Should have only one entry for user1
        assert len(result[result['contributor'] == 'user1']) == 1

    def test_analyze_bug_closure_distribution_multiple_years(self, analyzer):
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
        
        result = analyzer.analyze_bug_closure_distribution(issues_df, events_df)
        
        assert len(result) > 0
        assert result['year'].nunique() > 1

    def test_single_contributor(self, analyzer):
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
        
        contributors = analyzer.build_contributors(issues_df, events_df)
        assert len(contributors) == 1
        assert contributors[0].username == 'user1'

    def test_date_parsing_robustness(self, analyzer):
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
        contributors = analyzer.build_contributors(issues_df, events_df)
        assert len(contributors) == 1

    def test_top_n_parameter_respected(self, analyzer):
        """Test that top_n parameter correctly limits results"""
        issues_df = pd.DataFrame([
            {'number': i, 'creator': f'user{i}', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['kind/feature']}
            for i in range(20)
        ])
        
        top_5, _ = analyzer.analyze_top_feature_requesters(issues_df, top_n=5)
        assert len(top_5) <= 5
        
        top_10, _ = analyzer.analyze_top_feature_requesters(issues_df, top_n=10)
        assert len(top_10) <= 10

    def test_analyze_bug_closure_year_boundary(self, analyzer):
        """Test bug closure distribution across year boundaries (Dec 31 → Jan 1)"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-12-31 23:00:00'),
             'updated_date': pd.Timestamp('2024-01-01 01:00:00'),
             'labels': ['kind/bug']},
            {'number': 2, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-12-30 10:00:00'),
             'updated_date': pd.Timestamp('2024-01-02 10:00:00'),
             'labels': ['kind/bug']}
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'closed', 'event_author': 'user2',
             'event_date': pd.Timestamp('2024-01-01 01:00:00'), 'label': None, 'comment': None},
            {'issue_number': 2, 'event_type': 'closed', 'event_author': 'user2',
             'event_date': pd.Timestamp('2024-01-02 10:00:00'), 'label': None, 'comment': None}
        ])
        
        result = analyzer.analyze_bug_closure_distribution(issues_df, events_df)
        
        # Should have data for 2024 (based on closure date)
        assert len(result) > 0
        assert 2024 in result['year'].values

    def test_analyze_engagement_heatmap_concurrent_activity(self, analyzer):
        """Test engagement heatmap with multiple users active in same hour/day"""
        now = pd.Timestamp.now(tz=timezone.utc)
        same_time = now.replace(hour=14, minute=0, second=0, microsecond=0)
        
        # Create multiple contributors with activity at the same time
        contributors = []
        for i in range(5):
            c = Contributor(f'user{i}')
            c.first_activity = same_time
            c.last_activity = same_time
            contributors.append(c)
        
        heatmap = analyzer.analyze_engagement_heatmap(contributors)
        
        # Activity should be accumulated for that specific hour
        day_name = same_time.strftime('%a')
        hour = same_time.hour
        assert heatmap.loc[day_name, hour] == 5

    def test_analyze_lifecycle_stages_timezone_edge_cases(self, analyzer):
        """Test lifecycle stages with different timezone scenarios"""
        # Create contributors with activities in different timezones
        now = pd.Timestamp.now(tz=timezone.utc)
        
        c1 = Contributor('user_utc')
        c1.first_activity = now - timedelta(days=100)
        c1.last_activity = now - timedelta(days=50)
        
        c2 = Contributor('user_naive')
        c2.first_activity = (now - timedelta(days=100)).replace(tzinfo=None)
        c2.last_activity = (now - timedelta(days=50)).replace(tzinfo=None)
        
        result = analyzer.analyze_lifecycle_stages([c1, c2])
        
        # Should handle both timezone-aware and naive datetimes
        assert len(result) == 2
        assert result['stage'].notna().all()

    def test_compute_unique_commenters_monthly_aggregation(self, analyzer):
        """Test that unique commenters are correctly aggregated by month"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'open',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-03-20'),
             'labels': ['kind/bug']}
        ])
        
        # Multiple comments from same user in same month should count as 1
        # Multiple comments from different users should count separately
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user2',
             'event_date': pd.Timestamp('2023-02-05 10:00:00'), 'label': None, 'comment': 'comment1'},
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user2',
             'event_date': pd.Timestamp('2023-02-15 14:00:00'), 'label': None, 'comment': 'comment2'},
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user3',
             'event_date': pd.Timestamp('2023-02-20 16:00:00'), 'label': None, 'comment': 'comment3'},
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user2',
             'event_date': pd.Timestamp('2023-03-05 09:00:00'), 'label': None, 'comment': 'comment4'}
        ])
        
        result = analyzer.compute_unique_commenters(events_df, issues_df)
        
        # February should have 2 unique commenters (user2 and user3)
        feb_data = result[(result['issue_number'] == 1) & (result['month'].dt.month == 2)]
        if not feb_data.empty:
            assert feb_data['n_unique_commenters'].iloc[0] == 2
        
        # March should have 1 unique commenter (user2)
        mar_data = result[(result['issue_number'] == 1) & (result['month'].dt.month == 3)]
        if not mar_data.empty:
            assert mar_data['n_unique_commenters'].iloc[0] == 1

    def test_analyze_top_active_users_per_year_edge_cases(self, analyzer):
        """Test top active users with edge cases: single year, no activity"""
        # Single contributor, single year
        c1 = Contributor('user1')
        c1.first_activity = pd.Timestamp('2023-06-15', tz=timezone.utc)
        c1.last_activity = pd.Timestamp('2023-06-20', tz=timezone.utc)
        
        result = analyzer.analyze_top_active_users_per_year([c1])
        
        assert 2023 in result
        assert len(result[2023]) == 1
        assert result[2023]['user'].iloc[0] == 'user1'

    def test_analyze_top_active_users_activity_calculation(self, analyzer):
        """Test that activity is correctly calculated (issues + comments)"""
        c1 = Contributor('user1')
        c1.first_activity = pd.Timestamp('2023-01-15', tz=timezone.utc)
        c1.last_activity = pd.Timestamp('2023-12-20', tz=timezone.utc)
        c1.issues_created = [1, 2, 3]  # 3 issues
        c1.comments = [
            type('obj', (object,), {'created_at': pd.Timestamp('2023-06-15', tz=timezone.utc)})(),
            type('obj', (object,), {'created_at': pd.Timestamp('2023-07-20', tz=timezone.utc)})()
        ]  # 2 comments
        
        result = analyzer.analyze_top_active_users_per_year([c1])
        
        # Total activity should be 5 (3 issues + 2 comments)
        assert result[2023]['activity'].iloc[0] == 5

    def test_analyze_docs_issues_empty_events(self, analyzer):
        """Test docs issues analysis with no events data"""
        class MockLoader:
            def filter_by_label(self, df, label):
                return df[df['labels'].apply(lambda L: any('doc' in str(l).lower() for l in L))]
        
        docs_issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['area/docs']}
        ])
        
        empty_events = pd.DataFrame(columns=['issue_number', 'event_type', 'event_author', 'event_date', 'label', 'comment'])
        
        status_counts, avg_commenters = analyzer.analyze_docs_issues(docs_issues_df, empty_events, MockLoader())
        
        # Should handle empty events gracefully
        assert status_counts is not None
        assert 'closed' in status_counts['state'].values

    def test_analyze_docs_issues_state_distribution(self, analyzer):
        """Test docs issues state distribution calculation"""
        class MockLoader:
            def filter_by_label(self, df, label):
                return df[df['labels'].apply(lambda L: any('doc' in str(l).lower() for l in L))]
        
        docs_issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['area/docs']},
            {'number': 2, 'creator': 'user2', 'state': 'open',
             'created_date': pd.Timestamp('2023-02-10'),
             'updated_date': pd.Timestamp('2023-02-12'),
             'labels': ['area/docs']},
            {'number': 3, 'creator': 'user3', 'state': 'closed',
             'created_date': pd.Timestamp('2023-03-05'),
             'updated_date': pd.Timestamp('2023-03-10'),
             'labels': ['area/docs']}
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'userA',
             'event_date': pd.Timestamp('2023-01-18'), 'label': None, 'comment': 'test'}
        ])
        
        status_counts, avg_commenters = analyzer.analyze_docs_issues(docs_issues_df, events_df, MockLoader())
        
        # Should have 2 closed and 1 open
        closed_count = status_counts[status_counts['state'] == 'closed']['count'].iloc[0]
        open_count = status_counts[status_counts['state'] == 'open']['count'].iloc[0]
        
        assert closed_count == 2
        assert open_count == 1

    def test_label_filtering_case_insensitive(self, analyzer):
        """Test that label filtering handles mixed case correctly"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['Kind/Bug', 'Priority/HIGH']},  # Mixed case
            {'number': 2, 'creator': 'user2', 'state': 'open',
             'created_date': pd.Timestamp('2023-02-10'),
             'updated_date': pd.Timestamp('2023-02-12'),
             'labels': ['kind/FEATURE', 'area/DOCS']}  # All caps
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'closed', 'event_author': 'user1',
             'event_date': pd.Timestamp('2023-01-20'), 'label': None, 'comment': None}
        ])
        
        # Should handle case-insensitive label matching
        result = analyzer.analyze_bug_closure_distribution(issues_df, events_df)
        assert len(result) > 0

    def test_label_filtering_special_characters(self, analyzer):
        """Test label filtering with special characters and spaces"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['kind/bug-fix', 'area: documentation']},  # Special chars
            {'number': 2, 'creator': 'user2', 'state': 'open',
             'created_date': pd.Timestamp('2023-02-10'),
             'updated_date': pd.Timestamp('2023-02-12'),
             'labels': ['kind/feature_request', 'good-first-issue']}
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user3',
             'event_date': pd.Timestamp('2023-01-18'), 'label': None, 'comment': 'test'}
        ])
        
        contributors = analyzer.build_contributors(issues_df, events_df)
        assert len(contributors) == 3

    def test_large_dataset_performance(self, analyzer):
        """Test handling of large datasets (1000+ issues)"""
        # Create 1000 issues
        issues_df = pd.DataFrame([
            {'number': i, 'creator': f'user{i % 100}', 'state': 'closed' if i % 2 == 0 else 'open',
             'created_date': pd.Timestamp('2023-01-01') + timedelta(days=i % 365),
             'updated_date': pd.Timestamp('2023-01-01') + timedelta(days=i % 365, hours=5),
             'labels': ['kind/bug' if i % 3 == 0 else 'kind/feature']}
            for i in range(1000)
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': i, 'event_type': 'commented', 'event_author': f'user{(i+1) % 100}',
             'event_date': pd.Timestamp('2023-01-01') + timedelta(days=i % 365, hours=2),
             'label': None, 'comment': f'comment{i}'}
            for i in range(0, 1000, 10)  # Every 10th issue has a comment
        ])
        
        # Should complete without errors or excessive time
        contributors = analyzer.build_contributors(issues_df, events_df)
        assert len(contributors) == 100  # 100 unique users
        
        result = analyzer.analyze_top_active_users_per_year(contributors)
        assert len(result) > 0

    def test_null_and_missing_dates(self, analyzer):
        """Test handling of null/NaT dates in various scenarios"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user1', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.NaT,  # NaT (Not a Time)
             'labels': ['kind/bug']},
            {'number': 2, 'creator': 'user2', 'state': 'open',
             'created_date': None,  # None
             'updated_date': pd.Timestamp('2023-02-12'),
             'labels': ['kind/feature']},
            {'number': 3, 'creator': 'user3', 'state': 'closed',
             'created_date': pd.Timestamp('2023-03-05'),
             'updated_date': pd.Timestamp('2023-03-10'),
             'labels': ['kind/bug']}
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user1',
             'event_date': pd.NaT, 'label': None, 'comment': 'test'},
            {'issue_number': 3, 'event_type': 'commented', 'event_author': 'user2',
             'event_date': pd.Timestamp('2023-03-08'), 'label': None, 'comment': 'test2'}
        ])
        
        # Should handle NaT/None dates gracefully
        contributors = analyzer.build_contributors(issues_df, events_df)
        assert len(contributors) > 0

    def test_special_usernames(self, analyzer):
        """Test handling of usernames with special characters"""
        issues_df = pd.DataFrame([
            {'number': 1, 'creator': 'user-name', 'state': 'closed',
             'created_date': pd.Timestamp('2023-01-15'),
             'updated_date': pd.Timestamp('2023-01-20'),
             'labels': ['kind/bug']},
            {'number': 2, 'creator': 'user_name_2', 'state': 'open',
             'created_date': pd.Timestamp('2023-02-10'),
             'updated_date': pd.Timestamp('2023-02-12'),
             'labels': ['kind/feature']},
            {'number': 3, 'creator': 'user.name.3', 'state': 'closed',
             'created_date': pd.Timestamp('2023-03-05'),
             'updated_date': pd.Timestamp('2023-03-10'),
             'labels': ['kind/bug']}
        ])
        
        events_df = pd.DataFrame([
            {'issue_number': 1, 'event_type': 'commented', 'event_author': 'user-name',
             'event_date': pd.Timestamp('2023-01-18'), 'label': None, 'comment': 'test'}
        ])
        
        contributors = analyzer.build_contributors(issues_df, events_df)
        
        usernames = [c.username for c in contributors]
        assert 'user-name' in usernames
        assert 'user_name_2' in usernames
        assert 'user.name.3' in usernames

    def test_lifecycle_stages_graduated_threshold(self, analyzer):
        """Test graduated contributor threshold (6+ months inactive)"""
        now = pd.Timestamp.now(tz=timezone.utc)
        
        # Exactly 6 months inactive (should be graduated)
        c1 = Contributor('exactly_6_months')
        c1.first_activity = now - timedelta(days=365)
        c1.last_activity = now - timedelta(days=183)  # ~6 months
        
        # Just under 6 months (should be active)
        c2 = Contributor('under_6_months')
        c2.first_activity = now - timedelta(days=365)
        c2.last_activity = now - timedelta(days=175)  # <6 months
        
        # Over 6 months (should be graduated)
        c3 = Contributor('over_6_months')
        c3.first_activity = now - timedelta(days=365)
        c3.last_activity = now - timedelta(days=200)  # >6 months
        
        result = analyzer.analyze_lifecycle_stages([c1, c2, c3], reference_date=now)
        
        stages = result.set_index('contributor')['stage'].to_dict()
        assert stages['exactly_6_months'] == 'Graduated Contributor'
        assert stages['under_6_months'] in ['Active', 'Core Maintainer']
        assert stages['over_6_months'] == 'Graduated Contributor'

    def test_engagement_heatmap_all_days_covered(self, analyzer):
        """Test that engagement heatmap includes all 7 days and 24 hours"""
        # Create contributors with activity spread across different days/hours
        contributors = []
        for day in range(7):
            for hour in range(0, 24, 6):  # Every 6 hours
                c = Contributor(f'user_d{day}_h{hour}')
                # Create timestamp for specific day and hour
                base_date = pd.Timestamp('2023-01-02', tz=timezone.utc)  # Monday
                activity_time = base_date + timedelta(days=day, hours=hour)
                c.first_activity = activity_time
                c.last_activity = activity_time
                contributors.append(c)
        
        heatmap = analyzer.analyze_engagement_heatmap(contributors)
        
        # All days should have some activity
        assert (heatmap.sum(axis=1) > 0).all()
        
        # Some hours should have activity
        assert (heatmap.sum(axis=0) > 0).any()
