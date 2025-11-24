"""
White Box Testing for Feature 2 - Contributors Dashboard
Tests internal logic paths, branches, and data flow
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from app.feature_runner import FeatureRunner


class TestFeature2WhiteBox(unittest.TestCase):
    """White box tests for Contributors Dashboard feature"""
    
    def setUp(self):
        """Set up test fixtures before each test"""
        self.runner = FeatureRunner()
        self.runner.config = Mock()
        self.runner.config.get_data_path.return_value = "test/data"
        self.runner.config.get_output_path.return_value = "test/output"
        self.runner.contributors_controller = Mock()
        
        # Sample test data
        self.sample_issues = pd.DataFrame({
            'id': [1, 2, 3],
            'created_at': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'user': ['user1', 'user2', 'user1']
        })
        
        self.sample_events = pd.DataFrame({
            'issue_id': [1, 2, 3],
            'event': ['closed', 'labeled', 'closed'],
            'created_at': ['2023-01-10', '2023-02-05', '2023-03-15']
        })
    
    def test_feature2_entry_point(self):
        """Test feature 2 execution path is reached"""
        self.runner.contributors_controller.load_contributor_data.return_value = (
            self.sample_issues, self.sample_events
        )
        self.runner.contributors_controller.analyzer = Mock()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = []
        
        # Mock all plot methods to return None
        self.runner.contributors_controller.plot_bug_closure_distribution.return_value = Mock()
        self.runner.contributors_controller.plot_top_feature_requesters.return_value = None
        self.runner.contributors_controller.plot_docs_issues.return_value = None
        self.runner.contributors_controller.plot_issues_created_per_user.return_value = None
        self.runner.contributors_controller.plot_top_active_users_per_year.return_value = None
        self.runner.contributors_controller.run_engagement_heatmap.return_value = None
        self.runner.contributors_controller.run_contributor_lifecycle.return_value = None
        
        # Execute feature 2
        with patch('builtins.print'):
            self.runner.run_feature(2)
        
        # Verify load_contributor_data was called
        self.runner.contributors_controller.load_contributor_data.assert_called_once()
    
    def test_data_loading_branch(self):
        """Test data loading logic branch"""
        self.runner.contributors_controller.load_contributor_data.return_value = (
            self.sample_issues, self.sample_events
        )
        self.runner.contributors_controller.analyzer = Mock()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = []
        
        # Mock plots
        self.runner.contributors_controller.plot_bug_closure_distribution.return_value = Mock()
        self.runner.contributors_controller.plot_top_feature_requesters.return_value = None
        self.runner.contributors_controller.plot_docs_issues.return_value = None
        self.runner.contributors_controller.plot_issues_created_per_user.return_value = None
        self.runner.contributors_controller.plot_top_active_users_per_year.return_value = None
        self.runner.contributors_controller.run_engagement_heatmap.return_value = None
        self.runner.contributors_controller.run_contributor_lifecycle.return_value = None
        
        with patch('builtins.print'):
            self.runner.run_feature(2)
        
        # Verify config methods called for paths
        self.runner.config.get_data_path.assert_called_once()
        self.runner.config.get_output_path.assert_called_once()
    
    def test_build_contributors_execution(self):
        """Test contributors building logic path"""
        mock_contributors = [{'user': 'user1', 'activity': 10}]
        
        self.runner.contributors_controller.load_contributor_data.return_value = (
            self.sample_issues, self.sample_events
        )
        self.runner.contributors_controller.analyzer = Mock()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = mock_contributors
        
        # Mock all plots
        self.runner.contributors_controller.plot_bug_closure_distribution.return_value = Mock()
        self.runner.contributors_controller.plot_top_feature_requesters.return_value = None
        self.runner.contributors_controller.plot_docs_issues.return_value = None
        self.runner.contributors_controller.plot_issues_created_per_user.return_value = None
        self.runner.contributors_controller.plot_top_active_users_per_year.return_value = None
        self.runner.contributors_controller.run_engagement_heatmap.return_value = None
        self.runner.contributors_controller.run_contributor_lifecycle.return_value = None
        
        with patch('builtins.print'):
            self.runner.run_feature(2)
        
        # Verify build_contributors called with loaded data
        self.runner.contributors_controller.analyzer.build_contributors.assert_called_once_with(
            self.sample_issues, self.sample_events
        )
    
    def test_graph1_always_added(self):
        """Test Graph 1 (bug closures) always added to figs dict"""
        mock_fig1 = Mock()
        
        self.runner.contributors_controller.load_contributor_data.return_value = (
            self.sample_issues, self.sample_events
        )
        self.runner.contributors_controller.analyzer = Mock()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = []
        self.runner.contributors_controller.plot_bug_closure_distribution.return_value = mock_fig1
        
        # Other plots return None
        self.runner.contributors_controller.plot_top_feature_requesters.return_value = None
        self.runner.contributors_controller.plot_docs_issues.return_value = None
        self.runner.contributors_controller.plot_issues_created_per_user.return_value = None
        self.runner.contributors_controller.plot_top_active_users_per_year.return_value = None
        self.runner.contributors_controller.run_engagement_heatmap.return_value = None
        self.runner.contributors_controller.run_contributor_lifecycle.return_value = None
        
        self.runner.contributors_controller.visualizer = Mock()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Graph 1 should be saved
        self.runner.contributors_controller.visualizer.save_figure.assert_any_call(
            mock_fig1, "test/output/graph1_bug_closures.png"
        )
    
    def test_conditional_graph2_added_when_not_none(self):
        """Test Graph 2 only added if not None"""
        mock_fig2 = Mock()
        
        self.runner.contributors_controller.load_contributor_data.return_value = (
            self.sample_issues, self.sample_events
        )
        self.runner.contributors_controller.analyzer = Mock()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = []
        
        self.runner.contributors_controller.plot_bug_closure_distribution.return_value = Mock()
        self.runner.contributors_controller.plot_top_feature_requesters.return_value = mock_fig2
        self.runner.contributors_controller.plot_docs_issues.return_value = None
        self.runner.contributors_controller.plot_issues_created_per_user.return_value = None
        self.runner.contributors_controller.plot_top_active_users_per_year.return_value = None
        self.runner.contributors_controller.run_engagement_heatmap.return_value = None
        self.runner.contributors_controller.run_contributor_lifecycle.return_value = None
        
        self.runner.contributors_controller.visualizer = Mock()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Graph 2 should be saved since it's not None
        self.runner.contributors_controller.visualizer.save_figure.assert_any_call(
            mock_fig2, "test/output/graph2_top_feature_requesters.png"
        )
    
    def test_conditional_graph2_skipped_when_none(self):
        """Test Graph 2 skipped when None"""
        self.runner.contributors_controller.load_contributor_data.return_value = (
            self.sample_issues, self.sample_events
        )
        self.runner.contributors_controller.analyzer = Mock()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = []
        
        self.runner.contributors_controller.plot_bug_closure_distribution.return_value = Mock()
        self.runner.contributors_controller.plot_top_feature_requesters.return_value = None
        self.runner.contributors_controller.plot_docs_issues.return_value = None
        self.runner.contributors_controller.plot_issues_created_per_user.return_value = None
        self.runner.contributors_controller.plot_top_active_users_per_year.return_value = None
        self.runner.contributors_controller.run_engagement_heatmap.return_value = None
        self.runner.contributors_controller.run_contributor_lifecycle.return_value = None
        
        self.runner.contributors_controller.visualizer = Mock()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Verify graph2 not in saved calls
        save_calls = self.runner.contributors_controller.visualizer.save_figure.call_args_list
        graph2_saved = any('graph2' in str(call) for call in save_calls)
        self.assertFalse(graph2_saved)
    
    def test_all_conditional_graphs_paths(self):
        """Test all graphs with mixed None/not-None conditions"""
        # Mix of None and actual figures
        mock_figs = {
            'graph1': Mock(),  # Always added
            'graph2': Mock(),  # Conditionally added
            'graph3': None,    # Skipped
            'graph4': Mock(),  # Conditionally added
            'graph5': None,    # Skipped
            'graph6': Mock(),  # Conditionally added
            'graph7': None     # Skipped
        }
        
        self.runner.contributors_controller.load_contributor_data.return_value = (
            self.sample_issues, self.sample_events
        )
        self.runner.contributors_controller.analyzer = Mock()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = []
        
        self.runner.contributors_controller.plot_bug_closure_distribution.return_value = mock_figs['graph1']
        self.runner.contributors_controller.plot_top_feature_requesters.return_value = mock_figs['graph2']
        self.runner.contributors_controller.plot_docs_issues.return_value = mock_figs['graph3']
        self.runner.contributors_controller.plot_issues_created_per_user.return_value = mock_figs['graph4']
        self.runner.contributors_controller.plot_top_active_users_per_year.return_value = mock_figs['graph5']
        self.runner.contributors_controller.run_engagement_heatmap.return_value = mock_figs['graph6']
        self.runner.contributors_controller.run_contributor_lifecycle.return_value = mock_figs['graph7']
        
        self.runner.contributors_controller.visualizer = Mock()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Should have 4 save calls (graph1, 2, 4, 6)
        self.assertEqual(self.runner.contributors_controller.visualizer.save_figure.call_count, 4)
    
    def test_graph5_show_called_when_present(self):
        """Test Graph 5 interactive display when not None"""
        mock_fig5 = Mock()
        
        self.runner.contributors_controller.load_contributor_data.return_value = (
            self.sample_issues, self.sample_events
        )
        self.runner.contributors_controller.analyzer = Mock()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = []
        
        self.runner.contributors_controller.plot_bug_closure_distribution.return_value = Mock()
        self.runner.contributors_controller.plot_top_feature_requesters.return_value = None
        self.runner.contributors_controller.plot_docs_issues.return_value = None
        self.runner.contributors_controller.plot_issues_created_per_user.return_value = None
        self.runner.contributors_controller.plot_top_active_users_per_year.return_value = mock_fig5
        self.runner.contributors_controller.run_engagement_heatmap.return_value = None
        self.runner.contributors_controller.run_contributor_lifecycle.return_value = None
        
        self.runner.contributors_controller.visualizer = Mock()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Verify show() was called on fig5
        mock_fig5.show.assert_called_once()
    
    def test_matplotlib_show_called(self):
        """Test matplotlib.pyplot.show() called at end"""
        self.runner.contributors_controller.load_contributor_data.return_value = (
            self.sample_issues, self.sample_events
        )
        self.runner.contributors_controller.analyzer = Mock()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = []
        
        self.runner.contributors_controller.plot_bug_closure_distribution.return_value = Mock()
        self.runner.contributors_controller.plot_top_feature_requesters.return_value = None
        self.runner.contributors_controller.plot_docs_issues.return_value = None
        self.runner.contributors_controller.plot_issues_created_per_user.return_value = None
        self.runner.contributors_controller.plot_top_active_users_per_year.return_value = None
        self.runner.contributors_controller.run_engagement_heatmap.return_value = None
        self.runner.contributors_controller.run_contributor_lifecycle.return_value = None
        
        self.runner.contributors_controller.visualizer = Mock()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show') as mock_show:
            self.runner.run_feature(2)
        
        # Verify pyplot.show() called once
        mock_show.assert_called_once()
    
    def test_save_figure_loop_all_figs(self):
        """Test save loop iterates through all non-None figures"""
        mock_figs = [Mock() for _ in range(3)]
        
        self.runner.contributors_controller.load_contributor_data.return_value = (
            self.sample_issues, self.sample_events
        )
        self.runner.contributors_controller.analyzer = Mock()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = []
        
        # Return 3 figures, rest None
        self.runner.contributors_controller.plot_bug_closure_distribution.return_value = mock_figs[0]
        self.runner.contributors_controller.plot_top_feature_requesters.return_value = mock_figs[1]
        self.runner.contributors_controller.plot_docs_issues.return_value = mock_figs[2]
        self.runner.contributors_controller.plot_issues_created_per_user.return_value = None
        self.runner.contributors_controller.plot_top_active_users_per_year.return_value = None
        self.runner.contributors_controller.run_engagement_heatmap.return_value = None
        self.runner.contributors_controller.run_contributor_lifecycle.return_value = None
        
        self.runner.contributors_controller.visualizer = Mock()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Should save exactly 3 figures
        self.assertEqual(self.runner.contributors_controller.visualizer.save_figure.call_count, 3)
    
    def test_output_path_used_in_save(self):
        """Test output path from config used in file saves"""
        mock_fig = Mock()
        
        self.runner.contributors_controller.load_contributor_data.return_value = (
            self.sample_issues, self.sample_events
        )
        self.runner.contributors_controller.analyzer = Mock()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = []
        
        self.runner.contributors_controller.plot_bug_closure_distribution.return_value = mock_fig
        self.runner.contributors_controller.plot_top_feature_requesters.return_value = None
        self.runner.contributors_controller.plot_docs_issues.return_value = None
        self.runner.contributors_controller.plot_issues_created_per_user.return_value = None
        self.runner.contributors_controller.plot_top_active_users_per_year.return_value = None
        self.runner.contributors_controller.run_engagement_heatmap.return_value = None
        self.runner.contributors_controller.run_contributor_lifecycle.return_value = None
        
        self.runner.contributors_controller.visualizer = Mock()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Verify output path is in saved filename
        call_args = self.runner.contributors_controller.visualizer.save_figure.call_args[0]
        self.assertIn('test/output', call_args[1])


if __name__ == '__main__':
    unittest.main()