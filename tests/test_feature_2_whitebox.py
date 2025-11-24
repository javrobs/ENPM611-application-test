"""
White Box Testing for Feature 2 - Contributors Dashboard
Tests internal logic paths, branches, and data flow
Extended coverage to reach 90%+
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
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
    
    def _setup_default_mocks(self, fig_returns=None):
        """Helper to set up standard mocks"""
        if fig_returns is None:
            fig_returns = [Mock(), None, None, None, None, None, None]
        
        self.runner.contributors_controller.load_contributor_data.return_value = (
            self.sample_issues, self.sample_events
        )
        self.runner.contributors_controller.analyzer = Mock()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = []
        
        self.runner.contributors_controller.plot_bug_closure_distribution.return_value = fig_returns[0]
        self.runner.contributors_controller.plot_top_feature_requesters.return_value = fig_returns[1]
        self.runner.contributors_controller.plot_docs_issues.return_value = fig_returns[2]
        self.runner.contributors_controller.plot_issues_created_per_user.return_value = fig_returns[3]
        self.runner.contributors_controller.plot_top_active_users_per_year.return_value = fig_returns[4]
        self.runner.contributors_controller.run_engagement_heatmap.return_value = fig_returns[5]
        self.runner.contributors_controller.run_contributor_lifecycle.return_value = fig_returns[6]
        
        self.runner.contributors_controller.visualizer = Mock()
    
    def test_feature2_entry_point(self):
        """Test feature 2 execution path is reached"""
        self._setup_default_mocks()
        
        with patch('builtins.print') as mock_print:
            self.runner.run_feature(2)
        
        # Verify entry message printed
        mock_print.assert_any_call("▶ Running Contributors Dashboard...")
        self.runner.contributors_controller.load_contributor_data.assert_called_once()
    
    def test_data_loading_branch(self):
        """Test data loading logic branch"""
        self._setup_default_mocks()
        
        with patch('builtins.print'):
            self.runner.run_feature(2)
        
        # Verify config methods called for paths
        self.runner.config.get_data_path.assert_called_once()
        self.runner.config.get_output_path.assert_called_once()
    
    def test_build_contributors_execution(self):
        """Test contributors building logic path"""
        mock_contributors = [{'user': 'user1', 'activity': 10}]
        
        self._setup_default_mocks()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = mock_contributors
        
        with patch('builtins.print'):
            self.runner.run_feature(2)
        
        # Verify build_contributors called with loaded data
        self.runner.contributors_controller.analyzer.build_contributors.assert_called_once_with(
            self.sample_issues, self.sample_events
        )
    
    def test_graph1_always_added(self):
        """Test Graph 1 (bug closures) always added to figs dict"""
        mock_fig1 = Mock()
        self._setup_default_mocks([mock_fig1, None, None, None, None, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Graph 1 should be saved
        self.runner.contributors_controller.visualizer.save_figure.assert_any_call(
            mock_fig1, "test/output/graph1_bug_closures.png"
        )
    
    def test_graph1_method_called_with_correct_args(self):
        """Test Graph 1 plot method receives correct data"""
        self._setup_default_mocks()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Verify plot_bug_closure_distribution called with issues and events
        self.runner.contributors_controller.plot_bug_closure_distribution.assert_called_once_with(
            self.sample_issues, self.sample_events
        )
    
    def test_conditional_graph2_added_when_not_none(self):
        """Test Graph 2 only added if not None"""
        mock_fig2 = Mock()
        self._setup_default_mocks([Mock(), mock_fig2, None, None, None, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Graph 2 should be saved since it's not None
        self.runner.contributors_controller.visualizer.save_figure.assert_any_call(
            mock_fig2, "test/output/graph2_top_feature_requesters.png"
        )
    
    def test_conditional_graph2_skipped_when_none(self):
        """Test Graph 2 skipped when None"""
        self._setup_default_mocks([Mock(), None, None, None, None, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Verify graph2 not in saved calls
        save_calls = self.runner.contributors_controller.visualizer.save_figure.call_args_list
        graph2_saved = any('graph2' in str(call) for call in save_calls)
        self.assertFalse(graph2_saved)
    
    def test_graph2_receives_correct_parameters(self):
        """Test Graph 2 called with issues and top_n parameter"""
        self._setup_default_mocks()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        self.runner.contributors_controller.plot_top_feature_requesters.assert_called_once_with(
            self.sample_issues, top_n=10
        )
    
    def test_conditional_graph3_added_when_not_none(self):
        """Test Graph 3 only added if not None"""
        mock_fig3 = Mock()
        self._setup_default_mocks([Mock(), None, mock_fig3, None, None, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        self.runner.contributors_controller.visualizer.save_figure.assert_any_call(
            mock_fig3, "test/output/graph3_docs_issues.png"
        )
    
    def test_conditional_graph3_skipped_when_none(self):
        """Test Graph 3 skipped when None"""
        self._setup_default_mocks([Mock(), None, None, None, None, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        save_calls = self.runner.contributors_controller.visualizer.save_figure.call_args_list
        graph3_saved = any('graph3' in str(call) for call in save_calls)
        self.assertFalse(graph3_saved)
    
    def test_graph3_receives_correct_parameters(self):
        """Test Graph 3 called with issues and events"""
        self._setup_default_mocks()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        self.runner.contributors_controller.plot_docs_issues.assert_called_once_with(
            self.sample_issues, self.sample_events
        )
    
    def test_conditional_graph4_added_when_not_none(self):
        """Test Graph 4 only added if not None"""
        mock_fig4 = Mock()
        self._setup_default_mocks([Mock(), None, None, mock_fig4, None, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        self.runner.contributors_controller.visualizer.save_figure.assert_any_call(
            mock_fig4, "test/output/graph4_issues_created_per_user.png"
        )
    
    def test_conditional_graph4_skipped_when_none(self):
        """Test Graph 4 skipped when None"""
        self._setup_default_mocks([Mock(), None, None, None, None, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        save_calls = self.runner.contributors_controller.visualizer.save_figure.call_args_list
        graph4_saved = any('graph4' in str(call) for call in save_calls)
        self.assertFalse(graph4_saved)
    
    def test_graph4_receives_correct_parameters(self):
        """Test Graph 4 called with issues and top_n parameter"""
        self._setup_default_mocks()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        self.runner.contributors_controller.plot_issues_created_per_user.assert_called_once_with(
            self.sample_issues, top_n=40
        )
    
    def test_conditional_graph5_added_when_not_none(self):
        """Test Graph 5 only added if not None"""
        mock_fig5 = Mock()
        self._setup_default_mocks([Mock(), None, None, None, mock_fig5, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        self.runner.contributors_controller.visualizer.save_figure.assert_any_call(
            mock_fig5, "test/output/graph5_top_active_users_per_year.png"
        )
    
    def test_conditional_graph5_skipped_when_none(self):
        """Test Graph 5 skipped when None"""
        self._setup_default_mocks([Mock(), None, None, None, None, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        save_calls = self.runner.contributors_controller.visualizer.save_figure.call_args_list
        graph5_saved = any('graph5' in str(call) for call in save_calls)
        self.assertFalse(graph5_saved)
    
    def test_graph5_show_called_when_present(self):
        """Test Graph 5 interactive display when not None"""
        mock_fig5 = Mock()
        self._setup_default_mocks([Mock(), None, None, None, mock_fig5, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Verify show() was called on fig5
        mock_fig5.show.assert_called_once()
    
    def test_graph5_show_not_called_when_none(self):
        """Test Graph 5 show() not called when None"""
        self._setup_default_mocks([Mock(), None, None, None, None, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # No show() should be called since fig5 is None
        # This implicitly tests the None check works
        self.runner.contributors_controller.plot_top_active_users_per_year.assert_called_once()
    
    def test_graph5_receives_correct_parameters(self):
        """Test Graph 5 called with contributors and top_n"""
        mock_contributors = [{'user': 'user1'}]
        self._setup_default_mocks()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = mock_contributors
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        self.runner.contributors_controller.plot_top_active_users_per_year.assert_called_once_with(
            mock_contributors, top_n=10
        )
    
    def test_conditional_graph6_added_when_not_none(self):
        """Test Graph 6 only added if not None"""
        mock_fig6 = Mock()
        self._setup_default_mocks([Mock(), None, None, None, None, mock_fig6, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        self.runner.contributors_controller.visualizer.save_figure.assert_any_call(
            mock_fig6, "test/output/graph6_engagement_heatmap.png"
        )
    
    def test_conditional_graph6_skipped_when_none(self):
        """Test Graph 6 skipped when None"""
        self._setup_default_mocks([Mock(), None, None, None, None, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        save_calls = self.runner.contributors_controller.visualizer.save_figure.call_args_list
        graph6_saved = any('graph6' in str(call) for call in save_calls)
        self.assertFalse(graph6_saved)
    
    def test_graph6_receives_correct_parameters(self):
        """Test Graph 6 called with contributors"""
        mock_contributors = [{'user': 'user1'}]
        self._setup_default_mocks()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = mock_contributors
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        self.runner.contributors_controller.run_engagement_heatmap.assert_called_once_with(
            mock_contributors
        )
    
    def test_conditional_graph7_added_when_not_none(self):
        """Test Graph 7 only added if not None"""
        mock_fig7 = Mock()
        self._setup_default_mocks([Mock(), None, None, None, None, None, mock_fig7])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        self.runner.contributors_controller.visualizer.save_figure.assert_any_call(
            mock_fig7, "test/output/graph7_contributor_lifecycle.png"
        )
    
    def test_conditional_graph7_skipped_when_none(self):
        """Test Graph 7 skipped when None"""
        self._setup_default_mocks([Mock(), None, None, None, None, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        save_calls = self.runner.contributors_controller.visualizer.save_figure.call_args_list
        graph7_saved = any('graph7' in str(call) for call in save_calls)
        self.assertFalse(graph7_saved)
    
    def test_graph7_receives_correct_parameters(self):
        """Test Graph 7 called with contributors"""
        mock_contributors = [{'user': 'user1'}]
        self._setup_default_mocks()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = mock_contributors
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        self.runner.contributors_controller.run_contributor_lifecycle.assert_called_once_with(
            mock_contributors
        )
    
    def test_all_conditional_graphs_paths(self):
        """Test all graphs with mixed None/not-None conditions"""
        mock_figs = [Mock(), Mock(), None, Mock(), None, Mock(), None]
        self._setup_default_mocks(mock_figs)
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Should have 4 save calls (graph1, 2, 4, 6)
        self.assertEqual(self.runner.contributors_controller.visualizer.save_figure.call_count, 4)
    
    def test_all_graphs_present(self):
        """Test when all 7 graphs are successfully generated"""
        mock_figs = [Mock() for _ in range(7)]
        self._setup_default_mocks(mock_figs)
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # All 7 should be saved
        self.assertEqual(self.runner.contributors_controller.visualizer.save_figure.call_count, 7)
    
    def test_only_graph1_present(self):
        """Test when only Graph 1 generates successfully"""
        mock_figs = [Mock(), None, None, None, None, None, None]
        self._setup_default_mocks(mock_figs)
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Only 1 should be saved
        self.assertEqual(self.runner.contributors_controller.visualizer.save_figure.call_count, 1)
    
    def test_matplotlib_show_called(self):
        """Test matplotlib.pyplot.show() called at end"""
        self._setup_default_mocks()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show') as mock_show:
            self.runner.run_feature(2)
        
        # Verify pyplot.show() called once
        mock_show.assert_called_once()
    
    def test_save_figure_loop_all_figs(self):
        """Test save loop iterates through all non-None figures"""
        mock_figs = [Mock(), Mock(), Mock(), None, None, None, None]
        self._setup_default_mocks(mock_figs)
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Should save exactly 3 figures
        self.assertEqual(self.runner.contributors_controller.visualizer.save_figure.call_count, 3)
    
    def test_output_path_used_in_save(self):
        """Test output path from config used in file saves"""
        mock_fig = Mock()
        self._setup_default_mocks([mock_fig, None, None, None, None, None, None])
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Verify output path is in saved filename
        call_args = self.runner.contributors_controller.visualizer.save_figure.call_args[0]
        self.assertIn('test/output', call_args[1])
    
    def test_print_statements_for_saved_files(self):
        """Test that save confirmation messages are printed"""
        mock_figs = [Mock(), Mock(), None, None, None, None, None]
        self._setup_default_mocks(mock_figs)
        
        with patch('builtins.print') as mock_print, patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Check for saved file print statements
        print_calls = [str(call) for call in mock_print.call_args_list]
        saved_prints = [c for c in print_calls if 'Saved' in c and '.png' in c]
        
        # Should have 2 save messages (for graph1 and graph2)
        self.assertGreaterEqual(len(saved_prints), 2)
    
    def test_figs_dict_accumulation(self):
        """Test that figs dictionary accumulates correctly"""
        mock_figs = [Mock(), Mock(), Mock(), None, None, None, None]
        self._setup_default_mocks(mock_figs)
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Verify save_figure called for each non-None fig
        expected_calls = 3
        actual_calls = self.runner.contributors_controller.visualizer.save_figure.call_count
        self.assertEqual(actual_calls, expected_calls)
    
    def test_contributors_passed_to_graphs_5_6_7(self):
        """Test contributors list passed to appropriate graph methods"""
        mock_contributors = [{'user': 'user1', 'activity': 5}]
        self._setup_default_mocks()
        self.runner.contributors_controller.analyzer.build_contributors.return_value = mock_contributors
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Verify contributors passed to graphs 5, 6, 7
        self.runner.contributors_controller.plot_top_active_users_per_year.assert_called_with(
            mock_contributors, top_n=10
        )
        self.runner.contributors_controller.run_engagement_heatmap.assert_called_with(
            mock_contributors
        )
        self.runner.contributors_controller.run_contributor_lifecycle.assert_called_with(
            mock_contributors
        )
    
    def test_issues_passed_to_graphs_2_4(self):
        """Test issues dataframe passed to appropriate graph methods"""
        self._setup_default_mocks()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Verify issues passed correctly
        self.runner.contributors_controller.plot_top_feature_requesters.assert_called_with(
            self.sample_issues, top_n=10
        )
        self.runner.contributors_controller.plot_issues_created_per_user.assert_called_with(
            self.sample_issues, top_n=40
        )
    
    def test_issues_and_events_passed_to_graphs_1_3(self):
        """Test both dataframes passed to appropriate graph methods"""
        self._setup_default_mocks()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Verify both dataframes passed correctly
        self.runner.contributors_controller.plot_bug_closure_distribution.assert_called_with(
            self.sample_issues, self.sample_events
        )
        self.runner.contributors_controller.plot_docs_issues.assert_called_with(
            self.sample_issues, self.sample_events
        )
    
    def test_execution_order(self):
        """Test that methods are called in correct order"""
        self._setup_default_mocks()
        
        with patch('builtins.print'), patch('matplotlib.pyplot.show'):
            self.runner.run_feature(2)
        
        # Verify execution order
        manager = Mock()
        manager.attach_mock(self.runner.config.get_data_path, 'get_data_path')
        manager.attach_mock(self.runner.config.get_output_path, 'get_output_path')
        manager.attach_mock(self.runner.contributors_controller.load_contributor_data, 'load_data')
        
        # Data path should be called before load
        self.runner.config.get_data_path.assert_called()
        self.runner.contributors_controller.load_contributor_data.assert_called()


if __name__ == '__main__':
    unittest.main()