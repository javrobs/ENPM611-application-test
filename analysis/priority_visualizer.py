"""
Priority Visualization Module
Generates bar charts for priority distribution and complexity scores.
"""

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


class PriorityVisualizer:
    """
    Creates visualizations for priority prediction results.
    """
    
    def __init__(self, predictions):
        """
        Initialize visualizer with predictions data.
        
        Args:
            predictions (list): List of prediction dictionaries
        """
        self.predictions = predictions
    
    def create_visualizations(self, output_dir='output/visualizations'):
        """
        Generate all visualizations and save to files.
        
        Args:
            output_dir (str): Directory to save visualization images
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure with 3 subplots
        fig = plt.figure(figsize=(20, 6))
        
        # Create grid: 2 equal plots on left, 1 larger plot on right
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        
        fig.suptitle('Priority Prediction Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        # Plot 1: Priority Distribution
        self._plot_priority_distribution(ax1)
        
        # Plot 2: Complexity Score Distribution
        self._plot_complexity_distribution(ax2)
        
        # Plot 3: Priority-Complexity Overlap (NEW!)
        self._plot_priority_complexity_overlap(ax3)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 0.98, 0.96])  # Leave space for legend on right
        output_path = f'{output_dir}/priority_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualizations saved to {output_path}")
        
        # Show the plot (optional - comment out if running in non-interactive environment)
        # plt.show()
        
        plt.close()
    
    def _plot_priority_distribution(self, ax):
        """
        Plot bar chart of priority distribution.
        
        Args:
            ax: Matplotlib axis object
        """
        # Count priorities
        priorities = [p['predicted_priority'] for p in self.predictions]
        priority_counts = Counter(priorities)
        
        # Define priority order and colors
        priority_order = ['Critical', 'High', 'Medium', 'Low']
        colors = {
            'Critical': '#dc2626',  # Red
            'High': '#ea580c',      # Orange
            'Medium': '#eab308',    # Yellow
            'Low': '#16a34a'        # Green
        }
        
        # Prepare data
        labels = []
        counts = []
        bar_colors = []
        
        for priority in priority_order:
            if priority in priority_counts:
                labels.append(priority)
                counts.append(priority_counts[priority])
                bar_colors.append(colors[priority])
        
        # Create bar chart
        bars = ax.bar(labels, counts, color=bar_colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Priority Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Issues', fontsize=12, fontweight='bold')
        ax.set_title('Issue Priority Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add total count
        total = sum(counts)
        ax.text(0.5, 0.98, f'Total Issues: {total}', 
               transform=ax.transAxes,
               ha='center', va='top',
               fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_complexity_distribution(self, ax):
        """
        Plot bar chart of complexity score ranges.
        
        Args:
            ax: Matplotlib axis object
        """
        # Extract complexity scores
        complexity_scores = [p['complexity_score'] for p in self.predictions]
        
        # Define ranges
        ranges = [
            (0, 25, 'Simple\n(0-25)'),
            (25, 50, 'Moderate\n(25-50)'),
            (50, 75, 'Complex\n(50-75)'),
            (75, 101, 'Highly Complex\n(75-100)')
        ]
        
        colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']  # Green, Blue, Orange, Red
        
        # Count issues in each range
        labels = []
        counts = []
        bar_colors = []
        
        for (low, high, label), color in zip(ranges, colors):
            count = sum(1 for score in complexity_scores if low <= score < high)
            labels.append(label)
            counts.append(count)
            bar_colors.append(color)
        
        # Create bar chart
        bars = ax.bar(labels, counts, color=bar_colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Formatting
        ax.set_xlabel('Complexity Score Range', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Issues', fontsize=12, fontweight='bold')
        ax.set_title('Issue Complexity Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add statistics
        mean_complexity = np.mean(complexity_scores)
        median_complexity = np.median(complexity_scores)
        
        stats_text = f'Mean: {mean_complexity:.1f}\nMedian: {median_complexity:.1f}'
        ax.text(0.98, 0.98, stats_text, 
               transform=ax.transAxes,
               ha='right', va='top',
               fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    def _plot_priority_complexity_overlap(self, ax):
        """
        Plot grouped bar chart showing priority-complexity overlap.
        X-axis: Priorities
        Grouped bars: Different complexity ranges
        Y-axis: Number of issues
        
        Args:
            ax: Matplotlib axis object
        """
        # Define categories
        priority_order = ['Critical', 'High', 'Medium', 'Low']
        complexity_ranges = [
            (0, 25, 'Simple (0-25)'),
            (25, 50, 'Moderate (25-50)'),
            (50, 75, 'Complex (50-75)'),
            (75, 101, 'Highly Complex (75-100)')
        ]
        
        complexity_colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
        
        # Build matrix: priority x complexity
        data_matrix = {}
        for priority in priority_order:
            data_matrix[priority] = {}
            for low, high, label in complexity_ranges:
                count = sum(1 for p in self.predictions 
                           if p['predicted_priority'] == priority 
                           and low <= p['complexity_score'] < high)
                data_matrix[priority][label] = count
        
        # Prepare data for grouped bar chart
        x = np.arange(len(priority_order))
        width = 0.2  # Width of each bar
        multiplier = 0
        
        # Plot bars for each complexity range
        for (low, high, label), color in zip(complexity_ranges, complexity_colors):
            counts = [data_matrix[priority][label] for priority in priority_order]
            offset = width * multiplier
            bars = ax.bar(x + offset, counts, width, label=label, 
                         color=color, edgecolor='black', linewidth=1)
            
            # Add value labels on bars (only if > 0)
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            multiplier += 1
        
        # Formatting
        ax.set_xlabel('Priority Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Issues', fontsize=12, fontweight='bold')
        ax.set_title('Priority-Complexity Overlap Analysis', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(priority_order)
        
        # Place legend outside the plot area to avoid hiding data
        ax.legend(title='Complexity Range', loc='upper left', bbox_to_anchor=(1.02, 1), 
                 fontsize=9, frameon=True, fancybox=True, shadow=True)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add "Quick Wins" annotation if there are High priority + Simple issues
        high_simple = data_matrix.get('High', {}).get('Simple (0-25)', 0)
        if high_simple > 0:
            ax.text(0.02, 0.98, f'🔴 Quick Wins:\n{high_simple} High Priority\nSimple Issues', 
                   transform=ax.transAxes,
                   ha='left', va='top',
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='#fee2e2', alpha=0.8, edgecolor='#dc2626', linewidth=2))
    
    def print_summary_statistics(self):
        """
        Print summary statistics about predictions.
        """
        print("\n" + "="*60)
        print("PREDICTION SUMMARY STATISTICS")
        print("="*60)
        
        # Priority distribution
        priorities = [p['predicted_priority'] for p in self.predictions]
        priority_counts = Counter(priorities)
        
        print("\n✓ Priority Distribution:")
        for priority in ['Critical', 'High', 'Medium', 'Low']:
            if priority in priority_counts:
                count = priority_counts[priority]
                percentage = (count / len(priorities)) * 100
                print(f"  {priority}: {count} issues ({percentage:.1f}%)")
        
        # Complexity statistics
        complexity_scores = [p['complexity_score'] for p in self.predictions]
        
        print("\n✓ Complexity Statistics:")
        print(f"  Mean: {np.mean(complexity_scores):.1f}")
        print(f"  Median: {np.median(complexity_scores):.1f}")
        print(f"  Min: {min(complexity_scores)}")
        print(f"  Max: {max(complexity_scores)}")
        print(f"  Std Dev: {np.std(complexity_scores):.1f}")
        
        # Complexity ranges
        print("\n✓ Complexity Range Distribution:")
        ranges = [
            (0, 25, 'Simple'),
            (25, 50, 'Moderate'),
            (50, 75, 'Complex'),
            (75, 101, 'Highly Complex')
        ]
        
        for low, high, label in ranges:
            count = sum(1 for score in complexity_scores if low <= score < high)
            percentage = (count / len(complexity_scores)) * 100
            print(f"  {label} ({low}-{high-1}): {count} issues ({percentage:.1f}%)")
        
        # Priority-Complexity overlap (NEW!)
        print("\n✓ Priority-Complexity Overlap:")
        priority_order = ['Critical', 'High', 'Medium', 'Low']
        
        for priority in priority_order:
            priority_issues = [p for p in self.predictions if p['predicted_priority'] == priority]
            if not priority_issues:
                continue
            
            print(f"\n  {priority} Priority ({len(priority_issues)} issues):")
            for low, high, label in ranges:
                count = sum(1 for p in priority_issues if low <= p['complexity_score'] < high)
                if count > 0:
                    percentage = (count / len(priority_issues)) * 100
                    print(f"    - {label}: {count} ({percentage:.1f}%)")
        
        # Highlight "Quick Wins"
        high_simple = sum(1 for p in self.predictions 
                         if p['predicted_priority'] == 'High' 
                         and p['complexity_score'] < 25)
        
        if high_simple > 0:
            print(f"\n🔴 QUICK WINS IDENTIFIED: {high_simple} High Priority + Simple Complexity issues!")