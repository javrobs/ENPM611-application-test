from data_loader import DataLoader
from analysis.contributors_analyzer import ContributorsAnalyzer
from visualization.visualizer import Visualizer
import pandas as pd

class ContributorsController:
    def __init__(self):
        self.data_loader = DataLoader()
        self.analyzer = ContributorsAnalyzer()
        self.visualizer = Visualizer()

    def load_contributor_data(self):
        issues_df = self.data_loader.parse_issues()
        events_df = self.data_loader.parse_events(issues_df)
        return issues_df, events_df

    def plot_bug_closure_distribution(self, issues_df, events_df):
        """Controller method for Graph 1: Bug Closure Distribution. Analyzes yearly 
           bug closures and plots the share handled by the top 5 contributors 
           vs the rest of the community."""
        yearly_distribution = self.analyzer.analyze_bug_closure_distribution(issues_df, events_df)

        # Printing top 5 bug closers per year in CLI
        for _, row in yearly_distribution.iterrows():
            print(f"Year {int(row['year'])}: Top 5 bug closers -> {row['top5_users']}")

        fig = self.visualizer.create_bug_closure_distribution_chart(
            yearly_distribution,
            "Community Load Distribution: % of Bug Closures by Top 5 vs Rest"
        )
        return fig

    def plot_top_feature_requesters(self, issues_df, top_n=10):
        """Controller method for Graph 2: Top 10 Feature Requesters (open vs closed).
           Returns a figure showing stacked bars for top requesters."""
        top_requesters, feature_issues = self.analyzer.analyze_top_feature_requesters(issues_df, top_n=top_n)
        if top_requesters is None:
            return None

        fig = self.visualizer.create_top_feature_requesters_chart(
            top_requesters,
            feature_issues,
            "Top 10 Contributors by Feature Requests Submitted"
        )
        return fig

    def plot_docs_issues(self, issues_df, events_df):
        """Controller method for Graph 3: Documentation Issues (open vs closed per month).
            Returns a figure with stacked bars for issue counts and a line showing the
            average number of unique commenters per doc issue each month."""
        status_counts, avg_commenters = self.analyzer.analyze_docs_issues(issues_df, events_df, self.data_loader)
        if status_counts is None:
            return None
        return self.visualizer.create_docs_issues_chart(
            status_counts,
            avg_commenters,
            "Docs Issues: Open vs Closed per Month with Avg Commenters"
        )
    
    def plot_issues_created_per_user(self, issues_df, top_n=40):
        """Controller method for Graph 4: Top 40 Contributors by Issues Created.
           Returns a figure showing the top 40 users ranked by number of issues created."""
        issues_per_user, all_counts = self.analyzer.analyze_issues_created_per_user(issues_df, top_n=top_n)
        if issues_per_user is None:
            return None

        return self.visualizer.create_issues_created_per_user_chart(
            issues_per_user,
            all_counts,
            f"Top {top_n} Contributors by Issues Created"
        )

    def plot_top_active_users_per_year(self, contributors, top_n=10):
        """Controller method for Graph 5: Top Active Users per Year.
        """
        yearly_data = self.analyzer.analyze_top_active_users_per_year(contributors)

        return self.visualizer.create_top_active_users_per_year_chart(yearly_data, top_n)
    
    def run_engagement_heatmap(self, contributors):
        """Controller method for Graph 6: Engagement Heatmap.
           Analyzes contributor activity to produce a heatmap of engagement
           across days of the week and hours of the day.
        """
        # rows = days of the week, columns = hours of the day, values = activity counts
        heatmap_df = self.analyzer.analyze_engagement_heatmap(contributors)
        
        # Normalize each day's row so values represent percentages of that day's total activity
        heatmap_norm = heatmap_df.div(heatmap_df.sum(axis=1), axis=0).fillna(0) * 100

        # === Overall busiest hours across all days ===
        print("\n=== Overall Busiest Hours (across all days) ===")
        avg_by_hour = heatmap_norm.mean(axis=0).sort_values(ascending=False)
        for hour, val in avg_by_hour.head(5).items():
            print(f"Hour {hour:02d}: {val:.2f}% (average share of a day's activity)")

        # === Top 3 Busy Hours per day ===
        print("\n=== Top 3 Busy Hours Per Day ===")
        for day in heatmap_norm.index:
            top3 = heatmap_norm.loc[day].sort_values(ascending=False).head(3)
            total_top3 = top3.sum()
            print(f"\n{day}:")
            for hour, val in top3.items():
                print(f"  Hour {hour:02d} → {val:.2f}% of {day}'s activity")
            print(f"  Total (Top 3) → {total_top3:.2f}% of {day}'s activity")

        fig = self.visualizer.create_engagement_heatmap_chart(heatmap_df)
        return fig
    
    def run_contributor_lifecycle(self, contributors):
        """Controller method for Graph 7: Contributor Lifecycle Stages
        """
        df = self.analyzer.analyze_lifecycle_stages(contributors)

        # Counting how many contributors fall into each stage and calculate percentages
        summary = df["stage"].value_counts().to_frame("count")
        summary["%"] = summary["count"] / summary["count"].sum() * 100

        # Finding the latest activity date across all contributors
        latest_date = pd.concat([df["first_activity"].dropna(), df["last_activity"].dropna()]).max()

        fig = self.visualizer.create_lifecycle_chart(summary, latest_date=latest_date)
        return fig
    
    






