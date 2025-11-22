import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd


class Visualizer:
    def create_bug_closure_distribution_chart(self, yearly_distribution, title: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotting stacked bars: top 5 bug closers vs the rest of the contributors
        ax.bar(yearly_distribution["year"], yearly_distribution["top5_pct"],
            label="Top 5 Closers", color="steelblue")
        ax.bar(yearly_distribution["year"], yearly_distribution["rest_pct"],
            bottom=yearly_distribution["top5_pct"], label="Rest", color="lightgray")

        ax.set_ylabel("Percentage of Bug Closures (%)")
        ax.set_xlabel("Year")
        ax.set_title(title)
        ax.legend()

        # Adding percentage labels inside the bars
        for i, year in enumerate(yearly_distribution["year"]):
            top5_val = yearly_distribution.loc[i, "top5_pct"]
            rest_val = yearly_distribution.loc[i, "rest_pct"]

            ax.text(year, top5_val / 2, f"{top5_val:.1f}%", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
            ax.text(year, top5_val + rest_val / 2, f"{rest_val:.1f}%", ha="center", va="center",
                    color="black", fontsize=9)

        plt.tight_layout()
        return fig
    

    def create_top_feature_requesters_chart(self, top_requesters, feature_issues, title="Top 10 Contributors by Feature Requests Submitted"):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filtering only issues from the top requesters
        top_users = top_requesters.index.tolist()
        subset = feature_issues[feature_issues['creator'].isin(top_users)].copy()
        subset['state'] = subset['state'].astype(str).str.replace("State.", "", regex=False)
        
        # Counting open vs closed per contributor
        status_counts = subset.groupby(['creator', 'state']).size().unstack(fill_value=0)
        status_counts = status_counts.loc[top_requesters.index]

        colors = {"open": "#1f77b4", "closed": "#2ca02c"}
        status_counts.plot(kind='barh', stacked=True, color=colors, ax=ax)

        # Annotate total feature requests per contributor at the end of each bar
        for i, (user, row) in enumerate(status_counts.iterrows()):
            total = row.sum()
            ax.text(total + 0.5, i, str(total), va='center', fontsize=10, fontweight='bold')

        ax.invert_yaxis() # Done so that we keep highest requester on top

        ax.set_title(title)
        ax.set_xlabel("Number of Feature Requests")
        ax.set_ylabel("Contributor")
        ax.legend(title="State", loc="lower right")
        ax.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        return fig

    def create_docs_issues_chart(self, status_counts, avg_commenters,
                                 title="Docs Issues: Open vs Closed per Month with Avg Commenters"):
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plotting stacked bars: open issues on bottom, closed issues on top
        ax1.bar(status_counts.index, status_counts['open'], width=20,
                label='Open', color='skyblue')
        ax1.bar(status_counts.index, status_counts['closed'], width=20,
                bottom=status_counts['open'], label='Closed', color='lightgreen')
        ax1.set_ylabel("Number of Doc Issues")
        ax1.set_xlabel("Month")
        ax1.set_title(title)
        
        # Adding a second y-axis for the line plot
        ax2 = ax1.twinx()
        
        # Plotting the average number of unique commenters per doc issue per month
        ax2.plot(avg_commenters.index, avg_commenters.values,
                 color='red', marker='o', linewidth=2, label='Avg Unique Commenters')
        ax2.set_ylabel("Avg Unique Commenters per Doc Issue")
        
        # Showing legends for both bar and line plots
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        return fig

    def create_issues_created_per_user_chart(self, issues_per_user, all_counts, title="Top 40 Contributors by Issues Created"):

        # Calculates the total number of issues across all contributors
        total_issues = all_counts.sum()
        
        # Total issues created by the top 40 contributors
        top40 = issues_per_user.head(40)
        top40_total = issues_per_user.sum()
        
        # Percentage of all issues that the top 40 users account for
        pct40 = (top40_total / total_issues) * 100

        fig, ax = plt.subplots(figsize=(14, 7))

        # Horizontal orientation for readability
        top40.sort_values().plot(kind='barh', ax=ax, color='teal')

        ax.set_xlabel("Number of Issues Created")
        ax.set_ylabel("Contributor's username")

        fig.suptitle(title)

        # Calculates what percentage of the overall issues that these top 40 contributors have created
        fig.text(0.5, 0.12,
         f"Top 40 contributors created {top40_total}/{total_issues} issues "
         f"({pct40:.2f}%)",
         ha='center', fontsize=12, color="gray")

        # Annotate each bar with the count
        for i, v in enumerate(top40.sort_values().values):
            ax.text(v + 0.5, i, str(v), va='center')

        plt.tight_layout(rect=[0, 0, 1, 0.9])  # leave space for title + subtitle
        return fig
        
    def create_top_active_users_per_year_chart(self, yearly_data, top_n=10):
        
        # Sort years so that figures and dropdown menu are in chronological order
        years = sorted(yearly_data.keys())
        fig = go.Figure()

        if not years:
            # Edge case where there is no contributor activity at all
            fig.update_layout(
                title="No contributor activity data available",
                xaxis_title="Activity",
                yaxis_title="User",
                height=400
            )
            return fig
        
        # Adding a bar trace for each year, only the first year is visible initially
        for i, year in enumerate(years):
            df_year = yearly_data[year].head(top_n)
            fig.add_trace(go.Bar(
                x=df_year["activity"],
                y=df_year["user"],
                orientation="h",
                name=str(year),
                visible=(i == 0),
                marker_color="teal"
            ))

        # Dropdown menu so that we can switch between years
        buttons = []
        for i, year in enumerate(years):
            visible = [False] * len(years)
            visible[i] = True
            buttons.append(dict(
                label=str(year),
                method="update",
                args=[{"visible": visible},
                    #   {"title": f"Top {top_n} Active Users in {year}"}]
                    {"title": {"text": f"Top {top_n} Active Users in {year}"}}]
            ))

        fig.update_layout(
            updatemenus=[dict(active=0, buttons=buttons, x=1.15, y=1.05)],
            title={"text": f"Top {top_n} Active Users in {years[0]}"},
            xaxis_title="Activity (Issues Created + Closed + Comments)",
            yaxis_title="User",
            height=600
        )
        return fig
    
    def create_engagement_heatmap_chart(self, heatmap_df: pd.DataFrame):
        # Create a heatmap of contributor engagement by day of week and hour of the day
        
        heatmap_norm = heatmap_df.div(heatmap_df.sum(axis=1), axis=0).fillna(0) * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            heatmap_norm,
            cmap="YlGnBu",
            linewidths=0.5,
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "% of Day's Activity"},
            ax=ax
        )
        ax.set_title("Contributor Engagement Heatmap")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")
        return fig

    def create_lifecycle_chart(self, summary, latest_date=None):
        # Plotting a Bar chart of lifecycle stage distribution with annotations
        colors = {
            "Newcomer": "skyblue",
            "Active": "orange",
            "Core Maintainer": "green",
            "Graduated Contributor": "gray"   
        }

        # Stage definitions for the legend
        stage_definitions = {
            "Newcomer": "First activity within last 30 days",
            "Active": "Contributed in ≥3 of last 6 months",
            "Core Maintainer": "Consistently active >12 months",
            "Graduated Contributor": "No activity in last 6+ months"
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(summary.index, summary["count"],
                      color=[colors.get(stage, "lightgray") for stage in summary.index])

        # Annotating counts for each contributor type on top of each bar
        for bar, stage in zip(bars, summary.index):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + (0.01 * summary["count"].max()),
                    f"{int(height)}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_ylabel("Contributors")

        title = "Contributor Lifecycle Stages"
        if latest_date is not None:
            title += f" (as of {latest_date.strftime('%b %Y')})"
        ax.set_title(title)
       
        legend_handles = [plt.Rectangle((0,0),1,1, color=colors[stage]) for stage in summary.index]
        legend_labels = [f"{stage}: {stage_definitions.get(stage, '')}" for stage in summary.index]
        ax.legend(legend_handles, legend_labels, loc="upper right", frameon=True)

        return fig
    
    def save_figure(self, fig, filename):
        # A wrapper to save matplotlib / plotly figures
        
        # Matplotlib Figure
        if hasattr(fig, "savefig"):
            fig.savefig(filename, bbox_inches="tight")
        
        # Plotly Figure
        elif hasattr(fig, "write_image"):
            fig.write_image(filename)
    
    
        