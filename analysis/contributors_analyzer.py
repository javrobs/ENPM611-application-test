import pandas as pd
from model import Contributor, Issue, Event
from typing import List
import numpy as np
from utils.datetime_helper import extract_day_hour
from datetime import timezone

class ContributorsAnalyzer:
    def build_contributors(self, issues_df: pd.DataFrame, events_df: pd.DataFrame) -> List[Contributor]:
        IGNORED_USERS = {"stale[bot]", "github-actions[bot]"}

        # Ensure DataFrame date columns are parsed
        issues_df["created_date"] = pd.to_datetime(issues_df["created_date"], errors="coerce")
        if "closure_date" in issues_df.columns:
            issues_df["closure_date"] = pd.to_datetime(issues_df["closure_date"], errors="coerce")
        events_df["event_date"] = pd.to_datetime(events_df["event_date"], errors="coerce")

        contributors_map: dict[str, Contributor] = {}

        # Issues created
        for _, row in issues_df.iterrows():
            username = row["creator"]
            if username in IGNORED_USERS:
                continue
            if username not in contributors_map:
                contributors_map[username] = Contributor(username=username)

            issue_data = {
                "number": str(row["number"]),
                "creator": row["creator"],
                "state": row["state"],
                "created_date": row["created_date"].isoformat() if pd.notna(row["created_date"]) else None,
                "updated_date": row["updated_date"].isoformat() if pd.notna(row["updated_date"]) else None,
                "labels": row.get("labels", []),
                "events": []  # events attached separately
            }
            issue = Issue(issue_data)
            contributors_map[username].add_issue(issue)

        # Events (comments, labels, closures, etc.)
        for _, row in events_df.iterrows():
            username = row["event_author"]
            if username in IGNORED_USERS:
                continue
            if username not in contributors_map:
                contributors_map[username] = Contributor(username=username)

            event_data = {
                "event_type": row["event_type"],
                "author": row["event_author"],
                "event_date": row["event_date"].isoformat() if pd.notna(row["event_date"]) else None,
                "label": row.get("label"),
                "comment": row.get("comment")
            }
            event = Event(event_data)
            contributors_map[username].add_comment(event)

        return list(contributors_map.values())


    def analyze_bug_closure_distribution(self, issues_df, events_df) -> pd.DataFrame:
        # Filtering only issues labeled as bugs
        bug_issues = issues_df[
            issues_df['labels'].apply(lambda L: any('bug' in l.lower() for l in L))
        ].copy()

        # Find closure events for those bug issues
        bug_closures = events_df[
            (events_df['issue_number'].isin(bug_issues['number'])) &
            (events_df['event_type'] == 'closed')
        ].copy()
        bug_closures['year'] = bug_closures['event_date'].dt.year
        
        # Counting how many bugs each contributor closed per year
        closer_counts = (
            bug_closures.groupby(['year', 'event_author'])
            .size()
            .reset_index(name='n_closed')
        )

        # A helper function to split yearly totals into top 5 vs the rest
        def top5_vs_rest(df):
            df = df.sort_values('n_closed', ascending=False)
            top5 = df.head(5)
            rest = df['n_closed'].sum() - top5['n_closed'].sum()
            total = df['n_closed'].sum()
            return pd.Series({
                'top5_pct': (top5['n_closed'].sum() / total) * 100 if total > 0 else 0,
                'rest_pct': (rest / total) * 100 if total > 0 else 0,
                'top5_users': ", ".join(top5['event_author'].tolist())
            })

        # Apply the split per year
        yearly_distribution = (
            closer_counts.groupby('year')
            .apply(top5_vs_rest)
            .reset_index()
        )

        return yearly_distribution
    
    
    def analyze_top_feature_requesters(self, issues_df, top_n=10):
        # Filtering only issues labeled as features
        feature_issues = issues_df[
            issues_df['labels'].apply(lambda L: any('feature' in str(l).lower() for l in L))
        ]
        if feature_issues.empty:
            return None, None
        
        # Ranking creators by number of feature requests
        top_requesters = feature_issues['creator'].value_counts().head(top_n)
        return top_requesters, feature_issues
    
    def compute_unique_commenters(self, events_df: pd.DataFrame, issues_df: pd.DataFrame) -> pd.DataFrame:
        # Get unique commenters per issue per month
        issue_numbers = issues_df['number'].unique()
        comment_events = events_df[
            (events_df['issue_number'].isin(issue_numbers)) &
            (events_df['event_type'] == 'commented')
        ].copy()

        if comment_events.empty:
            return pd.DataFrame(columns=['issue_number', 'month', 'n_unique_commenters'])

        # Add month column
        comment_events['month'] = comment_events['event_date'].dt.to_period('M')

        # Counting distinct authors per (issue, month)
        metrics = (
            comment_events.groupby(['issue_number', 'month'])['event_author']
            .nunique()
            .reset_index(name='n_unique_commenters')
        )
        return metrics
    
    def analyze_docs_issues(self, issues_df, events_df, loader):
        
        # Filter documentation-related issues
        docs_issues = loader.filter_by_label(issues_df, "doc")
        if docs_issues.empty:
            return None, None
        
        # Monthly open/closed counts
        docs_issues['month'] = docs_issues['created_date'].dt.to_period('M')

        status_counts = docs_issues.groupby(['month', 'state']).size().unstack(fill_value=0)
        status_counts.index = status_counts.index.to_timestamp()

        # Average unique commenters per doc issue per month
        docs_metrics = self.compute_unique_commenters(events_df, docs_issues)
        avg_commenters = docs_metrics.groupby('month')['n_unique_commenters'].mean()
        avg_commenters.index = avg_commenters.index.to_timestamp()

        return status_counts, avg_commenters
    
    def analyze_issues_created_per_user(self, issues_df, top_n=40):
        if issues_df.empty:
            return None
        # Counts issues created per user for top_n users and overall counts
        issues_per_user = issues_df['creator'].value_counts().head(top_n)
        all_counts = issues_df['creator'].value_counts()
        return (issues_per_user, all_counts)
    
    def analyze_top_active_users_per_year(self, contributors: List["Contributor"]):
        result: dict[int, pd.DataFrame] = {}

        # Gather all distinct years in which any contributor was active
        all_years = set()
        for c in contributors:
            all_years.update(c.get_active_years())

        # Computing activity counts for each year
        for year in sorted(all_years):
            rows = []
            for c in contributors:
                activity = c.get_activity_count_by_year(year)
                if activity > 0:
                    rows.append((c.username, activity))
                    
            # Only build a DataFrame if there is activity in that year
            if rows:
                df = pd.DataFrame(rows, columns=["user", "activity"])
                df = df.sort_values("activity", ascending=False)
                result[year] = df
        
        # Returns a mapping: year -> DataFrame of top users
        return result
    
    def analyze_engagement_heatmap(self, contributors: List["Contributor"]) -> pd.DataFrame:
        # Initializing a 7(days) x 24(hours) matrix with zeros.
        heatmap = np.zeros((7, 24), dtype=int)
        
        # Populating the heatmap with activity counts
        for c in contributors:
            # Issues created
            for issue in c.issues_created:
                coords = extract_day_hour(issue.created_date)
                if coords:
                    heatmap[coords] += 1
            # Comments made
            for event in c.comments:
                coords = extract_day_hour(event.event_date)
                if coords:
                    heatmap[coords] += 1
            # Issues closed
            for issue in getattr(c, "issues_closed", []):
                coords = extract_day_hour(issue.get_closure_date())
                if coords:
                    heatmap[coords] += 1

        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        hours = list(range(24))
        return pd.DataFrame(heatmap, index=days, columns=hours)

    def analyze_lifecycle_stages(self, contributors, reference_date=None):
        # Build a list of contributor records, ensuring each username is only included once
        data = []
        seen = set()
        for c in contributors:
            if c.username in seen:
                continue
            seen.add(c.username)
            data.append({
                "contributor": c.username,
                "first_activity": c.first_activity,
                "last_activity": c.last_activity
            })
            
        # Creating a DataFrame and collapse duplicates by username
        df = pd.DataFrame(data)
        df = df.groupby("contributor", as_index=False).agg({
            "first_activity": "min",
            "last_activity": "max"
        })


        # If no reference_date provided, reference date would be the latest activity in the dataset
        if reference_date is None:
            all_dates = pd.concat([df["first_activity"].dropna(), df["last_activity"].dropna()])
            reference_date = all_dates.max()  # dataset's "present"

        # Ensuring reference_date is timezone-aware (UTC)
        if reference_date.tzinfo is None:
            reference_date = reference_date.replace(tzinfo=timezone.utc)

        def assign_stage(row):
            first, last = row["first_activity"], row["last_activity"]
            # Normalize contributor timestamps to UTC
            if first is not None and first.tzinfo is None:
                first = first.replace(tzinfo=timezone.utc)
            if last is not None and last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            
            # Stage assignment rules:
            # - First activity within last 30 days → Newcomer
            # - No recorded last activity or no activity in the last 6 months → Graduated Contributor
            # - Active for more than a year (and still contributing) → Core Maintainer
            # - Otherwise → Active
            if last is None:
                return "Graduated Contributor"
            elif (reference_date - first).days <= 30:
                return "Newcomer"
            elif (reference_date - last).days > 180:
                return "Graduated Contributor"
            elif (reference_date - first).days > 365:
                return "Core Maintainer"
            else:
                return "Active"

        df["stage"] = df.apply(assign_stage, axis=1)
        return df