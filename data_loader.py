import json
import pandas as pd
from typing import List
import config
from model import Issue

_ISSUES: List[Issue] = None

class DataLoader:
    # Loads the issue data into runtime objects and provides DataFrame views
    def __init__(self):
        self.file_path: str = config.get_parameter('ENPM611_PROJECT_DATA_PATH')

    def load_json(self):
        with open(self.file_path, 'r') as fin:
            return json.load(fin)

    def get_issues(self) -> List[Issue]:
        """Return list of Issue objects (cached)."""
        global _ISSUES
        if _ISSUES is None:
            _ISSUES = [Issue(i) for i in self.load_json()]
            print(f'Loaded {len(_ISSUES)} issues from {self.file_path}.')
        return _ISSUES

    def parse_issues(self) -> pd.DataFrame:
        # Returns a DataFrame view of issues
        issues = self.get_issues()
        return pd.DataFrame([{
            "number": i.number,
            "creator": i.creator,
            "labels": i.labels,
            "state": i.state,
            "created_date": i.created_date,
            "updated_date": i.updated_date,
            "events": i.events,
            "closure_date": i.get_closure_date(),
            "resolution_time": i.get_resolution_time(),
            "comment_count": i.get_comment_count()
        } for i in issues])

    def parse_events(self, issues_df: pd.DataFrame) -> pd.DataFrame:
        # Flatten events from issues into a DataFrame.
        events = []
        for _, row in issues_df.iterrows():
            for ev in row["events"]:
                events.append({
                    "issue_number": row["number"],
                    "event_type": ev.event_type,
                    "event_author": ev.author,
                    "event_date": ev.event_date,
                    "label": ev.label,
                    "comment": ev.comment
                })
        return pd.DataFrame(events)

    def validate_data(self) -> bool:
        # Basic validation: returns issues loaded and have numbers
        issues = self.get_issues()
        return all(i.number is not None for i in issues)

    def filter_by_state(self, state: str) -> List[Issue]:
        # Return issues filtered by state ('open' or 'closed')
        return [i for i in self.get_issues() if i.state.value == state]
    
    def filter_by_label(self, issues_df: pd.DataFrame, keyword: str) -> pd.DataFrame:
        # Returns a DataFrame of issues whose labels contain the keyword.
        return issues_df[
            issues_df['labels'].apply(lambda L: any(keyword in str(l).lower() for l in L))
        ].copy()
