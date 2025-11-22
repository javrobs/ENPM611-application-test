"""
Priority Analysis Module
Handles feature extraction and urgency categorization for issues.
"""

from datetime import datetime
from collections import Counter
import numpy as np


class PriorityAnalyzer:
    """
    Analyzes issues to extract features and assign urgency categories.
    """
    
    def __init__(self, issues):
        """
        Initialize the analyzer with issues data.
        
        Args:
            issues (list): List of issue dictionaries
        """
        self.issues = issues
        self.closed_issues = [issue for issue in issues if issue['state'] == 'closed']
        self.open_issues = [issue for issue in issues if issue['state'] == 'open']
    
    def get_resolution_time(self, issue):
        """
        Calculate resolution time in hours for a closed issue.
        
        Args:
            issue (dict): Issue dictionary
            
        Returns:
            float: Resolution time in hours, or None if unavailable
        """
        try:
            created = datetime.fromisoformat(issue['created_date'].replace('Z', '+00:00'))
        except:
            return None
            
        closed_date = None
        
        # Try to find close event
        for event in issue.get('events', []):
            if event.get('event_type') == 'closed':
                date_str = event.get('event_date') or event.get('created_date')
                if date_str:
                    try:
                        closed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        break
                    except:
                        continue
        
        # Fallback to updated_date
        if closed_date is None:
            try:
                closed_date = datetime.fromisoformat(issue['updated_date'].replace('Z', '+00:00'))
            except:
                return None
        
        hours = (closed_date - created).total_seconds() / 3600
        return max(0, hours) if hours > 0 else None
    
    def assign_urgency_category(self, issue, resolution_time):
        """
        Assign urgency category based on multiple factors:
        - Labels (bug, critical, security)
        - Community engagement (comments, participants)
        - Maintainer response time
        - Resolution time
        
        Args:
            issue (dict): Issue dictionary
            resolution_time (float): Resolution time in hours (optional)
            
        Returns:
            str: Urgency category ('Critical', 'High', 'Medium', 'Low')
        """
        score = 0
        
        labels = [l.lower() for l in issue.get('labels', [])]
        
        # Label-based urgency (boosted for critical indicators)
        if any('critical' in l or 'blocker' in l or 'regression' in l for l in labels):
            score += 5  # Increased from 4
        if any('bug' in l or 'error' in l or 'crash' in l for l in labels):
            score += 4  # Increased from 3
        if any('security' in l or 'vulnerability' in l for l in labels):
            score += 5  # Increased from 4
            
        # Community engagement
        num_comments = len([e for e in issue.get('events', []) if e.get('event_type') == 'commented'])
        num_participants = len(set([e.get('actor') for e in issue.get('events', []) if e.get('actor')]))
        
        if num_comments > 20:
            score += 3
        elif num_comments > 10:
            score += 2
        elif num_comments > 5:
            score += 1
            
        if num_participants > 5:
            score += 2
        elif num_participants > 3:
            score += 1
        
        # Maintainer response time
        first_response_hours = self._get_first_response_time(issue)
        if first_response_hours and first_response_hours < 24:
            score += 2
        elif first_response_hours and first_response_hours < 72:
            score += 1
            
        # Resolution time (as one of many factors)
        if resolution_time:
            days = resolution_time / 24
            if 1 < days < 7:  # Fixed within a week
                score += 2
            elif days < 1:  # Emergency fix
                score += 3
        
        # Map to urgency levels (adjusted thresholds)
        if score >= 10:
            return 'Critical'
        elif score >= 7:
            return 'High'
        elif score >= 4:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_first_response_time(self, issue):
        """
        Calculate time until first response from a non-creator.
        
        Args:
            issue (dict): Issue dictionary
            
        Returns:
            float: Time to first response in hours, or None
        """
        try:
            created = datetime.fromisoformat(issue['created_date'].replace('Z', '+00:00'))
        except:
            return None
            
        creator = issue.get('creator')
        
        # Filter events that have dates and sort them
        events_with_dates = [
            e for e in issue.get('events', []) 
            if e.get('event_date') or e.get('created_date')
        ]
        
        # Sort by date
        try:
            sorted_events = sorted(
                events_with_dates, 
                key=lambda x: x.get('event_date') or x.get('created_date') or ''
            )
        except:
            sorted_events = events_with_dates
        
        for event in sorted_events:
            if event.get('event_type') in ['commented', 'labeled']:
                actor = event.get('actor')
                if actor and actor != creator:
                    date_str = event.get('event_date') or event.get('created_date')
                    if date_str:
                        try:
                            response_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            return (response_time - created).total_seconds() / 3600
                        except:
                            continue
        return None
    
    def extract_features(self, issue):
        """
        Extract comprehensive features from an issue.
        
        Args:
            issue (dict): Issue dictionary
            
        Returns:
            dict: Feature dictionary
        """
        features = {}
        
        # Text features
        title = issue.get('title') or ''
        body = issue.get('text') or ''
        features['text'] = f"{title} {body}"
        
        # Numeric features
        features['title_len'] = len(title)
        features['body_len'] = len(body)
        features['num_code_blocks'] = body.count('```') if body else 0
        features['has_stack_trace'] = int(('traceback' in body.lower() or 'error:' in body.lower()) if body else 0)
        
        # Activity features
        events = issue.get('events', [])
        features['num_comments'] = len([e for e in events if e.get('event_type') == 'commented'])
        features['num_events'] = len(events)
        features['num_participants'] = len(set([e.get('actor') for e in events if e.get('actor')]))
        features['num_labels'] = len(issue.get('labels', []))
        
        # Label features
        labels = [l.lower() for l in issue.get('labels', [])]
        features['has_bug'] = int(any('bug' in l for l in labels))
        features['has_feature'] = int(any('feature' in l or 'enhancement' in l for l in labels))
        features['has_docs'] = int(any('doc' in l for l in labels))
        features['has_critical'] = int(any('critical' in l or 'blocker' in l for l in labels))
        features['has_triage'] = int(any('triage' in l for l in labels))
        
        # Time-based features
        features['first_response_hours'] = self._get_first_response_time(issue) or 0
        
        return features
    
    def get_urgency_statistics(self):
        """
        Calculate urgency distribution statistics for closed issues.
        
        Returns:
            dict: Statistics about urgency categories
        """
        urgency_counts = Counter()
        
        for issue in self.closed_issues:
            resolution_time = self.get_resolution_time(issue)
            if resolution_time:
                urgency = self.assign_urgency_category(issue, resolution_time)
                urgency_counts[urgency] += 1
        
        total = sum(urgency_counts.values())
        return {
            'counts': dict(urgency_counts),
            'percentages': {k: round(v/total*100, 1) for k, v in urgency_counts.items()} if total > 0 else {},
            'total': total
        }
    
    def calculate_complexity_score(self, issue):
        """
        Calculate complexity score based on TECHNICAL factors only.
        This is independent of priority/urgency.
        
        Factors:
        - Code depth (length, structure)
        - Technical indicators (stack traces, code blocks)
        - Technical labels (not priority labels)
        - Number of affected components
        
        Returns a score from 0-100 (higher = more technically complex)
        """
        score = 0
        
        body = issue.get('text') or ''
        title = issue.get('title') or ''
        labels = [l.lower() for l in issue.get('labels', [])]
        
        # 1. Text complexity and depth (0-30 points)
        body_len = len(body)
        if body_len > 3000:
            score += 30
        elif body_len > 2000:
            score += 25
        elif body_len > 1000:
            score += 20
        elif body_len > 500:
            score += 15
        elif body_len > 200:
            score += 10
        else:
            score += 5
        
        # 2. Technical indicators (0-30 points)
        code_blocks = body.count('```')
        if code_blocks >= 4:
            score += 20
        elif code_blocks >= 2:
            score += 15
        elif code_blocks >= 1:
            score += 10
        
        # Stack traces indicate technical depth
        has_stack_trace = 'traceback' in body.lower() or 'error:' in body.lower() or 'exception' in body.lower()
        if has_stack_trace:
            score += 10
        
        # 3. Technical scope labels (0-25 points)
        # These indicate technical complexity, not priority
        technical_scope = [
            'architecture', 'refactor', 'performance', 'compatibility',
            'integration', 'dependency', 'api', 'breaking-change',
            'typescript', 'build', 'ci', 'testing'
        ]
        scope_matches = sum(5 for label in labels if any(tech in label for tech in technical_scope))
        score += min(scope_matches, 25)
        
        # 4. Multiple component involvement (0-15 points)
        # Issues mentioning multiple technical components are more complex
        components = ['plugin', 'installer', 'resolver', 'venv', 'lock', 'cache', 'config']
        component_count = sum(1 for comp in components if comp in body.lower() or comp in title.lower())
        if component_count >= 4:
            score += 15
        elif component_count >= 3:
            score += 10
        elif component_count >= 2:
            score += 5
        
        return min(score, 100)  # Cap at 100
    
    def get_resolution_statistics(self):
        """
        Calculate resolution time statistics.
        
        Returns:
            dict: Statistics about resolution times
        """
        resolution_times = []
        
        for issue in self.closed_issues:
            res_time = self.get_resolution_time(issue)
            if res_time:
                resolution_times.append(res_time)
        
        if not resolution_times:
            return {}
        
        return {
            'median_days': round(np.median(resolution_times) / 24, 1),
            'mean_days': round(np.mean(resolution_times) / 24, 1),
            'p75_days': round(np.percentile(resolution_times, 75) / 24, 1),
            'p95_days': round(np.percentile(resolution_times, 95) / 24, 1),
            'count': len(resolution_times)
        }
    
    def get_urgency_statistics(self):
        """
        Calculate urgency distribution statistics for closed issues.
        
        Returns:
            dict: Statistics about urgency categories
        """
        from collections import Counter
        urgency_counts = Counter()
        
        for issue in self.closed_issues:
            resolution_time = self.get_resolution_time(issue)
            if resolution_time:
                urgency = self.assign_urgency_category(issue, resolution_time)
                urgency_counts[urgency] += 1
        
        total = sum(urgency_counts.values())
        return {
            'counts': dict(urgency_counts),
            'percentages': {k: round(v/total*100, 1) for k, v in urgency_counts.items()} if total > 0 else {},
            'total': total
        }
        """
        Calculate resolution time statistics.
        
        Returns:
            dict: Statistics about resolution times
        """
        resolution_times = []
        
        for issue in self.closed_issues:
            res_time = self.get_resolution_time(issue)
            if res_time:
                resolution_times.append(res_time)
        
        if not resolution_times:
            return {}
        
        return {
            'median_days': round(np.median(resolution_times) / 24, 1),
            'mean_days': round(np.mean(resolution_times) / 24, 1),
            'p75_days': round(np.percentile(resolution_times, 75) / 24, 1),
            'p95_days': round(np.percentile(resolution_times, 95) / 24, 1),
            'count': len(resolution_times)
        }