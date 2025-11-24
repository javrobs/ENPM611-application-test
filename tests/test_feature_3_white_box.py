import unittest
from unittest.mock import patch
from app.feature_runner import FeatureRunner
import config
from argparse import Namespace
from model import Issue
import pandas as pd
from data_loader import DataLoader

MOCK_JSON = json_issue_sample = [{
        "url": "https://github.com/python-poetry/poetry/issues/9785",
        "creator": "dbrtly",
        "labels": [
        "kind/bug",
        "status/triage"
        ],
        "state": "closed",
        "assignees": [],
        "title": "python version resolution is messed up",
        "text":"Sample text",
        "number": 9785,
        "created_date": "2024-10-20T00:33:06+00:00",
        "updated_date": "2024-10-20T08:00:46+00:00",
        "timeline_url": "https://api.github.com/repos/python-poetry/poetry/issues/9785/timeline",
        "events": [
        {
            "event_type": "labeled",
            "author": "dbrtly",
            "event_date": "2024-10-20T00:33:06+00:00",
            "label": "kind/bug"
        },
        {
            "event_type": "labeled",
            "author": "dbrtly",
            "event_date": "2024-10-20T00:33:06+00:00",
            "label": "status/triage"
        },
        {
            "event_type": "commented",
            "author": "finswimmer",
            "event_date": "2024-10-20T08:00:46+00:00",
            "comment": "Comment"},
        {
            "event_type": "closed",
            "author": "finswimmer",
            "event_date": "2024-10-20T08:00:46+00:00"
        },]
    },
    {
        "url": "https://github.com/python-poetry/poetry/issues/9782",
        "creator": "srittau",
        "labels": [
        "area/docs",
        "status/triage"
        ],
        "state": "open",
        "assignees": [],
        "title": "Document PEP 440 (Compatible Release) specifiers",
        "text": "Sample text 2",
        "number": 9782,
        "created_date": "2024-10-19T15:01:45+00:00",
        "updated_date": "2024-10-19T15:42:09+00:00",
        "timeline_url": "https://api.github.com/repos/python-poetry/poetry/issues/9782/timeline",
        "events": [
        {
            "event_type": "labeled",
            "author": "srittau",
            "event_date": "2024-10-19T15:01:45+00:00",
            "label": "area/docs"
        },
        {
            "event_type": "labeled",
            "author": "srittau",
            "event_date": "2024-10-19T15:01:45+00:00",
            "label": "status/triage"
        },
        {
            "event_type": "commented",
            "author": "radoering",
            "event_date": "2024-10-19T15:42:08+00:00",
            "comment": "> Would you accept a PR add a section about them?\r\n\r\nI do not see why we should not. \ud83d\ude03"
        },
        {
            "event_type": "referenced",
            "author": "srittau",
            "event_date": "2024-10-19T15:54:01+00:00"
        },
        {
            "event_type": "cross-referenced",
            "author": "srittau",
            "event_date": "2024-10-19T15:55:18+00:00"
        }
        ]
    },
    {
        "url": "https://github.com/python-poetry/poetry/issues/9781",
        "creator": "ethan-neidhart37",
        "labels": [
        "kind/bug",
        "status/triage"
        ],
        "state": "closed",
        "assignees": [],
        "title": "Cannot modify pyproject.toml: list index out of range",
        "text": "Sample text 2",
        "number": 9781,
        "created_date": "2024-10-18T15:29:38+00:00",
        "updated_date": "2024-10-18T15:55:36+00:00",
        "timeline_url": "https://api.github.com/repos/python-poetry/poetry/issues/9781/timeline",
        "events": [
        {
            "event_type": "labeled",
            "author": "ethan-neidhart37",
            "event_date": "2024-10-18T15:29:38+00:00",
            "label": "kind/bug"
        },
        {
            "event_type": "labeled",
            "author": "ethan-neidhart37",
            "event_date": "2024-10-18T15:29:38+00:00",
            "label": "status/triage"
        },
        {
            "event_type": "commented",
            "author": "dimbleby",
            "event_date": "2024-10-18T15:32:38+00:00",
            "comment": "Please search for duplicates, please close"
        },
        {
            "event_type": "commented",
            "author": "ethan-neidhart37",
            "event_date": "2024-10-18T15:35:14+00:00",
            "comment": "> Please search for duplicates, please close\r\n\r\nThe closest duplicate I found was #9505 but that seems to be slightly different.\r\nThis issue is happening regardless of which package I install, and is not happening on other contributor's computers running the same poetry version.\r\n\r\nIf there is another issue which does match mine, I apologize but I was unable to find one"
        },
        {
            "event_type": "commented",
            "author": "dimbleby",
            "event_date": "2024-10-18T15:36:04+00:00",
            "comment": "This is an exact duplicate of #9505"
        },
        {
            "event_type": "commented",
            "author": "dimbleby",
            "event_date": "2024-10-18T15:44:57+00:00",
            "comment": "isodate 0.7.0 is your issue. Please read your own logs, please close"
        },
        {
            "event_type": "closed",
            "author": "ethan-neidhart37",
            "event_date": "2024-10-18T15:54:55+00:00"
        },
        {
            "event_type": "commented",
            "author": "ethan-neidhart37",
            "event_date": "2024-10-18T15:55:34+00:00",
            "comment": "Thank you, sorry about that"
        }
        ]}]
        

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.patcher = patch("data_loader.DataLoader.load_json", return_value=MOCK_JSON)
        # self.mock_load = self.patcher.start()
        self.patcher.start()
        self.loader = DataLoader()

    def test_dataloader_get_issues(self):
        loader = self.loader
        self.assertEqual(len(loader.get_issues()),3,"get_issues didn't return the right amount of objects")
        self.assertIsInstance(loader.get_issues()[0],Issue,"get_issues didn't return an issue object")
    
    def test_dataloader_parse_issues(self):
        loader = self.loader
        parse_issues = loader.parse_issues()
        # self.assertEqual(len(loader.parse_issues()),3,"parse_issues didn't return the right amount of objects")
        self.assertIsInstance(parse_issues,pd.DataFrame,"parse_issues didn't return a DataFrame object")
        self.assertEqual(parse_issues['number'].count(),3,"parse_issues didn't return the right amount of row")

class TestRunner(unittest.TestCase):

    def setUp(self):
        self.featureRunner = FeatureRunner()
        self.patcher = patch("data_loader.DataLoader.load_json", return_value=MOCK_JSON)
        self.patcher.start()

    def test_runner_feature_one(self):
        self.featureRunner.initialize_components()
        predictions = self.featureRunner.run_feature(3)
        print(predictions)
        self.assertIsInstance(predictions,list)
    

class TestAnalysisThree(unittest.TestCase):
    """
    The unittest module will be able to run this class as a test
    suite containing multiple test cases. Each function is considered
    a separate test case. When unittest runs these tests, it will not
    run them in the order in which they appear in this class.
    
    Run this function with the following command:
    python -m unittest test_black_box.py
    """
    
    
    def setUp(self):
        """
        This will be executed before any of the tests are run.
        # """
        # self.runner = FeatureRunner()
        # self.runner.initialize_components()

        # self.issues = [Issue(i) for i in json_issue_sample]
        
        

# if __name__ == "__main__":
#     unittest.main()