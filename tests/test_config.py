import unittest
import os
import json
import tempfile
from argparse import Namespace
from config import ConfigManager, get_parameter, set_parameter, convert_to_typed_value, overwrite_from_args


class TestConfigModule(unittest.TestCase):
    def setUp(self):
        self.config_manager = ConfigManager()
        # Store original env vars to restore later
        self.original_env = os.environ.copy()

    def tearDown(self):
        # Restore original environment variables
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_config_load(self):
        """Test that ConfigManager loads successfully"""
        self.assertIsNotNone(self.config_manager)
        self.assertIsInstance(self.config_manager, ConfigManager)

    def test_get_data_path(self):
        """Test getting the data path from config"""
        data_path = self.config_manager.get_data_path()
        self.assertEqual(data_path, "data/poetry_issues_all.json")

    def test_get_parameter_default(self):
        """Test getting a parameter with default value"""
        value = get_parameter("NON_EXISTENT_PARAM", default="default_value")
        self.assertEqual(value, "default_value")

    def test_get_parameter_from_env(self):
        """Test that environment variables take precedence"""
        os.environ["ENPM611_PROJECT_DATA_PATH"] = "env_value.json"
        value = get_parameter("ENPM611_PROJECT_DATA_PATH")
        self.assertEqual(value, "env_value.json")

    def test_convert_to_typed_value_int(self):
        """Test converting integer values"""
        result = convert_to_typed_value("42")
        self.assertEqual(result, 42)

    def test_convert_to_typed_value_bool(self):
        """Test converting boolean values"""
        result = convert_to_typed_value("true")
        self.assertTrue(result)

    def test_config_manager_get_method(self):
        """Test ConfigManager.get() method"""
        value = self.config_manager.get("ENPM611_PROJECT_DATA_PATH")
        self.assertEqual(value, "data/poetry_issues_all.json")

    def test_config_manager_set_method(self):
        """Test ConfigManager.set() method"""
        self.config_manager.set("NEW_PARAM", "new_value")
        value = self.config_manager.get("NEW_PARAM")
        self.assertEqual(value, "new_value")

    def test_config_manager_overwrite_from_args(self):
        """Test ConfigManager.overwrite_from_args() method"""
        args = Namespace(
            ENPM611_PROJECT_DATA_PATH="new_data_path.json",
            custom_param="custom_value"
        )
        
        self.config_manager.overwrite_from_args(args)
        
        # Verify the parameters were set
        self.assertEqual(self.config_manager.get("ENPM611_PROJECT_DATA_PATH"), "new_data_path.json")
        self.assertEqual(self.config_manager.get("custom_param"), "custom_value")


    def test_config_manager_get_output_path(self):
        """Test getting the output path from config"""
        output_path = self.config_manager.get_output_path()
        self.assertEqual(output_path, "output/")


if __name__ == '__main__':
    unittest.main()
        