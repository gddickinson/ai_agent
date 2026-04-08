"""Unit tests for configuration utilities."""

import os
import sys
import tempfile
import pytest
import yaml

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, merge_configs, _validate_config


class TestLoadConfig:
    """Tests for load_config()."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid YAML config file."""
        config_data = {
            "hardware": {"platform": "mac", "sensors": {}, "actuators": {}},
            "llm": {"models": {}},
            "memory": {"memory_dir": "memory"},
            "perception": {},
            "cognition": {},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        result = load_config(str(config_file))
        assert result["hardware"]["platform"] == "mac"
        assert "llm" in result

    def test_load_missing_file_raises(self):
        """Test that loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        """Test that invalid YAML raises an error."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{{invalid yaml: [}")
        with pytest.raises(yaml.YAMLError):
            load_config(str(bad_file))


class TestValidateConfig:
    """Tests for _validate_config()."""

    def test_missing_platform_gets_default(self):
        """Test that missing platform is set to 'mac'."""
        config = {"hardware": {}, "llm": {}, "memory": {}, "perception": {}, "cognition": {}}
        _validate_config(config)
        assert config["hardware"]["platform"] == "mac"

    def test_missing_sections_warns(self):
        """Test that missing required sections don't crash."""
        config = {}
        _validate_config(config)  # Should not raise


class TestMergeConfigs:
    """Tests for merge_configs()."""

    def test_override_top_level(self):
        """Test that override replaces top-level values."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)
        assert result["a"] == 1
        assert result["b"] == 3
        assert result["c"] == 4

    def test_deep_merge(self):
        """Test that nested dicts are merged recursively."""
        base = {"hardware": {"platform": "mac", "sensors": {"cam": True}}}
        override = {"hardware": {"sensors": {"mic": True}}}
        result = merge_configs(base, override)
        assert result["hardware"]["platform"] == "mac"
        assert result["hardware"]["sensors"]["cam"] is True
        assert result["hardware"]["sensors"]["mic"] is True
