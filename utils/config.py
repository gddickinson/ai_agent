"""
Configuration Utilities
Handles loading and validating configuration.
"""

import logging
import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If config file is invalid
    """
    logger = logging.getLogger(__name__)
    
    # Check if file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate configuration (basic checks)
        _validate_config(config)
        
        return config
    
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration: {e}")
        raise
    
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def _validate_config(config: Dict[str, Any]):
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    logger = logging.getLogger(__name__)
    
    # Check for required top-level sections
    required_sections = ['hardware', 'llm', 'memory', 'perception', 'cognition']
    for section in required_sections:
        if section not in config:
            logger.warning(f"Missing configuration section: {section}")
    
    # Check hardware platform
    if 'hardware' in config:
        platform = config['hardware'].get('platform')
        if not platform:
            logger.warning("No hardware platform specified, using 'mac' as default")
            config['hardware']['platform'] = 'mac'
    
    # Check LLM configuration
    if 'llm' in config and 'models' in config['llm']:
        for model_name, model_config in config['llm']['models'].items():
            if 'type' not in model_config:
                logger.warning(f"Model {model_name} missing type, using 'ollama' as default")
                model_config['type'] = 'ollama'
                
            if model_config['type'] == 'ollama' and 'model_id' not in model_config:
                logger.warning(f"Ollama model {model_name} missing model_id, using 'llama3' as default")
                model_config['model_id'] = 'llama3'
    
    logger.info("Configuration validation complete")

def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Raises:
        yaml.YAMLError: If config cannot be serialized
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {config_path}")
    
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge configurations, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    def _merge_dict(d1, d2):
        """Recursively merge dictionaries."""
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                _merge_dict(d1[k], v)
            else:
                d1[k] = v
    
    _merge_dict(result, override_config)
    return result
