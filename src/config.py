"""
Configuration loader for recommendation experiments.
"""

import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration class for managing experiment settings."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def __getattr__(self, name: str) -> Any:
        """Allow dot notation access to config values."""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def get(self, key: str, default=None) -> Any:
        """Get config value with default."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration values."""
        self.config.update(updates)
    
    def __repr__(self) -> str:
        return f"Config({self.config})"

