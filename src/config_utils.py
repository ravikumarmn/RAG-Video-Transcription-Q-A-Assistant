import json
from pathlib import Path
from typing import Dict, Any

class Config:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        config_path = Path(__file__).parent.parent / "config" / "config.json"
        try:
            with open(config_path, 'r') as f:
                self._config = json.load(f)
        except Exception as e:
            raise Exception(f"Error loading config file: {str(e)}")

    @property
    def models(self) -> Dict[str, str]:
        return self._config["models"]

    @property
    def paths(self) -> Dict[str, str]:
        return self._config["paths"]

    @property
    def retrieval(self) -> Dict[str, Any]:
        return self._config["retrieval"]

    @property
    def display_sources(self) -> Dict[str, int]:
        return self._config["display_sources"]

# Create a singleton instance
config = Config()
