"""
Configuration loader for the RAG Video Transcription application.
"""
import json
from pathlib import Path
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json file."""
    config_path = Path(__file__).parent.parent / "config" / "config.json"
    try:
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

# Global config instance
CONFIG = load_config()

# Convenience accessors
def get_model_config() -> Dict[str, str]:
    return CONFIG["models"]

def get_paths_config() -> Dict[str, str]:
    return CONFIG["paths"]

def get_retrieval_config() -> Dict[str, Any]:
    return CONFIG["retrieval"]

def get_display_config() -> Dict[str, int]:
    return CONFIG["display_sources"]

import os
from pathlib import Path
from typing import Dict, Optional
import json

class Config:
    def __init__(self):
        self.base_dir = Path(os.path.dirname(os.path.dirname(__file__)))
        self.data_dir = self.base_dir / "data"
        self.config_dir = self.base_dir / "config"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        (self.data_dir / "videos").mkdir(exist_ok=True)
        (self.data_dir / "transcripts").mkdir(exist_ok=True)
        
        # Paths
        self.videos_dir = self.data_dir / "videos"
        self.transcripts_dir = self.data_dir / "transcripts"
        self.metadata_file = self.config_dir / "index_metadata.json"
        
        # Load metadata
        self.metadata = self.load_metadata()
        
        # Elasticsearch settings
        self.es_config = {
            "url": "http://localhost:9200",
            "index_name": "video-transcriptions",
            "username": "elastic",
            "password": "changeme"
        }
    
    def load_metadata(self) -> Dict:
        """Load metadata from index_metadata.json"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"transcript_metadata": []}
    
    def save_metadata(self, metadata: Dict) -> None:
        """Save metadata to index_metadata.json"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_video_metadata(self, video_path: str) -> Optional[Dict]:
        """Get metadata for a specific video"""
        for item in self.metadata.get("transcript_metadata", []):
            if item["video_path"] == video_path:
                return item
        return None
    
    def update_video_metadata(self, video_path: str, metadata: Dict) -> None:
        """Update metadata for a specific video"""
        for item in self.metadata.get("transcript_metadata", []):
            if item["video_path"] == video_path:
                item.update(metadata)
                break
        else:
            if "transcript_metadata" not in self.metadata:
                self.metadata["transcript_metadata"] = []
            self.metadata["transcript_metadata"].append(metadata)
        
        self.save_metadata(self.metadata)

# Global config instance
config = Config()
