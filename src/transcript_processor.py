import os
import re
from typing import List, Dict, Optional
from datetime import datetime
import webvtt
from pathlib import Path
from config import config


class TranscriptSegment:
    def __init__(self, text: str, start: str, end: str):
        self.text = text
        self.start = start
        self.end = end

    def to_dict(self) -> Dict:
        return {"text": self.text, "start_time": self.start, "end_time": self.end}


class TranscriptProcessor:
    def __init__(self, videos_dir: str, transcripts_dir: str):
        self.videos_dir = Path(videos_dir)
        self.transcripts_dir = Path(transcripts_dir)

    def parse_vtt(self, vtt_path: str) -> List[TranscriptSegment]:
        """Parse a VTT file into segments."""
        segments = []
        try:
            for caption in webvtt.read(vtt_path):
                try:
                    # Clean the text: remove multiple spaces, newlines, and speaker tags
                    text = caption.text
                    # Remove speaker tags if present
                    text = re.sub(r'<v [^>]*>', '', text)
                    text = re.sub(r'</v>', '', text)
                    # Clean up whitespace
                    text = " ".join(text.split())
                    
                    if text.strip():  # Only add non-empty segments
                        segments.append(
                            TranscriptSegment(text=text, start=caption.start, end=caption.end)
                        )
                except Exception as e:
                    print(f"Warning: Failed to parse caption in {vtt_path}: {str(e)}")
                    continue
                    
            if not segments:
                raise ValueError(f"No valid segments found in transcript: {vtt_path}")
                
            return segments
        except Exception as e:
            raise ValueError(f"Failed to parse VTT file {vtt_path}: {str(e)}")

    def extract_metadata(self, video_path: str, transcript_path: str) -> Dict:
        """Extract metadata from video and transcript files."""
        video_path = Path(video_path)
        transcript_path = Path(transcript_path)

        # Get video information
        video_stats = video_path.stat()

        # Extract timestamp from transcript filename if it exists
        # Format: filename_YYYYMMDD_HHMMSS.vtt
        timestamp_match = re.search(r"_(\d{8}_\d{6})\.vtt$", transcript_path.name)
        processed_date = None
        if timestamp_match:
            date_str = timestamp_match.group(1)
            try:
                processed_date = datetime.strptime(
                    date_str, "%Y%m%d_%H%M%S"
                ).isoformat()
            except ValueError:
                pass

        # Start with file system metadata
        metadata = {
            "video_filename": video_path.name,
            "video_size_bytes": video_stats.st_size,
            "video_created_at": datetime.fromtimestamp(
                video_stats.st_ctime
            ).isoformat(),
            "video_modified_at": datetime.fromtimestamp(
                video_stats.st_mtime
            ).isoformat(),
            "transcript_filename": transcript_path.name,
            "transcript_processed_at": processed_date,
            "file_extension": video_path.suffix.lower(),
        }

        # Get custom metadata from index_metadata.json
        custom_metadata = config.get_video_metadata(video_path.name)
        if custom_metadata:
            # Merge custom metadata, preserving file system metadata if keys conflict
            custom_metadata = {k: v for k, v in custom_metadata.items() 
                             if k not in metadata}
            metadata.update(custom_metadata)

        return metadata

    def find_matching_transcript(self, video_filename: str) -> Optional[Path]:
        """Find the matching transcript file for a video."""
        try:
            base_name = Path(video_filename).stem.lower()
            
            # First try exact match with same name
            exact_match = self.transcripts_dir / f"{base_name}.vtt"
            if exact_match.exists():
                print(f"Found exact transcript match: {exact_match.name}")
                return exact_match
            
            # Then try case-insensitive match
            for transcript in self.transcripts_dir.glob("*.vtt"):
                if transcript.stem.lower() == base_name:
                    print(f"Found case-insensitive match: {transcript.name}")
                    return transcript
            
            # Try partial match (for timestamped files)
            matching_transcripts = []
            for transcript in self.transcripts_dir.glob("*.vtt"):
                transcript_stem = transcript.stem.lower()
                if transcript_stem.startswith(base_name) or base_name.startswith(transcript_stem):
                    matching_transcripts.append(transcript)
            
            if matching_transcripts:
                # Return the most recent transcript if multiple exist
                best_match = sorted(matching_transcripts, key=lambda x: x.stat().st_mtime)[-1]
                print(f"Found partial match: {best_match.name}")
                return best_match
            
            print(f"No transcript found for video: {video_filename}")
            print(f"Looked for:")
            print(f"1. Exact match: {base_name}.vtt")
            print(f"2. Case-insensitive match for: {base_name}")
            print(f"3. Partial matches starting with: {base_name}")
            return None
            
        except Exception as e:
            print(f"Warning: Error finding transcript for {video_filename}: {str(e)}")
            return None

    def process_video(self, video_filename: str) -> Optional[Dict]:
        """Process a single video and its transcript."""
        try:
            video_path = self.videos_dir / video_filename
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_filename}")

            transcript_path = self.find_matching_transcript(video_filename)
            if not transcript_path:
                raise FileNotFoundError(
                    f"No matching transcript found for video: {video_filename}"
                )

            # Parse the transcript
            segments = self.parse_vtt(str(transcript_path))
            if not segments:
                raise ValueError(f"No valid segments found in transcript: {transcript_path}")

            # Extract metadata
            metadata = self.extract_metadata(video_path, transcript_path)

            # Add some additional metadata
            metadata.update({
                "segment_count": len(segments),
                "transcript_path": str(transcript_path)
            })

            return {"segments": [seg.to_dict() for seg in segments], "metadata": metadata}
            
        except Exception as e:
            raise Exception(f"Failed to process video {video_filename}: {str(e)}")

    def process_all_videos(self) -> List[Dict]:
        """Process all videos in the videos directory."""
        results = []
        for video_file in self.videos_dir.glob("*.mp4"):
            try:
                result = self.process_video(video_file.name)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {video_file.name}: {str(e)}")
        return results
