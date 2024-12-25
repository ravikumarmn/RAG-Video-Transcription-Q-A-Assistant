"""
Script to upsert videos and transcripts to the vector store.
"""
from pathlib import Path
from typing import List, Dict
from vector_store import VideoTranscriptionStore
import os
from dotenv import load_dotenv
import time
import json

load_dotenv()

def get_video_files(videos_dir: str) -> List[str]:
    """Get list of video files in the directory."""
    return [f.name for f in Path(videos_dir).glob("*.mp4")]

def get_transcript_files(transcripts_dir: str) -> Dict[str, str]:
    """Get mapping of video names to transcript files."""
    transcript_map = {}
    for f in Path(transcripts_dir).glob("*.vtt"):
        # Match exact video name without extension
        video_name = f"{f.stem}.mp4"
        transcript_map[video_name] = str(f)
        print(f"Mapped {video_name} -> {f.name}")
    return transcript_map

def get_video_metadata(videos_dir: str) -> Dict[str, Dict]:
    """Get metadata for videos including speaker information."""
    metadata_file = Path(videos_dir) / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            return json.load(f)
    return {}

def main():
    # Initialize paths
    base_dir = Path(os.path.dirname(os.path.dirname(__file__)))
    videos_dir = base_dir / "data" / "videos"
    transcripts_dir = base_dir / "data" / "transcripts"

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Upsert videos to vector store')
    parser.add_argument('--force', action='store_true', help='Force update existing videos')
    parser.add_argument('--metadata-only', action='store_true', help='Only update metadata for existing videos')
    args = parser.parse_args()

    # Initialize store
    try:
        store = VideoTranscriptionStore(
            videos_dir=str(videos_dir),
            transcripts_dir=str(transcripts_dir)
        )
        print("Successfully connected to Elasticsearch")
    except Exception as e:
        print(f"Failed to initialize store: {str(e)}")
        return

    # Get list of videos, transcripts, and metadata
    videos = get_video_files(videos_dir)
    transcript_map = get_transcript_files(transcripts_dir)
    video_metadata = get_video_metadata(str(videos_dir))

    print("\nProcessing videos...")
    failed_videos = []
    success_count = 0
    for video in videos:
        video_lower = video.lower()
        try:
            # Check if video exists in store
            video_exists = store.is_video_upserted(video)
            
            # Handle metadata-only update
            if args.metadata_only:
                if video_exists:
                    try:
                        store.update_video_metadata(video)
                        success_count += 1
                        print(f"✓ {video}: Metadata updated")
                    except Exception as e:
                        error_msg = str(e)
                        print(f"✗ {video}: Error updating metadata - {error_msg}")
                        failed_videos.append((video, f"Metadata update error: {error_msg}"))
                else:
                    print(f"⚠ {video}: Video not found in store, skipping metadata update")
                continue

            # Normal processing
            if video_exists and not args.force:
                print(f"✓ {video}: Already exists in vector store")
                success_count += 1
                continue
                
            # Check if transcript exists
            if video not in transcript_map:
                print(f"✗ {video}: No matching transcript found")
                failed_videos.append((video, "No matching transcript"))
                continue
                
            # Get speaker information from metadata
            metadata = video_metadata.get(video, {})
            speaker = metadata.get("speaker")
            
            # Delete existing video if force update
            if args.force and video_exists:
                print(f"🔄 {video}: Forcing update...")
                store.delete_video(video)
            
            # Upsert the video
            try:
                store.upsert_video(video, transcript_map[video], speaker)
                success_count += 1
                print(f"✓ {video}: Successfully upserted")
            except Exception as e:
                error_msg = str(e)
                if "rate limit" in error_msg.lower():
                    print(f"⚠ Rate limit exceeded. Please wait an hour before trying again.")
                    return
                print(f"✗ {video}: Error during upsert - {error_msg}")
                failed_videos.append((video, f"Upsert error: {error_msg}"))
                
            # Small delay between upserts to avoid rate limits
            time.sleep(1)
            
        except Exception as e:
            error_msg = str(e)
            print(f"✗ {video}: Unexpected error - {error_msg}")
            failed_videos.append((video, f"Unexpected error: {error_msg}"))

    # Print summary
    print(f"\nProcessing complete:")
    print(f"✓ Successfully processed: {success_count}/{len(videos)} videos")
    if failed_videos:
        print("\nFailed videos:")
        for video, reason in failed_videos:
            print(f"✗ {video}: {reason}")

if __name__ == "__main__":
    main()
