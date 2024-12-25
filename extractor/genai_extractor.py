import os
import time
import json
import google.generativeai as genai
from datetime import datetime
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
VIDEOS_DIR = "data/videos"
TRANSCRIPTS_DIR = "data/transcripts"
SUPPORTED_VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
METADATA_FILE = "config/index_metadata.json"


def load_metadata():
    """Load or create the metadata file."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"transcript_metadata": []}


def save_metadata(metadata):
    """Save metadata to file."""
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def get_video_metadata(video_path):
    """Get metadata for a specific video."""
    metadata = load_metadata()
    video_name = os.path.basename(video_path)
    
    for entry in metadata.get("transcript_metadata", []):
        if entry.get("metadata", {}).get("file_path") == video_name:
            return entry["metadata"]
    return None


def should_process_video(video_path):
    """Check if a video needs to be processed based on modification time."""
    video_metadata = get_video_metadata(video_path)
    if not video_metadata:
        return True
    
    video_stat = os.stat(video_path)
    video_mtime = datetime.fromtimestamp(video_stat.st_mtime)
    last_processed = datetime.fromisoformat(video_metadata["last_processed"])
    
    return video_mtime > last_processed


def update_metadata(video_path, vtt_file):
    """Update metadata for a processed video."""
    metadata = load_metadata()
    video_name = os.path.basename(video_path)
    file_stat = os.stat(video_path)
    
    # Create new metadata entry
    new_metadata = {
        "file_path": video_name,
        "last_processed": datetime.now().isoformat(),
        "file_size": file_stat.st_size,
        "modified_time": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
        "vtt_file": os.path.basename(vtt_file),
    }
    
    # Find and update existing entry or add new one
    found = False
    for entry in metadata.get("transcript_metadata", []):
        if entry.get("metadata", {}).get("file_path") == video_name:
            entry["metadata"] = new_metadata
            found = True
            break
    
    if not found:
        if not metadata.get("transcript_metadata"):
            metadata["transcript_metadata"] = []
        metadata["transcript_metadata"].append({"metadata": new_metadata})
    
    save_metadata(metadata)
    return new_metadata


def get_video_files():
    """Get all video files from the videos directory."""
    video_files = []
    for filename in os.listdir(VIDEOS_DIR):
        if filename.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
            video_files.append(os.path.join(VIDEOS_DIR, filename))
    return video_files


def generate_filename(video_filename, extension):
    """Generate a filename based on the video filename and current timestamp."""
    base_name = os.path.splitext(video_filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"


def format_timestamp(timestamp):
    """Format timestamp to ensure exactly 3 decimal places."""
    if '.' in timestamp:
        main, decimal = timestamp.split('.')
        decimal = decimal[:3].ljust(3, '0')
        return f"{main}.{decimal}"
    return f"{timestamp}.000"


def is_uuid_line(line):
    """Check if a line contains UUID pattern."""
    # UUID pattern with or without dashes
    uuid_pattern = r'[0-9a-f]{8}[-]?[0-9a-f]{4}[-]?[0-9a-f]{4}[-]?[0-9a-f]{4}[-]?[0-9a-f]{12}'
    return bool(re.search(uuid_pattern, line))


def clean_vtt_content(content):
    """Clean and format the VTT content."""
    lines = content.strip().split('\n')
    cleaned_lines = ['WEBVTT', '']
    
    current_segment = None
    
    for line in lines:
        line = line.strip()
        if not line or line == 'WEBVTT' or is_uuid_line(line):
            continue
        
        # Format timestamps if present
        if '-->' in line:
            if current_segment:
                cleaned_lines.extend(current_segment)
                cleaned_lines.append('')
            
            # Format timestamp line
            start, end = line.split(' --> ')
            timestamp_line = f"{format_timestamp(start.strip())} --> {format_timestamp(end.strip())}"
            current_segment = [timestamp_line]
        
        elif line and current_segment is not None:
            # Handle text content
            if '<v' in line:
                # Keep original speaker tag if present
                current_segment.append(line)
            else:
                # Add default speaker tag if missing
                current_segment.append(f"<v Speaker>{line}</v>")
    
    # Add the last segment
    if current_segment:
        cleaned_lines.extend(current_segment)
        cleaned_lines.append('')
    
    return '\n'.join(cleaned_lines)


def extract_transcript_data(content):
    """Extract transcript data into a structured format."""
    segments = []
    lines = content.strip().split('\n')
    
    current_segment = None
    
    for line in lines:
        line = line.strip()
        if not line or line == 'WEBVTT' or is_uuid_line(line):
            continue
        
        if '-->' in line:
            if current_segment:
                segments.append(current_segment)
            
            start, end = line.split(' --> ')
            current_segment = {
                'start_time': format_timestamp(start.strip()),
                'end_time': format_timestamp(end.strip()),
                'text': '',
                'speaker': 'Speaker'
            }
        elif current_segment is not None and '<v' in line:
            # Extract speaker name and text
            match = re.match(r'<v ([^>]+)>(.*)</v>', line)
            if match:
                current_segment['speaker'] = match.group(1)
                current_segment['text'] = match.group(2).strip()
    
    # Add the last segment
    if current_segment:
        segments.append(current_segment)
    
    return segments


def save_vtt_content(content, output_path):
    """Save the VTT content to a file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved VTT transcript to: {output_path}")


# def save_json_content(transcript_data, metadata, output_path):
#     """Save the transcript data as JSON."""
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump({
#             'metadata': metadata,
#             'transcript': transcript_data
#         }, f, indent=2, ensure_ascii=False)
#     print(f"Saved JSON transcript to: {output_path}")


def transcribe_video(video_file_path, output_dir="data/transcripts"):
    """Transcribe video using Gemini and save as VTT and JSON files."""
    try:
        # Configure Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

        # Generate output filenames
        video_filename = os.path.basename(video_file_path)
        vtt_file = os.path.join(output_dir, generate_filename(video_filename, ".vtt"))
        # json_file = os.path.join(output_dir, generate_filename(video_filename, ".json"))

        # Upload video file
        print(f"Uploading file: {video_file_path}")
        video_file = genai.upload_file(path=video_file_path)
        print(f"Completed upload: {video_file.uri}")

        # Check whether the file is ready to be used
        while video_file.state.name == "PROCESSING":
            print(".", end="")
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)

        # Initialize model
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")

        # Define transcription prompt
        TRANSCRIBE_VIDEO_PROMPT = """
        Transcribe the video audio into WebVTT format following these exact rules:
        1. Start with only "WEBVTT" on the first line
        2. Add exactly one blank line after WEBVTT
        3. For each segment:
           - Use timestamps in format HH:MM:SS.mmm (exactly 3 decimal places)
           - Format: XX:XX:XX.XXX --> XX:XX:XX.XXX
           - Each timestamp line must be followed by a speaker tag line
           - Use format: <v Speaker Name>text content</v>
           - Keep speaker name consistent throughout the transcript
           - Add exactly one blank line between segments
        4. Keep segments around 5 seconds each
        5. Do not add any extra text or formatting

        Example format:
        WEBVTT

        00:00:01.000 --> 00:00:05.000
        <v John Smith>Hello everyone, welcome to today's presentation.</v>

        00:00:05.500 --> 00:00:10.000
        <v John Smith>Today we'll be discussing the importance of AI.</v>
        """

        # Generate transcription
        print("Generating transcription...")
        response = model.generate_content(
            [TRANSCRIBE_VIDEO_PROMPT, video_file],
            generation_config={"temperature": 0.1}
        )

        # Clean and format the content
        formatted_content = clean_vtt_content(response.text)

        # Extract transcript data
        transcript_data = extract_transcript_data(formatted_content)

        # Update metadata
        metadata = update_metadata(video_file_path, vtt_file)

        # Save files
        save_vtt_content(formatted_content, vtt_file)
        # save_json_content(transcript_data, metadata, json_file)

        return vtt_file

    except Exception as e:
        print(f"Error transcribing video {video_file_path}: {str(e)}")
        return None, None


def transcribe_videos():
    """Process all videos in the videos directory."""
    try:
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        video_files = get_video_files()
        
        if not video_files:
            print("No video files found in the videos directory.")
            return
        
        print(f"Found {len(video_files)} video files.")
        for video_file in video_files:
            if should_process_video(video_file):
                print(f"\nProcessing video: {video_file}")
                vtt_file = transcribe_video(video_file)
                if vtt_file:
                    print(f"Successfully processed {video_file}")
            else:
                print(f"\nSkipping {video_file} - already processed")
        
        print("\nAll videos processed successfully!")
    
    except Exception as e:
        print(f"Error processing videos: {str(e)}")


if __name__ == "__main__":
    transcribe_videos()
