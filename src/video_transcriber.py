import os
from pathlib import Path
from datetime import datetime
import google.generativeai as genai
from moviepy.editor import VideoFileClip
import tempfile

TRANSCRIBE_VIDEO_PROMPT = """
Transcribe the video audio into the WEBVTT format as shown below:

WEBVTT

00:00:46.000 --> 00:00:50.020
<v SpeakerName>OK, So this is our agenda for today.</v>

00:00:50.440 --> 00:00:51.050
<v SpeakerName>OK.</v>

00:00:51.710 --> 00:00:53.640
<v SpeakerName>Timewise, we're probably talking about a couple of hours.</v>

- Use exact timestamps.
- Replace "SpeakerName" with the actual speaker's name if provided.
- Maintain accuracy in both the transcription and formatting.
- Use the speaker tag format <v SpeakerName> and </v> consistently.
"""

class VideoTranscriber:
    def __init__(self, api_key: str):
        """Initialize the transcriber with Gemini API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro-vision')
    
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file."""
        video = VideoFileClip(video_path)
        
        # Create a temporary file for the audio
        temp_dir = tempfile.gettempdir()
        temp_audio_path = os.path.join(temp_dir, "temp_audio.mp3")
        
        # Extract audio
        video.audio.write_audiofile(temp_audio_path, codec='mp3')
        video.close()
        
        return temp_audio_path
    
    def transcribe_video(self, video_path: str, output_dir: str) -> str:
        """
        Transcribe a video file using Gemini and save as VTT format.
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save the transcript
            
        Returns:
            Path to the generated transcript file
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract audio from video
        audio_path = self.extract_audio(str(video_path))
        
        try:
            # Generate transcript using Gemini
            print(f"Transcribing {video_path.name}...")
            
            # Load the audio file
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Create the prompt with the audio data
            response = self.model.generate_content([
                TRANSCRIBE_VIDEO_PROMPT,
                audio_data
            ])
            
            transcript_text = response.text
            
            # Generate output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{video_path.stem}_{timestamp}.vtt"
            output_path = output_dir / output_filename
            
            # Save the transcript
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            
            print(f"Transcript saved to: {output_path}")
            return str(output_path)
            
        finally:
            # Clean up temporary audio file
            try:
                os.remove(audio_path)
            except:
                pass
