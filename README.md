# RAG Video Transcription

A system for processing video transcripts using Retrieval Augmented Generation (RAG) with Elasticsearch as the vector store.

## Features

- Video transcript extraction and processing
- RAG implementation with Elasticsearch
- Streamlit-based user interface
- Support for various video formats
- Question answering based on video content
- Flexible video processing options (metadata updates, force updates)

## Prerequisites

- Python 3.8+
- Elasticsearch 8.0+
- OpenAI API key
- Docker (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ravikumarmn/RAG-Video-Transcription.git
cd RAG-Video-Transcription
```

2. Create and activate a virtual environment:

For Linux/macOS:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # For Linux
```

For Windows:
```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# For Command Prompt
.venv\Scripts\activate.bat

# For PowerShell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .  # Install the package in editable mode
```

4. Set up environment variables:
Create a `.env` file with the following:
```
OPENAI_API_KEY=your_openai_api_key
```

Replace `your_openai_api_key` with your actual OpenAI API key.

## Project Structure

- `src/`: Core source code
  - `app.py`: Main Streamlit application
  - `vector_store.py`: Elasticsearch vector store implementation
  - `generator.py`: RAG implementation
  - `upsert_videos.py`: Video processing and ingestion
- `config/`: Configuration files
- `data/`: Data storage
  - `videos/`: Video files (.mp4)
  - `transcripts/`: Transcript files (.vtt)
- `videos/`: Video storage directory

## Transcript Metadata Format

The system expects video metadata to be provided in JSON format. Each video entry should include the following fields:

- `video_path`: Path to the video file
- `transcript_path`: Path to the transcript file (.vtt format)
- `title`: Title of the video
- `description`: Description of the video content
- `meeting_date`: Date of the video recording (YYYY-MM-DD format)

Example metadata format:
```json
{
  "transcript_metadata": [
    {
      "video_path": "python_tutor.mp4",
      "transcript_path": "python_tutor_20241130_113230.vtt",
      "title": "Python Tutorials",
      "description": "Educational video about Python programming fundamentals and concepts",
      "meeting_date": "2024-11-30"
    }
  ]
}
```

## Usage

1. Start Elasticsearch:
```bash
docker-compose up -d elasticsearch
```

2. Process videos:

Basic processing:
```bash
python src/upsert_videos.py
```

Update only metadata for existing videos:
```bash
python src/upsert_videos.py --metadata-only
```

Force update existing videos:
```bash
python src/upsert_videos.py --force
```

Force update with metadata only:
```bash
python src/upsert_videos.py --force --metadata-only
```

3. Run the Streamlit application:
```bash
streamlit run src/app.py
```
### Processing Options

- `--metadata-only`: Updates only the metadata for videos that are already in the system, without reprocessing video content or transcripts
- `--force`: Forces reprocessing of videos even if they already exist in the system. Useful when you want to update existing video content or fix processing issues

## Docker Deployment

Build and run using Docker Compose:
```bash
docker-compose up --build
```

The application will be available at `http://localhost:8501`
