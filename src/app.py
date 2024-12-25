import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Optional
import os
from pathlib import Path
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial, lru_cache
import logging
from generator import (
    VideoResponseGenerator,
    SearchResponse,
    VideoSegment,
    VideoTimestamp,
)
from collections import defaultdict
from config_utils import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Streamlit
st.set_page_config(
    page_title="Video Q&A Assistant",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Apply custom CSS for better UI
st.markdown(
    """
<style>
    /* Increase dialog width */
    .stDialog > div {
        max-width: 99.5% !important;
        max-height: 98vh !important;
    }
    
    /* Improve chat container */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Enhance video display */
    .stVideo {
        width: 100%;
        border-radius: 8px;
    }
    
    /* Better button styling */
    .stButton button {
        width: 100%;
        border-radius: 20px;
        transition: all 0.3s ease;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #FF4B4B !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource()
def get_generator() -> Optional[VideoResponseGenerator]:
    """Initialize and cache the VideoResponseGenerator."""
    try:
        return VideoResponseGenerator(
            videos_dir="videos",
            transcripts_dir="data/transcripts",
            provider="openai",  # Explicitly set OpenAI as the default provider
        )
    except Exception as e:
        logger.error(f"Failed to initialize VideoResponseGenerator: {e}")
        return None


@lru_cache(maxsize=32)
def get_video_title(video_filename: str) -> str:
    """Get formatted video title from filename."""
    return os.path.splitext(video_filename)[0].replace("_", " ").title()


def parse_timestamp(timestamp: str) -> int:
    """Parse timestamp string to seconds."""
    try:
        if ":" in timestamp:
            parts = timestamp.split(":")
            if len(parts) == 2:
                minutes, seconds = map(float, parts)
                return int(minutes * 60 + seconds)
            elif len(parts) == 3:
                hours, minutes, seconds = map(float, parts)
                return int(hours * 3600 + minutes * 60 + seconds)
        return int(float(timestamp))
    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing timestamp {timestamp}: {e}")
        return 0


def filter_top_k_per_video(
    sources, display_k: int = config.display_sources["display_k"]
):
    """Filter and return top k sources per video based on score."""
    video_groups = defaultdict(list)
    for source in sources:
        video_groups[source["video"]].append(source)

    filtered_sources = []
    for video, segments in video_groups.items():
        # Sort by score in descending order (higher scores first)
        top_k = sorted(segments, key=lambda x: x["score"], reverse=True)[:display_k]
        filtered_sources.extend(top_k)

    return filtered_sources


# Filter top k sources per video
@st.dialog("Video Sources", width="large")
def show_sources(sources: List[VideoSegment]):
    """Display video sources in an optimized dialog."""
    if not sources:
        st.warning("No relevant video sources found.")
        return

    # Filter sources to be taken from one file only with high score
    sources = sorted(sources, key=lambda x: (x["video"], x["score"]), reverse=True)
    top_k_sources = filter_top_k_per_video(
        sources, display_k=config.display_sources["display_k"]
    )

    # Display videos in grid
    cols_per_row = min(2, len(top_k_sources))
    if cols_per_row > 0:
        cols = st.columns(cols_per_row)

        for idx, source in enumerate(top_k_sources):
            with cols[idx % cols_per_row]:
                try:
                    video_path = Path("data/videos") / source["video"]
                    video_file = open(video_path, "rb")
                    video_bytes = video_file.read()

                    if not video_path.exists():
                        st.error(f"Video file not found: {source['video']}")
                        continue

                    st.video(
                        video_bytes,
                        start_time=source["timestamp"]["start"],
                        end_time=source["timestamp"]["end"],
                    )
                    # Get title from metadata or format filename
                    title = get_video_title(source["video"])
                    if "metadata" in source:
                        title = source["metadata"].get("title", title)
                    st.markdown(f"**{title}**")

                    # st.markdown(f"**Meeting Date**: {source.metadata.get('meeting_date', '')}")
                    # st.markdown(f"**Timestamp**: {source.timestamp.start} - {source.timestamp.end}")
                    # st.markdown(f"**Confidence Score**: {source.score:.2f}")
                except Exception as e:
                    st.error(f"Error displaying video {source['video']}: {str(e)}")
                    logger.error(f"Error in show_sources: {str(e)}")

    if st.button("Close", key="close_sources"):
        st.session_state.show_sources = None
        st.rerun()


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "max_messages" not in st.session_state:
        st.session_state.max_messages = 50  # Increased from 20
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = None
    if "show_about" not in st.session_state:
        st.session_state.show_about = False
    if "last_response_time" not in st.session_state:
        st.session_state.last_response_time = 0


def main():
    """Main application function."""
    init_session_state()
    
    # Add sidebar
    with st.sidebar:
        st.title("About")
        st.markdown("""
        ### Video Q&A Assistant
        
        An advanced video search and analysis system that helps you:
        - Find specific moments in videos with precise timestamps
        - Access video transcripts for detailed content review
        - Get AI-powered answers to questions about video content
        - Upload and process new videos automatically
        
        #### How it works:
        1. **Video Upload**: Upload your videos and get automatic transcription
        2. **Processing**: Transcripts are processed and stored in vector database
        3. **Smart Search**: System finds relevant video segments with exact timestamps
        4. **Content Access**: Get both video timestamps and transcripts for each result
        5. **AI Response**: GPT generates detailed answers based on the found content
        
        #### Features:
        - 📤 Easy video upload and processing
        - 🎯 Precise video timestamp retrieval
        - 📝 Full transcript access
        - 🔍 Source verification with timestamps
        - 💡 Natural language understanding
        - ⚡ Fast and efficient retrieval
        - 🔄 Automatic vectorstore updates
        
        #### Technology Stack:
        - 🤖 **AI Models**: OpenAI API / Azure OpenAI
        - 🔎 **Search**: Elasticsearch in Docker
        - 🎥 **Processing**: Video transcription & indexing
        - 🗄️ **Storage**: Vector database for efficient search
        - 🚀 **Frontend**: Streamlit
        
        #### Tips for best results:
        - Upload clear, high-quality videos
        - Allow processing time for new uploads
        - Ask specific, detailed questions
        - Reference video content directly
        - Check sources for verification
        """)
        
        # Add a divider
        st.divider()
            
    st.title("Video Q&A Assistant")

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if (
                "sources" in message
                and message["sources"]
                and message["content"] != "I don't have enough information."
            ):
                if st.button("Show Sources", key=f"source_btn_{idx}"):
                    st.session_state.show_sources = message["sources"]
                    st.rerun()

    # Show sources dialog if needed
    if st.session_state.show_sources is not None:
        show_sources(st.session_state.show_sources)

    # Check message limit
    if len(st.session_state.messages) >= st.session_state.max_messages:
        st.warning("🚨 Maximum message limit reached. Please start a new conversation.")
        return

    # Initialize generator
    generator = get_generator()
    if generator is None:
        st.error("⚠️ System initialization failed. Please check configuration.")
        return

    # Handle user input
    if prompt := st.chat_input("Ask a detailed question about the videos..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Track response time
                start_time = time.time()

                with st.spinner("🔍 Searching through videos..."):

                    response = generator.generate_response(
                        query=prompt,
                        k=config.retrieval["max_sources"],
                        score_threshold=config.retrieval["similarity_threshold"],
                    )

                # Calculate response time
                response_time = time.time() - start_time
                st.session_state.last_response_time = response_time

                if response.has_error:
                    st.error(f"😔 {response.error}")
                    message_content = (
                        "I apologize, but I encountered an error while processing your question. "
                        "Please try rephrasing or asking a different question."
                    )
                else:
                    message_content = response.answer
                    # Add response time info
                    # message_content += f"\n\n*Response generated in {response_time:.2f} seconds*"

                st.markdown(message_content)

                # Save message to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": message_content,
                        "sources": response.sources if not response.has_error else [],
                        # "response_time": response_time
                    }
                )

                # Show sources button
                if (
                    response.sources
                    and message_content != "I don't have enough information."
                ):
                    if st.button("Show Sources", key="source_btn_current"):
                        st.session_state.show_sources = response.sources
                        st.rerun()

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                st.error("🚨 An unexpected error occurred. Please try again.")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "I apologize, but I encountered an unexpected error.",
                        "sources": [],
                    }
                )


if __name__ == "__main__":
    main()
