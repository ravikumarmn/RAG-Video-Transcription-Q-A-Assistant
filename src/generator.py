from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import os
import openai
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from openai import OpenAI
from openai import AzureOpenAI
from retriever import VideoRetriever
import time
from config_utils import config

load_dotenv()


@dataclass(frozen=True)
class VideoTimestamp:
    """Immutable data class for video timestamps."""
    start: str
    end: str

    def __str__(self) -> str:
        return f"{self.start} - {self.end}"


@dataclass(frozen=True)
class VideoSegment:
    """Immutable data class for video segments."""
    text: str
    video: str
    timestamp: VideoTimestamp
    score: float
    metadata: Dict = field(default_factory=dict, hash=False)

    @classmethod
    def from_dict(cls, data: Dict) -> "VideoSegment":
        """Create a VideoSegment from a dictionary."""
        try:
            return cls(
                text=str(data["text"]).strip(),
                video=str(data["video"]),
                timestamp=VideoTimestamp(
                    start=str(data["timestamp"]["start"]),
                    end=str(data["timestamp"]["end"]),
                ),
                score=float(data["score"]),
                metadata=data.get("metadata", {}),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid segment data: {e}")

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "video": self.video,
            "timestamp": {"start": self.timestamp.start, "end": self.timestamp.end},
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class SearchResponse:
    """Data class for search responses."""
    answer: str
    sources: List[VideoSegment] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def has_error(self) -> bool:
        """Check if response contains an error."""
        return bool(self.error)


class VideoResponseError(Exception):
    """Custom exception for video response generation errors."""
    pass


class VideoResponseGenerator:
    """Video-based question answering using OpenAI or Azure OpenAI."""

    # Class-level constants
    HIGH_CONFIDENCE_THRESHOLD = config.retrieval["similarity_threshold"]
    MAX_TOKENS = config.retrieval["max_tokens"]
    TEMPERATURE = 0.7
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 500
    DEFAULT_SEARCH_LIMIT = config.retrieval["max_sources"]

    def __init__(
        self,
        videos_dir: str = config.paths["videos"],
        transcripts_dir: str = config.paths["transcripts"],
        high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
        provider: str = "openai",  # 'azure' or 'openai'
    ):
        """Initialize the OpenAI response generator."""
        if not isinstance(videos_dir, str) or not isinstance(transcripts_dir, str):
            raise TypeError("Directory paths must be strings")
        if not isinstance(high_confidence_threshold, (int, float)):
            raise TypeError("Confidence threshold must be a number")
        if not 0 <= high_confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        if provider not in ["azure", "openai"]:
            raise ValueError("Provider must be either 'azure' or 'openai'")

        self.provider = provider
        self.high_confidence_threshold = high_confidence_threshold
        self._init_retriever(videos_dir, transcripts_dir)
        self._init_client()

    def _init_retriever(self, videos_dir: str, transcripts_dir: str) -> None:
        """Initialize the video retriever with error handling."""
        try:
            self.retriever = VideoRetriever(videos_dir, transcripts_dir)
        except Exception as e:
            raise VideoResponseError(f"Failed to initialize VideoRetriever: {e}")

    def _init_client(self) -> None:
        """Initialize OpenAI client based on provider."""
        if self.provider == "azure":
            required_env_vars = [
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
            ]

            # Check for missing environment variables
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            if missing_vars:
                raise VideoResponseError(
                    f"Missing required environment variables: {', '.join(missing_vars)}"
                )

            try:
                
                self.client = AzureOpenAI(
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    api_key=os.environ["AZURE_OPENAI_API_KEY"]
                )

            except Exception as e:
                raise VideoResponseError(
                    f"Failed to initialize Azure OpenAI client: {str(e)}"
                )
        else:  # OpenAI
            required_env_vars = ["OPENAI_API_KEY"]

            # Check for missing environment variables
            missing_vars = [var for var in required_env_vars if not os.getenv(var)]
            if missing_vars:
                raise VideoResponseError(
                    f"Missing required environment variables: {', '.join(missing_vars)}"
                )

            try:
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                raise VideoResponseError(
                    f"Failed to initialize OpenAI client: {str(e)}"
                )

    def _validate_query(self, query: str) -> None:
        """Validate the search query."""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        if len(query.strip()) < self.MIN_QUERY_LENGTH:
            raise ValueError(
                f"Query must be at least {self.MIN_QUERY_LENGTH} characters"
            )
        if len(query) > self.MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query must not exceed {self.MAX_QUERY_LENGTH} characters"
            )

    def _format_context(self, segments: Tuple[VideoSegment, ...]) -> str:
        """Format video segments into a context string."""
        return "\n\n".join(
            f"From video {segment['video']} ({segment['timestamp']}):\n{segment['text']}"
            for segment in segments
        )

    def _display_segments(self, segments: List[Dict]) -> None:
        """Display relevant segments if any are found."""
        if segments:
            print("\nRelevant segments found:")
            for i, segment in enumerate(segments, 1):
                print(f"\n{i}. Video: {segment.get('video', 'Unknown')}")
                timestamp = segment.get("timestamp", {})
                print(
                    f"   Timestamp: {timestamp.get('start', 'N/A')} - {timestamp.get('end', 'N/A')}"
                )
                print(f"   Text: {segment.get('text', 'No text available')}")
                print(f"   Confidence Score: {segment.get('score', 0):.4f}")
        else:
            print("\nNo relevant segments found.")

    def generate_response(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_LIMIT,
        score_threshold: float = None,
    ) -> SearchResponse:
        """Generate a response to a video-related query using OpenAI."""
        try:
            self._validate_query(query)
            segments = self.retriever.search(query)

            if not segments:
                return SearchResponse(
                    answer="I couldn't find any relevant information in the video transcripts to answer your question.",
                    sources=[],
                )

            # self._display_segments(segments)
            context = self._format_context(tuple(segments))

            # Create chat completion messages
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions about videos "
                        "based on their transcripts. Use the provided video segments to "
                        "answer the question. If you're not sure about something, say so "
                        "rather than making things up."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context from video transcripts:\n{context}\n\nQuestion: {query}",
                },
            ]
            if self.provider == "azure":
                model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
            else:
                model_name = os.getenv("OPENAI_GENERATION_MODEL", "gpt-3.5-turbo")  # Use gpt-3.5-turbo as fallback

            try:
                # Generate response using the appropriate client
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.TEMPERATURE,
                    max_tokens=self.MAX_TOKENS,
                )

                return SearchResponse(
                    answer=response.choices[0].message.content.strip(),
                    sources=sorted(segments, key=lambda x: x["score"], reverse=True)[
                        :k
                    ],
                )
            except Exception as e:
                raise VideoResponseError(f"Failed to generate response: {str(e)}")

        except (ValueError, TypeError) as e:
            return SearchResponse(
                answer="I encountered an error while processing your question.",
                error=str(e),
            )
        except Exception as e:
            return SearchResponse(
                answer="I encountered an unexpected error while processing your question.",
                error=str(e),
            )


def display_response(response: SearchResponse) -> None:
    """Display the response in a formatted way."""
    if response.has_error:
        print(f"\nError: {response.error}")
        return

    print(f"\nAnswer: {response.answer}")

    if response.sources:
        print("\nSources:")
        for i, source in enumerate(response.sources, 1):
            print(f"\n{i}. Video: {source['video']}")
            print(f"   Timestamp: {source['timestamp']}")
            print(f"   Confidence Score: {source['score']:.4f}")


def main():
    """Run test queries with proper error handling."""
    try:
        # Initialize generator
        generator = VideoResponseGenerator(
            high_confidence_threshold=0.7, provider="openai"
        )

        # Test query
        query = "what is great red spot?"
        print(f"\nQuery: '{query}'")

        # Generate and display response
        response = generator.generate_response(query=query, k=3)
        display_response(response)

    except Exception as e:
        print(f"An error occurred while processing your request: {e}")


if __name__ == "__main__":
    main()
