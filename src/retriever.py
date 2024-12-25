"""
Video Retrieval Interface using VideoTranscriptionStore
"""
from typing import List, Dict, Optional
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
from vector_store import VideoTranscriptionStore

load_dotenv()


class VideoRetriever:
    def __init__(self, videos_dir: str = "videos", transcripts_dir: str = "transcripts"):
        """Initialize video retriever with directories."""
        # Convert to absolute paths
        self.videos_dir = str(Path(videos_dir).absolute())
        self.transcripts_dir = str(Path(transcripts_dir).absolute())

        # Ensure directories exist
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.transcripts_dir, exist_ok=True)

        # Initialize store
        self.store = VideoTranscriptionStore(
            videos_dir=self.videos_dir,
            transcripts_dir=self.transcripts_dir
        )

        # # Setup transcriber if API key exists
        # api_key = os.getenv("GOOGLE_API_KEY")
        # if api_key:
        #     self.store.init_transcriber(api_key)

    def parse_timestamp(self, time_val: str) -> float:
        """Parse timestamp to seconds."""
        try:
            # If already in seconds format
            if isinstance(time_val, (int, float)):
                return float(time_val)

            # If in HH:MM:SS format
            if ":" in str(time_val):
                parts = str(time_val).split(":")
                if len(parts) == 3:
                    h, m, s = parts
                    return float(h) * 3600 + float(m) * 60 + float(s)
                elif len(parts) == 2:
                    m, s = parts
                    return float(m) * 60 + float(s)

            # Try direct conversion
            return float(time_val)
        except (ValueError, TypeError):
            return 0.0

    def format_timestamp(self, time_val: str) -> str:
        """Convert timestamp to HH:MM:SS format."""
        try:
            seconds = self.parse_timestamp(time_val)
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        except:
            return "00:00:00"

    def format_result(self, result: Dict) -> Dict:
        """Format search result for display."""
        try:
            # Extract metadata
            metadata = result.get("metadata", {})

            # Get timestamps
            start_time = result.get("start_time", metadata.get("start_time", 0))
            end_time = result.get("end_time", metadata.get("end_time", 0))

            # Format result
            formatted = {
                "text": result.get("text", result.get("page_content", "")),
                "video": result.get(
                    "video_filename", metadata.get("video_filename", "")
                ),
                "timestamp": {
                    "start": self.format_timestamp(start_time),
                    "end": self.format_timestamp(end_time),
                },
                "score": float(result.get("score", 0)),
                "metadata": metadata,
            }

            # Ensure we have valid text and video filename
            if not formatted["text"] or not formatted["video"]:
                return None

            return formatted
        except Exception as e:
            print(f"Error formatting result: {e}")
            return None

    def search(
        self, query: str, k: int = 5, score_threshold: float = 0.60  # Lowered threshold
    ) -> List[Dict]:
        """
        Search for video segments matching query.

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score threshold (default: 0.60)

        Returns:
            List of relevant video segments with metadata
        """
        try:
            print(f"\nSearching for query: {query}")
            print(f"Score threshold: {score_threshold}")
            
            # Get results from vector store
            results = self.store.search_transcriptions(
                query=query, k=k, score_threshold=score_threshold
            )

            print(f"Number of raw results: {len(results)}")
            
            if not results:
                print(f"No results found with similarity score >= {score_threshold}")
                return []

            # Format results with deduplication
            seen_texts = {}  # Track unique texts with their highest scores
            formatted_results = []

            for result in results:
                # Format the result
                formatted = self.format_result(result)
                if not formatted:
                    continue
                
                # Check for duplicate text
                text = formatted["text"].strip().lower()
                
                print(f"\nResult found:")
                print(f"Text: {text[:100]}...")  # Print first 100 chars
                print(f"Score: {formatted['score']}")
                print(f"Video: {formatted.get('video_filename', 'N/A')}")

                # If we've seen this text before, only keep the one with higher score
                if text in seen_texts:
                    if formatted["score"] > seen_texts[text]["score"]:
                        # Remove the lower scored duplicate
                        formatted_results.remove(seen_texts[text])
                        formatted_results.append(formatted)
                        seen_texts[text] = formatted
                else:
                    formatted_results.append(formatted)
                    seen_texts[text] = formatted

            # Sort by score in descending order
            formatted_results.sort(key=lambda x: x["score"], reverse=True)
            print(f"\nFinal number of deduplicated results: {len(formatted_results)}")
            return formatted_results

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def process_video(self, video_filename: str) -> None:
        """Process and index a single video."""
        try:
            self.store.upsert_video(video_filename)
        except Exception as e:
            print(f"Error processing video {video_filename}: {e}")

    def process_all_videos(self) -> None:
        """Process all videos in the videos directory."""
        try:
            self.store.upsert_all_videos()
        except Exception as e:
            print(f"Error processing videos: {e}")


def main():
    """Example usage."""
    # Initialize retriever
    retriever = VideoRetriever()

    # Process videos
    print("\nProcessing videos...")
    retriever.process_all_videos()

    # Test queries
    test_queries = [
        "What is langchain?",
        "Tell me about vector stores",
        "How does RAG work?",
        "python tutorial",
    ]

    # Run searches
    for query in test_queries:
        print(f"\n\n=== Search Results for: '{query}' ===")
        results = retriever.search(query=query, k=5, score_threshold=0.70)

        if not results:
            print("No results found.")
            continue

        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Score: {result['score']:.3f}")
            print(f"Video: {result['video']}")
            print(
                f"Time: {result['timestamp']['start']} - {result['timestamp']['end']}"
            )
            print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()
