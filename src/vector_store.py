from langchain.schema import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import time
from elasticsearch import Elasticsearch
from typing import List, Dict, Optional
import json
from datetime import datetime
from transcript_processor import TranscriptProcessor
from pathlib import Path

load_dotenv()


class VideoTranscriptionStore:
    def __init__(self, videos_dir: str, transcripts_dir: str):
        self.vector_store = self.init_vector_store()
        self.processor = TranscriptProcessor(videos_dir, transcripts_dir)
        self.transcriber = None  # Will be initialized when needed

    def init_vector_store(self) -> ElasticsearchStore:
        # Initialize OpenAI embeddings with smaller dimensions for testing
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Use smaller model for testing
            max_retries=3,
            request_timeout=30,
        )

        # Wait for Elasticsearch to be ready
        es_client = Elasticsearch(
            os.getenv("ELASTICSEARCH_URL"),
            basic_auth=(
                os.getenv("ELASTICSEARCH_USERNAME"),
                os.getenv("ELASTICSEARCH_PASSWORD"),
            ),
            retry_on_timeout=True,
            max_retries=3,
            request_timeout=30,
        )

        # Wait for up to 30 seconds for Elasticsearch to be ready
        start_time = time.time()
        while time.time() - start_time < 30:
            try:
                if es_client.ping():
                    print("Successfully connected to Elasticsearch")

                    # Configure index settings if it doesn't exist
                    if not es_client.indices.exists(index="video-transcriptions"):
                        es_client.indices.create(
                            index="video-transcriptions",
                            body={
                                "settings": {
                                    "number_of_shards": 1,
                                    "number_of_replicas": 0,
                                    "refresh_interval": "30s",
                                    "index": {"max_result_window": 10000},
                                },
                                "mappings": {
                                    "properties": {
                                        "text": {"type": "text"},
                                        "embedding": {
                                            "type": "dense_vector",
                                            "dims": 1536,
                                        },
                                        "metadata": {"type": "object"},
                                    }
                                },
                            },
                        )
                        print(
                            "Created video-transcriptions index with optimized settings"
                        )
                    break
            except Exception as e:
                print(f"Waiting for Elasticsearch to be ready... {str(e)}")
                time.sleep(2)
        else:
            raise Exception("Could not connect to Elasticsearch after 30 seconds")

        # Initialize vector store
        return ElasticsearchStore(
            es_url=os.getenv("ELASTICSEARCH_URL"),
            index_name="video-transcriptions",
            embedding=embeddings,
            es_user=os.getenv("ELASTICSEARCH_USERNAME"),
            es_password=os.getenv("ELASTICSEARCH_PASSWORD"),
        )

    # def init_transcriber(self, api_key: str):
    #     """Initialize the video transcriber with API key."""
    #     # from video_transcriber import VideoTranscriber

    #     self.transcriber = VideoTranscriber(api_key)

    def is_video_upserted(self, video_filename: str) -> bool:
        """Check if a video is already upserted in the vector store."""
        try:
            # Use a direct Elasticsearch query to check for any documents with this video filename
            es_client = self.vector_store.client

            # First check if index exists
            if not es_client.indices.exists(index="video-transcriptions"):
                return False

            response = es_client.search(
                index="video-transcriptions",  # Use the hardcoded index name
                body={
                    "query": {
                        "term": {"metadata.video_filename.keyword": video_filename}
                    },
                    "size": 1,
                },
            )

            return response["hits"]["total"]["value"] > 0
        except Exception as e:
            print(f"Error checking video status: {str(e)}")
            # If check fails, assume not upserted to be safe
            return False

    def upsert_video(self, video_filename: str) -> None:
        """
        Process and upsert a single video's transcription.

        Args:
            video_filename: Name of the video file
        """
        try:
            # Check if video exists
            video_path = Path(self.processor.videos_dir) / video_filename
            if not video_path.exists():
                raise FileNotFoundError(f"Video file {video_filename} not found.")

            # Check if video is already upserted
            if self.is_video_upserted(video_filename):
                print(
                    f"Video {video_filename} is already in the vector store. Skipping."
                )
                return

            # Try to find existing transcript
            transcript_path = self.processor.find_matching_transcript(video_filename)
            if not transcript_path:
                raise FileNotFoundError(
                    f"No transcript found for {video_filename} and no transcriber configured."
                )

            # Process the video and its transcript
            print(f"Processing {video_filename}...")
            result = self.processor.process_video(video_filename)
            if not result or not result.get("segments"):
                raise ValueError(f"No valid segments found in video: {video_filename}")

            # Track unique segments to prevent duplicates
            seen_segments = set()
            documents = []
            failed_segments = []

            # Pre-process all segments first
            for segment in result["segments"]:
                try:
                    # Clean and validate the text
                    text = segment["text"].strip()
                    if not text or len(text) < 3:  # Skip very short segments
                        continue

                    # Create a unique key for this segment
                    segment_key = (
                        text,
                        segment["start_time"],
                        segment["end_time"],
                    )

                    # Skip if we've seen this segment before
                    if segment_key in seen_segments:
                        continue

                    seen_segments.add(segment_key)

                    # Create a Document object with the segment text and metadata
                    metadata = result["metadata"].copy()
                    metadata.update(
                        {
                            "start_time": segment["start_time"],
                            "end_time": segment["end_time"],
                            "video_filename": video_filename,
                            "segment_index": len(documents),
                            "segment_id": f"{video_filename}_{len(documents)}",
                        }
                    )

                    doc = Document(page_content=text, metadata=metadata)
                    documents.append(doc)
                except Exception as e:
                    print(f"Warning: Failed to process segment: {str(e)}")
                    failed_segments.append((segment, str(e)))

            if not documents:
                raise ValueError(
                    f"No valid segments to index in video: {video_filename}"
                )

            print(f"Found {len(documents)} valid segments in {video_filename}")

            # Add documents to the vector store in smaller batches
            batch_size = 5  # Even smaller batch size for testing
            successful_segments = 0

            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                try:
                    # Try to add the batch
                    print(
                        f"Attempting to index batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}..."
                    )
                    import uuid
                    uuids = [str(uuid.uuid4()) for _ in range(len(batch))]
                    self.vector_store.add_documents(batch, uuids)
                    successful_segments += len(batch)
                    print(
                        f"Successfully indexed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} for {video_filename}"
                    )
                except Exception as batch_error:
                    print(f"Batch {i//batch_size + 1} failed: {str(batch_error)}")
                    print("Trying individual documents...")

                    # If batch fails, try each document individually
                    for doc in batch:
                        try:
                            self.vector_store.add_documents([doc])
                            successful_segments += 1
                            print(
                                f"Successfully indexed individual segment: {doc.page_content[:50]}..."
                            )
                        except Exception as doc_error:
                            error_msg = str(doc_error)
                            print(f"Failed to index segment: {error_msg}")
                            failed_segments.append((doc.page_content, error_msg))

                # Small delay between batches
                time.sleep(1)

            # Report results
            if successful_segments > 0:
                print(
                    f"Successfully indexed {successful_segments}/{len(documents)} segments from {video_filename}"
                )

            if failed_segments:
                failed_count = len(failed_segments)
                if failed_count == len(documents):
                    raise Exception(
                        f"All {failed_count} segments failed to index in {video_filename}. First error: {failed_segments[0][1]}"
                    )
                else:
                    print(
                        f"Warning: {failed_count} segment(s) failed to index in {video_filename}"
                    )
                    print("First few errors:")
                    for content, error in failed_segments[:3]:
                        print(f"- {content[:50]}...: {error}")

        except Exception as e:
            raise Exception(f"Error processing video {video_filename}: {str(e)}")

    def update_video_metadata(self, video_filename: str) -> None:
        """Update metadata for all segments of a video in the vector store."""
        try:
            # Get the video's current metadata from config
            result = self.processor.process_video(video_filename)
            if not result or not result.get("segments"):
                raise ValueError(f"No valid segments found in video: {video_filename}")

            # Get the base metadata that will be applied to all segments
            base_metadata = result["metadata"]

            # Update all segments for this video
            es_client = self.vector_store.client
            es_client.update_by_query(
                index="video-transcriptions",
                body={
                    "query": {
                        "term": {"metadata.video_filename.keyword": video_filename}
                    },
                    "script": {
                        "source": """
                            // Preserve segment-specific metadata
                            def segmentMeta = ['start_time', 'end_time', 'segment_index', 'segment_id'];
                            def preservedValues = [:];
                            for (def key : segmentMeta) {
                                if (ctx._source.metadata.containsKey(key)) {
                                    preservedValues[key] = ctx._source.metadata[key];
                                }
                            }
                            // Update with new metadata
                            ctx._source.metadata = params.metadata;
                            // Restore segment-specific metadata
                            for (def entry : preservedValues.entrySet()) {
                                ctx._source.metadata[entry.getKey()] = entry.getValue();
                            }
                        """,
                        "params": {"metadata": base_metadata},
                        "lang": "painless"
                    }
                },
                refresh=True
            )
            print(f"✓ Updated metadata for all segments of {video_filename}")
        except Exception as e:
            print(f"Error updating metadata for {video_filename}: {str(e)}")
            raise

    def delete_video(self, video_filename: str) -> None:
        """Delete all segments for a video from the vector store."""
        try:
            es_client = self.vector_store.client
            es_client.delete_by_query(
                index="video-transcriptions",
                body={
                    "query": {
                        "term": {"metadata.video_filename.keyword": video_filename}
                    }
                }
            )
            print(f"Deleted all segments for {video_filename}")
        except Exception as e:
            print(f"Error deleting video {video_filename}: {str(e)}")
            raise

    def upsert_all_videos(self) -> None:
        """Process and upsert all videos in the videos directory."""
        videos_dir = Path(self.processor.videos_dir)
        for video_file in videos_dir.glob("*.mp4"):
            print(f"\nProcessing {video_file.name}...")
            self.upsert_video(video_file.name)

    def search_transcriptions(
        self, query: str, k: int = 5, score_threshold: float = 0.90
    ) -> List[Dict]:
        """
        Search transcriptions using a natural language query.

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score threshold (default: 0.90)

        Returns:
            List of relevant transcription segments with metadata
        """
        try:
            # Get more results initially for better coverage
            results = self.vector_store.similarity_search_with_score(
                query, k=k * 4  # Get even more results to account for filtering,
                # filter=[{"term": metadata}]
            )

            # Format and deduplicate the results
            seen_segments = set()  # Track unique segments
            formatted_results = []

            for doc, score in results:
                # Score from Elasticsearch is already a similarity score (not distance)
                if score >= score_threshold:
                    # Create a unique key using content and timing
                    text = doc.page_content.strip()
                    video_filename = doc.metadata.get("video_filename", "")

                    # Create segment key with video filename to allow similar segments from different videos
                    segment_key = (
                        text,
                        video_filename,
                        doc.metadata.get("start_time"),
                        doc.metadata.get("end_time"),
                    )

                    # Skip if we've seen this exact segment
                    if segment_key in seen_segments:
                        continue

                    seen_segments.add(segment_key)

                    # Format the result
                    result = {
                        "text": text,
                        "video_filename": video_filename,
                        "start_time": doc.metadata.get("start_time", 0),
                        "end_time": doc.metadata.get("end_time", 0),
                        "score": float(score),
                        "metadata": doc.metadata,
                    }
                    formatted_results.append(result)

            # Sort by score in descending order
            formatted_results.sort(key=lambda x: x["score"], reverse=True)

            # Return top k results
            return formatted_results

        except Exception as e:
            print(f"Search error in vector store: {e}")
            return []

    def search_transcriptions_old(
        self, query: str, k: int = 5, score_threshold: float = 0.90
    ) -> List[Dict]:
        """
        Search transcriptions using a natural language query.

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score threshold (default: 0.85)

        Returns:
            List of relevant transcription segments with metadata
        """
        # Perform the search with more results since we'll filter by score
        results = self.vector_store.similarity_search_with_score(
            query, k=k * 3  # Get more results to account for threshold filtering
        )

        # Format and deduplicate the results
        seen_segments = set()  # Track unique segments
        formatted_results = []

        for doc, score in results:
            # Convert distance to similarity score (0 to 1)
            similarity_score = score

            # Skip if below threshold
            if similarity_score < score_threshold:
                continue

            # Create a unique key for each segment using content and timing
            segment_key = (doc.page_content, doc.metadata.get("start_time"), doc.metadata.get("end_time"))

            # Skip if we've seen this segment
            if segment_key in seen_segments:
                continue

            seen_segments.add(segment_key)

            # Format the result
            result = {
                "text": doc.page_content,
                "video_filename": doc.metadata.get("video_filename", ""),
                "start_time": doc.metadata.get("start_time", 0),
                "end_time": doc.metadata.get("end_time", 0),
                "score": similarity_score,
                "metadata": doc.metadata
            }

            formatted_results.append(result)

            # Break if we have enough unique results
            if len(formatted_results) >= k:
                break

        return formatted_results


# def main():
#     # Initialize the store with your video and transcript directories
#     store = VideoTranscriptionStore(
#         videos_dir="/Users/ravikumar/Developer/upwork/RAG-Video-Transcription/data/videos",
#         transcripts_dir="/Users/ravikumar/Developer/upwork/RAG-Video-Transcription/data/transcripts",
#     )

#     # Initialize the transcriber if GEMINI_API_KEY is available
#     gemini_api_key = os.getenv("GOOGLE_API_KEY")
#     if gemini_api_key:
#         store.init_transcriber(gemini_api_key)
#     else:
#         print(
#             "Warning: GOOGLE_API_KEY not found in environment variables. Transcription will be skipped."
#         )

#     # Upsert all videos
#     store.upsert_all_videos()

#     # Example search
#     results = store.search_transcriptions(
#         query="What is great red spot? and how to install python?", k=3,score_threshold=0.80
#     )

#     # Print results
#     print("\nSearch Results:")
#     for result in results:
#         print(f"\nSegment: {result['text']}")
#         print(f"Video: {result['video_filename']}")
#         print(f"Timestamp: {result['start_time']} - {result['end_time']}")
#         print(f"Relevance Score: {result['score']}")
#         print("Additional Metadata:", json.dumps(result["metadata"], indent=2))


# if __name__ == "__main__":
#     main()
