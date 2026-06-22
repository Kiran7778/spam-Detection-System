"""
STEP 11: AWS S3 + SQS Integration (Adaptive Shield Wiring).

This script provides the integration layer between the FastAPI moderator 
and the human-in-the-loop review system.

Logic:
    1. If a prediction is UNCERTAIN (low confidence), we push to SQS.
    2. We provide a way to 'upload' human-labeled data to S3.
    3. If AWS is not configured, it falls back to a 'Mock' mode 
       (logging to data/processed/human_review_log.json).
"""
import json
import time
from pathlib import Path

# Try to import boto3 (AWS SDK)
try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
except ImportError:
    boto3 = None

import config

class ModerationBuffer:
    """
    Handles buffering uncertain samples to AWS SQS and labeled data to S3.
    """
    
    def __init__(self):
        self.use_mock = config.USE_MOCK_AWS or (boto3 is None)
        self.sqs = None
        self.s3 = None
        self.local_log = config.DATA_DIR / "human_review_buffer.jsonl"
        
        if not self.use_mock:
            try:
                # Try to initialize real AWS clients
                self.sqs = boto3.client('sqs', region_name=config.AWS_REGION)
                self.s3 = boto3.client('s3', region_name=config.AWS_REGION)
                # Verify connection
                self.sqs.list_queues()
                print(f"[OK] Connected to AWS SQS in {config.AWS_REGION}")
            except Exception as e:
                print(f"[!] AWS Connection failed: {e}")
                print("    Switching to MOCK mode (local logging).")
                self.use_mock = True

        if self.use_mock:
            print(f"[INFO] Using MOCK AWS Integration. Logging to: {self.local_log.name}")

    def send_to_review(self, text: str, prediction: str, confidence: float, username: str = "anonymous"):
        """
        Push an uncertain message to the human review queue.
        """
        payload = {
            "text": text,
            "model_prediction": prediction,
            "model_confidence": round(confidence, 4),
            "timestamp": time.time(),
            "status": "pending_human_review",
            "username": username
        }
        
        if self.use_mock:
            # Simulate SQS by appending to a local JSONL file
            with open(self.local_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
            return {"status": "buffered_locally", "path": str(self.local_log)}
        
        else:
            # Real SQS push
            try:
                queue_url = self.sqs.get_queue_url(QueueName=config.SQS_QUEUE_NAME)['QueueUrl']
                self.sqs.send_message(
                    QueueUrl=queue_url,
                    MessageBody=json.dumps(payload)
                )
                return {"status": "pushed_to_sqs", "queue": config.SQS_QUEUE_NAME}
            except ClientError as e:
                print(f"[ERROR] SQS Push failed: {e}")
                return {"status": "error", "message": str(e)}

    def upload_labeled_data(self, file_path: Path):
        """
        Upload a batch of newly labeled data to S3 for the next training cycle.
        """
        if self.use_mock:
            print(f"[MOCK] Simulating S3 upload for: {file_path.name}")
            return True
        
        try:
            self.s3.upload_file(str(file_path), config.S3_BUCKET_NAME, file_path.name)
            print(f"[OK] Uploaded {file_path.name} to S3 bucket {config.S3_BUCKET_NAME}")
            return True
        except Exception as e:
            print(f"[ERROR] S3 Upload failed: {e}")
            return False

# Global instance for the API to use
buffer = ModerationBuffer()

if __name__ == "__main__":
    # Test the integration
    print("\nTesting AWS Integration Layer...")
    res = buffer.send_to_review(
        "Free entry to win a prize! (Confidence is low)", 
        "spam", 
        0.72
    )
    print(f"Result: {res}")
