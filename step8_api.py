"""
STEP 8: FastAPI API for Spam Moderation.

This script wraps the trained ONNX model in a REST API using FastAPI.
In a real production environment, this API would be called by other services
(e.g., a messaging app, a comment section, or a chatbot) to moderate content.

Endpoints:
    GET  /health           -> Simple health check
    POST /moderate         -> Send a message, get a spam/ham prediction
    GET  /api/stats        -> Dashboard statistics
    GET  /api/review-queue -> Pending human review items
    POST /api/submit-review -> Submit corrected label

To run this:
    py -m uvicorn step8_api:app --reload
"""
import time
import os
import json
import csv
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config

# Import our production-ready classifier from Step 7
try:
    from step7_inference import SpamClassifier
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from step7_inference import SpamClassifier

# --- Pydantic Models for Input/Output Validation ---

class ModerateRequest(BaseModel):
    """Input: The text message to moderate."""
    text: str
    username: Optional[str] = "anonymous"

class ModerateResponse(BaseModel):
    """Output: The classification result."""
    label: str
    confidence: float
    elapsed_ms: float
    flagged_for_review: bool

class ReviewSubmitRequest(BaseModel):
    """Input: Human review correction submission."""
    text: str
    prediction: str
    confidence: float
    corrected_label: str
    timestamp: float

# --- Helper functions ---

def count_csv_rows(file_path) -> int:
    """Count number of lines in a CSV (excluding header)."""
    if not file_path.exists():
        return 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f) - 1
    except Exception:
        return 0

def read_review_queue() -> list:
    """Read pending reviews from the local jsonl buffer."""
    queue_path = config.DATA_DIR / "human_review_buffer.jsonl"
    if not queue_path.exists():
        return []
    items = []
    try:
        with open(queue_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    except Exception as e:
        print(f"[ERROR] Reading review queue failed: {e}")
    return items

def write_review_queue(items: list):
    """Write back updated reviews list to the jsonl buffer."""
    queue_path = config.DATA_DIR / "human_review_buffer.jsonl"
    try:
        with open(queue_path, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
    except Exception as e:
        print(f"[ERROR] Writing review queue failed: {e}")

def buffer_for_review(text: str, prediction: str, confidence: float, username: str = "anonymous"):
    """Buffer an uncertain prediction to the local review queue (JSONL file)."""
    payload = {
        "text": text,
        "model_prediction": prediction,
        "model_confidence": round(confidence, 4),
        "timestamp": time.time(),
        "status": "pending_human_review",
        "username": username
    }
    queue_path = config.DATA_DIR / "human_review_buffer.jsonl"
    try:
        with open(queue_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        print(f"[REVIEW] Buffered message for human review (conf: {confidence:.2f})")
    except Exception as e:
        print(f"[ERROR] Failed to buffer review item: {e}")

def log_human_feedback(text: str, label: str):
    """Log verified human feedback to processed dataset folder."""
    feedback_file = config.DATA_DIR / "processed" / "human_feedback_labeled.csv"
    file_exists = feedback_file.exists()
    try:
        with open(feedback_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["text", "label", "source", "timestamp"])
            writer.writerow([text, label, "human_review", time.time()])
    except Exception as e:
        print(f"[ERROR] Logging human feedback failed: {e}")

# --- App Initialization ---

app = FastAPI(
    title="SMS Spam Moderation API",
    description="Production-ready API for classifying SMS messages.",
    version="1.0.0"
)

# Add CORS middleware so the dashboard can connect from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the classifier globally so it loads the model once on startup
try:
    classifier = SpamClassifier()
    print("[OK] SpamClassifier initialized and model loaded.")
except Exception as e:
    print(f"[ERROR] Failed to initialize SpamClassifier: {e}")
    classifier = None

# --- API Endpoints ---

@app.get("/health")
def health_check():
    """Verify the API is alive and the model is loaded."""
    if classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy", "model": "Spam Classifier"}

@app.post("/moderate", response_model=ModerateResponse)
def moderate_content(request: ModerateRequest):
    """Classify a text message as spam or ham."""
    if classifier is None:
        raise HTTPException(status_code=500, detail="Classifier not available")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    start_time = time.time()

    try:
        result = classifier.predict(request.text)
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Inference engine failure")

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

    # If the model is uncertain, push to the local review queue
    flagged_for_review = False
    if result['confidence'] < config.CONFIDENCE_THRESHOLD:
        buffer_for_review(
            text=request.text,
            prediction=result['label'],
            confidence=result['confidence'],
            username=request.username
        )
        flagged_for_review = True

    return ModerateResponse(
        label=result['label'],
        confidence=result['confidence'],
        elapsed_ms=round(elapsed_ms, 2),
        flagged_for_review=flagged_for_review
    )

# --- Dashboard API Endpoints ---

@app.get("/")
def serve_dashboard_home():
    """Serve the main HTML page of the dashboard."""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Dashboard index.html not found")
    return FileResponse(index_path)

@app.get("/api/stats")
def get_dashboard_stats():
    """Retrieve database, model configuration, and pipeline stats."""
    train_size = count_csv_rows(config.TRAIN_FILE)
    val_size = count_csv_rows(config.VAL_FILE)
    test_size = count_csv_rows(config.TEST_FILE)

    queue_items = read_review_queue()
    review_queue_pending = len(queue_items)

    if classifier is not None:
        inference_engine = "ONNX Runtime" if classifier.use_onnx else "scikit-learn (Pickle)"
    else:
        inference_engine = "Not Loaded"

    return {
        "dataset": {
            "train": train_size,
            "val": val_size,
            "test": test_size
        },
        "review_queue_pending": review_queue_pending,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "inference_engine": inference_engine,
        "aws_mode": "Local Mode (No AWS)"
    }

@app.get("/api/review-queue")
def get_review_queue():
    """Return all pending messages requiring human verification."""
    return read_review_queue()

@app.post("/api/submit-review")
def submit_review_decision(review: ReviewSubmitRequest):
    """Submit corrected human label and resolve item in queue."""
    log_human_feedback(review.text, review.corrected_label)

    items = read_review_queue()
    new_items = []
    removed = False

    for item in items:
        if item.get("text") == review.text and abs(item.get("timestamp", 0) - review.timestamp) < 0.1:
            removed = True
            continue
        new_items.append(item)

    if not removed:
        new_items = []
        for item in items:
            if item.get("text") == review.text:
                removed = True
                continue
            new_items.append(item)

    write_review_queue(new_items)
    return {"status": "success", "removed": removed}

# Mount static folders for style/js files and pipeline output plots
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

plots_dir = str(config.PLOTS_DIR)
if os.path.exists(plots_dir):
    app.mount("/plots", StaticFiles(directory=plots_dir), name="plots")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 55)
    print("  ADAPTIVE SHIELD - SMS Spam Moderation Server")
    print("=" * 55)
    print("  Dashboard:     http://127.0.0.1:8000/")
    print("  API Docs:      http://127.0.0.1:8000/docs")
    print("  Health Check:  http://127.0.0.1:8000/health")
    print("=" * 55 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
