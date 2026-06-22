"""
STEP 9: Experiment Tracking with MLflow.

This script wraps Step 4 (Training) with MLflow logging.
It tracks:
    - Parameters (TF-IDF settings, SGD alpha, etc.)
    - Metrics (Accuracy, F1, Train time)
    - Artifacts (The saved .pkl models)

To view the dashboard after running:
    mlflow ui
"""
import sys
import time
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score

# Try to import MLflow
try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    print("[ERROR] MLflow not found. Please run: py -m pip install mlflow")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
import config

# Ignore some sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def train_with_mlflow():
    """Train the model and log everything to MLflow."""

    # 1. Set up MLflow Tracking URI to a local folder in the workspace
    # This avoids issues with spaces in user paths on Windows
    tracking_path = Path(__file__).parent / "mlruns"
    tracking_path.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{tracking_path.absolute().as_posix()}")
    
    # 2. Set up MLflow Experiment
    mlflow.set_experiment("SMS_Spam_Moderation")
    
    with mlflow.start_run(run_name="Baseline_SGD"):
        print("=" * 50)
        print("STEP 9: Training with MLflow Tracking")
        print("=" * 50)

        # --- Log Parameters from config.py ---
        mlflow.log_params({
            "tfidf_max_features": config.TFIDF_MAX_FEATURES,
            "tfidf_ngram_range": str(config.TFIDF_NGRAM_RANGE),
            "sgd_loss": config.SGD_LOSS,
            "sgd_alpha": config.SGD_ALPHA,
            "train_ratio": config.TRAIN_RATIO
        })

        # --- Log Input Dataset Info ---
        print("  Loading data...")
        df_train = pd.read_csv(config.TRAIN_FILE)
        df_val = pd.read_csv(config.VAL_FILE)
        
        # Track the training dataset in MLflow
        dataset = mlflow.data.from_pandas(df_train, source=config.TRAIN_FILE.as_posix(), name="SMS_Spam_Train")
        mlflow.log_input(dataset, context="training")
        
        X_train_text = df_train['text_clean'].fillna('')
        y_train = df_train['label'].values
        X_val_text = df_val['text_clean'].fillna('')
        y_val = df_val['label'].values

        # --- Step 1: Fit TF-IDF ---
        print("  Fitting TF-IDF...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            sublinear_tf=config.TFIDF_SUBLINEAR_TF
        )
        
        t0 = time.time()
        X_train = vectorizer.fit_transform(X_train_text)
        X_val = vectorizer.transform(X_val_text)
        tfidf_time = time.time() - t0
        
        mlflow.log_metric("tfidf_fit_time_s", tfidf_time)
        mlflow.log_param("vocab_size", len(vectorizer.vocabulary_))

        # --- Step 2: Train Classifier ---
        print("  Training SGDClassifier...")
        clf = SGDClassifier(
            loss=config.SGD_LOSS,
            alpha=config.SGD_ALPHA,
            random_state=config.SGD_RANDOM_STATE,
            class_weight='balanced'
        )
        
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        
        mlflow.log_metric("train_time_s", train_time)

        # --- Step 3: Evaluate ---
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        print(f"\n  Results: Accuracy={acc:.4f}, F1={f1:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "val_accuracy": acc,
            "val_f1": f1
        })

        # --- Step 4: Save and Log Artifacts ---
        # We save locally first as usual
        vec_path = config.MODEL_DIR / "tfidf_vectorizer.pkl"
        clf_path = config.MODEL_DIR / "classifier.pkl"
        
        with open(vec_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(clf_path, 'wb') as f:
            pickle.dump(clf, f)
            
        # 1. Log models as artifacts (Pickles)
        mlflow.log_artifact(str(vec_path), artifact_path="pickles")
        mlflow.log_artifact(str(clf_path), artifact_path="pickles")
        
        # 2. Log the config file for reproducibility
        mlflow.log_artifact("config.py", artifact_path="config")
        
        # 3. Log the EDA plots if they exist
        plot_path = config.PLOTS_DIR / "eda_overview.png"
        if plot_path.exists():
            mlflow.log_artifact(str(plot_path), artifact_path="plots")
        
        # 4. Log the scikit-learn model with a SIGNATURE (Input/Output schema)
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train_text[:5], y_pred[:5])
        
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=X_train_text[:5].tolist(),
            code_paths=["step9_mlflow.py", "config.py"]
        )
        
        # 5. Add tags for easier searching
        mlflow.set_tags({
            "project": "Adaptive Shield",
            "model_type": "SGDClassifier",
            "pipeline_step": "Step 9"
        })
        
        print(f"\n[OK] Run complete. Dashbord should now show:")
        print(f"     - Schema (Signature)")
        print(f"     - EDA Plots")
        print(f"     - Source Code")
        print(f"     - Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    train_with_mlflow()
    print("\nTo see the results, run: mlflow ui")
