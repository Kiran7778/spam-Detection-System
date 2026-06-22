"""
STEP 10: Active Learning Simulation (The Thesis Proof).

This is the core of the "Adaptive Shield" project. 
It simulates a real-world scenario where labeling data is expensive (human cost).
We compare two strategies:
    1. Random Sampling: Pick messages to label at random.
    2. Active Learning (Uncertainty): Pick messages where the model is "confused".

Goal: Show that Active Learning reaches higher accuracy with FEWER labeled samples.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

# Try to import MLflow for tracking this experiment
try:
    import mlflow
except ImportError:
    mlflow = None

sys.path.insert(0, str(Path(__file__).parent))
import config

def run_active_learning_simulation():
    """Run the AL vs Random simulation and plot results."""
    
    print("=" * 60)
    print("STEP 10: Active Learning Simulation")
    print("=" * 60)

    # 1. Load the full training set (which we will treat as our "Pool")
    df = pd.read_csv(config.TRAIN_FILE)
    df_test = pd.read_csv(config.TEST_FILE)
    
    # Simulation Parameters
    INITIAL_SIZE = 50      # Start with only 50 labeled samples
    SAMPLES_PER_ITER = 50  # Add 50 more samples each round
    ITERATIONS = 12        # Run for 12 rounds
    
    # --- Stable Feature Extraction ---
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE
    )
    
    # To show the TRUE advantage of AL, the vectorizer needs a stable vocabulary.
    print("  Fitting stable TF-IDF on the entire pool...")
    vectorizer.fit(df['text_clean'].fillna(''))
    X_test = vectorizer.transform(df_test['text_clean'].fillna(''))
    y_test = df_test['label']

    results = [] # To store (num_samples, random_f1, al_f1)

    # Prepare pools for both strategies
    # Fixed seed for initial selection to ensure fair comparison
    np.random.seed(config.RANDOM_SEED)
    initial_indices = np.random.choice(df.index, size=INITIAL_SIZE, replace=False)
    
    train_indices_random = list(initial_indices)
    train_indices_al     = list(initial_indices)
    
    # Start MLflow run if available
    if mlflow:
        mlflow.set_experiment("Active_Learning_Simulation")
        run = mlflow.start_run(run_name="AL_vs_Random_Sim_Corrected")
        mlflow.log_params({
            "initial_size": INITIAL_SIZE,
            "samples_per_iter": SAMPLES_PER_ITER,
            "iterations": ITERATIONS,
            "strategy": "Uncertainty vs Random"
        })

    print(f"Starting simulation with {INITIAL_SIZE} samples...")
    
    for i in range(ITERATIONS):
        current_count = len(train_indices_random)
        
        # --- Strategy 1: Random ---
        df_train_rand = df.loc[train_indices_random]
        X_train_rand = vectorizer.transform(df_train_rand['text_clean'].fillna(''))
        y_train_rand = df_train_rand['label']
        
        clf_rand = SGDClassifier(loss='log_loss', random_state=42, class_weight='balanced')
        clf_rand.fit(X_train_rand, y_train_rand)
        
        # Evaluate Random
        y_pred_rand = clf_rand.predict(X_test)
        f1_rand = f1_score(y_test, y_pred_rand)

        # --- Strategy 2: Active Learning (Uncertainty) ---
        df_train_al = df.loc[train_indices_al]
        X_train_al = vectorizer.transform(df_train_al['text_clean'].fillna(''))
        y_train_al = df_train_al['label']
        
        clf_al = SGDClassifier(loss='log_loss', random_state=42, class_weight='balanced')
        clf_al.fit(X_train_al, y_train_al)
        
        # Evaluate AL
        y_pred_al = clf_al.predict(X_test)
        f1_al = f1_score(y_test, y_pred_al)

        print(f"Iter {i+1:2}: Samples={current_count:4} | Random F1={f1_rand:.3f} | AL F1={f1_al:.3f}")
        results.append((current_count, f1_rand, f1_al))
        
        if mlflow:
            mlflow.log_metric("random_f1", f1_rand, step=current_count)
            mlflow.log_metric("al_f1", f1_al, step=current_count)

        # --- Select next samples for next iteration ---
        
        # 1. For Random: Just pick random indices not already in the set
        remaining_rand = df.index.difference(train_indices_random)
        new_rand = np.random.choice(remaining_rand, size=SAMPLES_PER_ITER, replace=False)
        train_indices_random.extend(new_rand)
        
        # 2. For Active Learning: Pick the most "Uncertain" samples
        remaining_al = df.index.difference(train_indices_al)
        df_pool = df.loc[remaining_al]
        
        # Get probabilities for the pool
        X_pool = vectorizer.transform(df_pool['text_clean'].fillna(''))
        probs = clf_al.predict_proba(X_pool)
        
        # Uncertainty Score: How close is the max probability to 0.5?
        # A score of 0.5 means the model is 50/50 confused.
        # Higher score = more uncertain.
        uncertainty_scores = 1 - np.max(probs, axis=1)
        
        # Pick top N uncertain samples
        top_uncertain_idx = np.argsort(uncertainty_scores)[-SAMPLES_PER_ITER:]
        new_al_indices = df_pool.index[top_uncertain_idx]
        train_indices_al.extend(new_al_indices)

    # --- Plotting ---
    counts, rand_f1s, al_f1s = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(counts, rand_f1s, 'o--', label='Random Sampling', color='gray')
    plt.plot(counts, al_f1s, 's-', label='Active Learning (Uncertainty)', color='#3498db', linewidth=2)
    
    plt.title('Learning Curve: Random vs Active Learning', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Labeled Samples')
    plt.ylabel('Test Set F1-Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plot_path = config.PLOTS_DIR / "active_learning_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n[OK] Simulation complete. Plot saved to: {plot_path}")
    
    if mlflow:
        mlflow.log_artifact(str(plot_path))
        mlflow.end_run()
        print("[OK] Results logged to MLflow.")

if __name__ == "__main__":
    run_active_learning_simulation()
