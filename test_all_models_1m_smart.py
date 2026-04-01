import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from models.simple_mf import SimpleMF
from models.biased_mf import BiasedMF
from models.nmf_model import NMF
from evaluation_metrics import calculate_mae
import time
import pickle
import os
from models.bemf_model import BeMF
from models.bnmf_model import BNMF
from models.urp_model import URP

ROOT_DIR = Path(__file__).resolve().parent

print("="*70)
print("TESTING ALL MF MODELS ON MOVIELENS 1M (Smart Version)")
print("Using Optimal Hyperparameters + Model Caching")
print("="*70)

# Setup saving
MODELS_DIR = ROOT_DIR / 'saved_models'
RESULTS_FILE = ROOT_DIR / 'all_models_results_1m.csv'
os.makedirs(MODELS_DIR, exist_ok=True)

# Load existing results if available
if RESULTS_FILE.exists():
    print(f"\n✅ Found existing results: {RESULTS_FILE}")
    df_existing = pd.read_csv(RESULTS_FILE)
    print(f"Already tested: {df_existing['Model'].tolist()}")
    results = df_existing.to_dict('records')
else:
    print("\nNo saved results found. Starting fresh.")
    results = []

# Load data ONCE
print("\nLoading MovieLens 1M...")
data = pd.read_csv(ROOT_DIR / "ml-1m" / "ml-1m" / "ratings.dat",
                   sep='::',
                   engine='python',
                   names=['user_id', 'item_id', 'rating', 'timestamp'],
                   encoding='latin-1')

data['user_id'] = data['user_id'] - 1
data['item_id'] = data['item_id'] - 1

train, test = train_test_split(data, test_size=0.2, random_state=42)
n_users = data['user_id'].max() + 1
n_items = data['item_id'].max() + 1

print(f"Users: {n_users:,}, Items: {n_items:,}")

# Optimal hyperparameters
OPTIMAL_PARAMS = {
    'n_factors': 10,
    'n_epochs': 25,
    'learning_rate': 0.01,
    'regularization': 0.01
}

print(f"\nOptimal Hyperparameters:")
for key, value in OPTIMAL_PARAMS.items():
    print(f"  {key}: {value}")

# Helper function to test a model
def test_model(model_name, model_class, paper_mae, model_kwargs):
    """Test a model with caching"""
    
    # Check if already tested
    existing_models = [r['Model'] for r in results]
    if model_name in existing_models:
        existing_mae = [r['Your_MAE'] for r in results if r['Model'] == model_name][0]
        print(f"\n✓ {model_name}: Already tested (MAE = {existing_mae:.4f}) - SKIPPING")
        return
    
    model_path = MODELS_DIR / f"{model_name.lower()}_model.pkl"
    
    print("\n" + "="*70)
    print(f"Testing {model_name}")
    print("="*70)
    
    start_time = time.time()
    
    # Check if model is saved
    if model_path.exists():
        print(f"⚡ Loading saved {model_name} model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("   Model loaded!")
    else:
        print(f"🔄 Training new {model_name} model...")
        model = model_class(n_users, n_items, **model_kwargs)
        model.train(train)
        
        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"   💾 Model saved to {model_path}")
    
    # Evaluate
    print("   Evaluating...")
    mae = calculate_mae(model, test)
    elapsed = time.time() - start_time
    
    print(f"\n✅ {model_name} Results:")
    print(f"   Your MAE:   {mae:.4f}")
    print(f"   Paper MAE:  {paper_mae:.3f}")
    print(f"   Difference: {mae - paper_mae:.4f} ({'BETTER ✅' if mae < paper_mae else 'WORSE ❌'})")
    print(f"   Time:       {elapsed:.1f}s")
    
    # Save result
    results.append({
        'Model': model_name,
        'Your_MAE': mae,
        'Paper_MAE': paper_mae,
        'Difference': mae - paper_mae,
        'Time_sec': elapsed
    })
    
    # Save results immediately
    df_temp = pd.DataFrame(results)
    df_temp.to_csv(RESULTS_FILE, index=False)
    print(f"   💾 Results saved to {RESULTS_FILE}")

# =========================================
# Test all models
# =========================================

# Model 1: PMF
test_model(
    model_name='PMF',
    model_class=SimpleMF,
    paper_mae=0.729,
    model_kwargs={
        'n_factors': OPTIMAL_PARAMS['n_factors'],
        'learning_rate': OPTIMAL_PARAMS['learning_rate'],
        'n_epochs': OPTIMAL_PARAMS['n_epochs']
    }
)

# Model 2: BiasedMF
test_model(
    model_name='BiasedMF',
    model_class=BiasedMF,
    paper_mae=0.712,
    model_kwargs={
        'n_factors': OPTIMAL_PARAMS['n_factors'],
        'learning_rate': OPTIMAL_PARAMS['learning_rate'],
        'regularization': OPTIMAL_PARAMS['regularization'],
        'n_epochs': OPTIMAL_PARAMS['n_epochs']
    }
)

# Model 3: NMF
test_model(
    model_name='NMF',
    model_class=NMF,
    paper_mae=0.744,
    model_kwargs={
        'n_factors': OPTIMAL_PARAMS['n_factors'],
        'learning_rate': OPTIMAL_PARAMS['learning_rate'],
        'regularization': OPTIMAL_PARAMS['regularization'],
        'n_epochs': OPTIMAL_PARAMS['n_epochs']
    }
)
# Model 4: BeMF
test_model(
    model_name='BeMF',
    model_class=BeMF,
    paper_mae=0.748,
    model_kwargs={
        'n_factors': OPTIMAL_PARAMS['n_factors'],
        'learning_rate': OPTIMAL_PARAMS['learning_rate'],
        'regularization': OPTIMAL_PARAMS['regularization'],
        'n_epochs': OPTIMAL_PARAMS['n_epochs']
    }
)
# Model 5: BNMF
# Model 5: BNMF
test_model(
    model_name='BNMF',
    model_class=BNMF,
    paper_mae=0.693,
    model_kwargs={
        'n_factors': OPTIMAL_PARAMS['n_factors'],
        'alpha': 0.4,  # Paper tested: {0.2, 0.4, 0.6, 0.8}
        'beta': 15,    # Paper tested: {5, 15, 25}
        'n_epochs': OPTIMAL_PARAMS['n_epochs']
    }
)
# Model 6: URP (last one!)
test_model(
    model_name='URP',
    model_class=URP,
    paper_mae=0.795,
    model_kwargs={
        'n_factors': OPTIMAL_PARAMS['n_factors'],
        'n_epochs': OPTIMAL_PARAMS['n_epochs'],
        'alpha': 0.5
    }
)


# =========================================
# Summary
# =========================================
print("\n" + "="*70)
print("RESULTS SUMMARY - MOVIELENS 1M")
print("="*70)

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Count wins
wins = sum(df_results['Difference'] < 0)
total = len(df_results)
print(f"\n🏆 You beat the paper on {wins}/{total} models!")

if wins == total:
    print("🎉 PERFECT SCORE! You beat the paper on ALL models!")
elif wins > total / 2:
    print("💪 Great job! You beat the paper on most models!")

print(f"\n💾 All results saved to '{RESULTS_FILE}'")
print(f"💾 Models saved in '{MODELS_DIR}/' folder")