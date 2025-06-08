#!/usr/bin/env python3
"""
Proper validation to detect overfitting by using train/validation split
"""

import json
import pickle
import math
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import time

def create_extensive_features(days, miles, receipts):
    """Create the ORIGINAL 112 features (before overfitting additions)"""
    features = []
    
    # === BASIC FEATURES ===
    features.append(days)
    features.append(miles)
    features.append(receipts)
    
    # === DERIVED RATIOS ===
    features.append(miles / days if days > 0 else 0)  # miles per day
    features.append(receipts / days if days > 0 else 0)  # receipts per day
    features.append(receipts / miles if miles > 0 else 0)  # receipts per mile
    features.append(miles / receipts if receipts > 0 else 0)  # miles per receipt dollar
    
    # === LOGARITHMIC FEATURES ===
    features.append(math.log(days + 1))
    features.append(math.log(miles + 1))
    features.append(math.log(receipts + 1))
    
    # === POLYNOMIAL FEATURES ===
    features.append(days ** 2)
    features.append(miles ** 2)
    features.append(receipts ** 2)
    features.append(days ** 0.5)
    features.append(miles ** 0.5)
    features.append(receipts ** 0.5)
    
    # === TRIP LENGTH CATEGORIES ===
    features.append(1 if days == 1 else 0)  # single day
    features.append(1 if days == 2 else 0)  # 2-day
    features.append(1 if days == 3 else 0)  # 3-day
    features.append(1 if days == 4 else 0)  # 4-day
    features.append(1 if days == 5 else 0)  # 5-day (sweet spot from interviews)
    features.append(1 if days == 6 else 0)  # 6-day
    features.append(1 if days == 7 else 0)  # 7-day
    features.append(1 if days >= 8 else 0)  # long trip (8+ days)
    features.append(1 if days >= 10 else 0)  # very long trip
    features.append(1 if days >= 12 else 0)  # extremely long trip
    
    # === MILEAGE TIERS ===
    features.append(1 if miles < 50 else 0)  # very short
    features.append(1 if 50 <= miles < 100 else 0)  # short
    features.append(1 if 100 <= miles < 200 else 0)  # medium-short
    features.append(1 if 200 <= miles < 300 else 0)  # medium
    features.append(1 if 300 <= miles < 500 else 0)  # medium-long
    features.append(1 if 500 <= miles < 800 else 0)  # long
    features.append(1 if miles >= 800 else 0)  # very long
    features.append(1 if miles >= 1000 else 0)  # extremely long
    
    # === RECEIPT TIERS (Critical from memories) ===
    features.append(1 if receipts < 50 else 0)  # very low
    features.append(1 if 50 <= receipts < 100 else 0)  # low
    features.append(1 if 100 <= receipts < 250 else 0)  # low-medium
    features.append(1 if 250 <= receipts < 500 else 0)  # medium
    features.append(1 if 500 <= receipts < 750 else 0)  # medium-high
    features.append(1 if 750 <= receipts < 1000 else 0)  # high
    features.append(1 if 1000 <= receipts < 1500 else 0)  # very high
    features.append(1 if 1500 <= receipts < 2000 else 0)  # extremely high
    features.append(1 if receipts >= 2000 else 0)  # penalty zone (from memories)
    features.append(1 if receipts >= 2500 else 0)  # severe penalty zone
    
    # === EFFICIENCY INDICATORS ===
    efficiency = miles / days if days > 0 else 0
    features.append(efficiency)
    features.append(1 if efficiency < 50 else 0)  # low efficiency
    features.append(1 if 50 <= efficiency < 100 else 0)  # medium efficiency
    features.append(1 if 100 <= efficiency < 200 else 0)  # good efficiency
    features.append(1 if efficiency >= 200 else 0)  # high efficiency (bonus from interviews)
    features.append(1 if efficiency >= 300 else 0)  # very high efficiency
    
    # === SPENDING PATTERNS ===
    daily_spend = receipts / days if days > 0 else 0
    features.append(daily_spend)
    features.append(1 if daily_spend < 50 else 0)  # frugal
    features.append(1 if 50 <= daily_spend < 100 else 0)  # moderate
    features.append(1 if 100 <= daily_spend < 200 else 0)  # high spending
    features.append(1 if daily_spend >= 200 else 0)  # very high spending
    
    # === INTERACTION FEATURES ===
    features.append(days * miles)  # trip intensity
    features.append(days * receipts)  # spending intensity
    features.append(miles * receipts)  # distance-spending interaction
    features.append(days * miles * receipts)  # three-way interaction
    
    # === COMPLEX RATIOS ===
    features.append((miles + receipts) / days if days > 0 else 0)  # combined intensity
    features.append(miles / (receipts + 1))  # mileage efficiency
    features.append(receipts / (miles + 1))  # spending per mile
    features.append((days * miles) / (receipts + 1))  # trip efficiency ratio
    
    # === SPECIAL PATTERNS FROM INTERVIEWS ===
    # "Magic numbers" and special cases mentioned in interviews
    features.append(1 if abs(receipts - 847) < 10 else 0)  # "lucky number" from Marcus
    features.append(1 if days == 5 and miles > 200 else 0)  # 5-day high-mileage bonus
    features.append(1 if days >= 8 and receipts >= 2000 else 0)  # double penalty case
    features.append(1 if days == 1 and miles < 100 else 0)  # single day short trip
    features.append(1 if efficiency > 250 and days <= 5 else 0)  # efficiency bonus case
    
    # === BINNED FEATURES ===
    # Create bins for continuous variables
    days_bin = min(int(days / 2), 7)  # 0-7 bins
    miles_bin = min(int(miles / 100), 12)  # 0-12 bins  
    receipts_bin = min(int(receipts / 200), 12)  # 0-12 bins
    
    for i in range(8):
        features.append(1 if days_bin == i else 0)
    for i in range(13):
        features.append(1 if miles_bin == i else 0)
    for i in range(13):
        features.append(1 if receipts_bin == i else 0)
    
    # === TRIGONOMETRIC FEATURES ===
    # Capture cyclical patterns
    features.append(math.sin(days * math.pi / 7))  # weekly cycle
    features.append(math.cos(days * math.pi / 7))
    features.append(math.sin(miles * math.pi / 500))  # mileage cycle
    features.append(math.cos(miles * math.pi / 500))
    
    # === PERCENTILE-BASED FEATURES ===
    # Based on data distribution (approximate percentiles)
    features.append(1 if days <= 3 else 0)  # bottom 25%
    features.append(1 if days >= 10 else 0)  # top 25%
    features.append(1 if miles <= 200 else 0)  # bottom 25%
    features.append(1 if miles >= 900 else 0)  # top 25%
    features.append(1 if receipts <= 500 else 0)  # bottom 25%
    features.append(1 if receipts >= 1800 else 0)  # top 25%
    
    return features

def main():
    print("🔍 Proper Validation: Detecting Overfitting")
    print("=" * 50)
    print()
    
    # Load data
    print("Loading data...")
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Prepare features and targets
    X = []
    y = []
    
    for case in data:
        inp = case['input']
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']
        target = case['expected_output']
        
        features = create_extensive_features(days, miles, receipts)
        X.append(features)
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset shape: {X.shape} ({X.shape[1]} features per sample)")
    print()
    
    # Split into train/validation (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print()
    
    # Test different model complexities
    complexities = [
        {"name": "Simple", "n_estimators": 50, "max_depth": 4, "learning_rate": 0.1},
        {"name": "Medium", "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        {"name": "Complex", "n_estimators": 500, "max_depth": 8, "learning_rate": 0.05},
        {"name": "Very Complex", "n_estimators": 1000, "max_depth": 12, "learning_rate": 0.03},
    ]
    
    results = []
    
    for config in complexities:
        print(f"Testing {config['name']} model...")
        
        model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predict
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate errors
        train_mae = np.mean(np.abs(train_pred - y_train))
        val_mae = np.mean(np.abs(val_pred - y_val))
        
        # Calculate exact matches
        train_exact = np.sum(np.abs(train_pred - y_train) < 0.01)
        val_exact = np.sum(np.abs(val_pred - y_val) < 0.01)
        
        overfitting_gap = val_mae - train_mae
        
        result = {
            "name": config['name'],
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_exact": train_exact,
            "val_exact": val_exact,
            "train_exact_pct": (train_exact / len(y_train)) * 100,
            "val_exact_pct": (val_exact / len(y_val)) * 100,
            "overfitting_gap": overfitting_gap,
            "train_time": train_time
        }
        results.append(result)
        
        print(f"  Train MAE: ${train_mae:.2f}")
        print(f"  Val MAE: ${val_mae:.2f}")
        print(f"  Overfitting gap: ${overfitting_gap:.2f}")
        print(f"  Train exact: {train_exact}/{len(y_train)} ({result['train_exact_pct']:.1f}%)")
        print(f"  Val exact: {val_exact}/{len(y_val)} ({result['val_exact_pct']:.1f}%)")
        print(f"  Training time: {train_time:.2f}s")
        print()
    
    # Summary
    print("📊 OVERFITTING ANALYSIS:")
    print("=" * 50)
    print(f"{'Model':<15} {'Train MAE':<10} {'Val MAE':<10} {'Gap':<8} {'Val Exact%':<12}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['name']:<15} ${r['train_mae']:<9.2f} ${r['val_mae']:<9.2f} ${r['overfitting_gap']:<7.2f} {r['val_exact_pct']:<11.1f}%")
    
    print()
    print("🎯 RECOMMENDATIONS:")
    
    # Find best validation performance
    best_val = min(results, key=lambda x: x['val_mae'])
    least_overfitting = min(results, key=lambda x: x['overfitting_gap'])
    
    print(f"• Best validation MAE: {best_val['name']} (${best_val['val_mae']:.2f})")
    print(f"• Least overfitting: {least_overfitting['name']} (gap: ${least_overfitting['overfitting_gap']:.2f})")
    
    if best_val['overfitting_gap'] > 5:
        print("⚠️  WARNING: Significant overfitting detected!")
        print("   Consider: simpler model, more regularization, or more training data")
    elif best_val['overfitting_gap'] > 2:
        print("⚠️  CAUTION: Moderate overfitting detected")
        print("   Model may not generalize well to private test cases")
    else:
        print("✅ Good generalization - low overfitting risk")

if __name__ == "__main__":
    main()
