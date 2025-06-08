#!/usr/bin/env python3
"""
Robust training with proper validation and regularization to prevent overfitting
"""

import json
import pickle
import math
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
import time

def create_robust_features(days, miles, receipts):
    """Create robust features that generalize well (no overfitting)"""
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
    
    # === POLYNOMIAL FEATURES (limited) ===
    features.append(days ** 2)
    features.append(miles ** 2)
    features.append(receipts ** 2)
    features.append(math.sqrt(days))
    features.append(math.sqrt(miles))
    features.append(math.sqrt(receipts))
    
    # === BROAD CATEGORICAL FEATURES ===
    # Trip length categories (broad ranges)
    features.append(1 if days <= 2 else 0)  # short trip
    features.append(1 if 3 <= days <= 5 else 0)  # medium trip
    features.append(1 if 6 <= days <= 8 else 0)  # long trip
    features.append(1 if days >= 9 else 0)  # very long trip
    
    # Mileage tiers (broad ranges)
    features.append(1 if miles < 200 else 0)  # short distance
    features.append(1 if 200 <= miles < 500 else 0)  # medium distance
    features.append(1 if 500 <= miles < 800 else 0)  # long distance
    features.append(1 if miles >= 800 else 0)  # very long distance
    
    # Receipt tiers (broad ranges, focusing on penalty zones)
    features.append(1 if receipts < 500 else 0)  # low spending
    features.append(1 if 500 <= receipts < 1000 else 0)  # medium spending
    features.append(1 if 1000 <= receipts < 1500 else 0)  # high spending
    features.append(1 if 1500 <= receipts < 2000 else 0)  # very high spending
    features.append(1 if receipts >= 2000 else 0)  # penalty zone
    
    # === EFFICIENCY INDICATORS ===
    efficiency = miles / days if days > 0 else 0
    features.append(efficiency)
    features.append(1 if efficiency < 100 else 0)  # low efficiency
    features.append(1 if 100 <= efficiency < 200 else 0)  # medium efficiency
    features.append(1 if efficiency >= 200 else 0)  # high efficiency
    
    # === SPENDING PATTERNS ===
    daily_spend = receipts / days if days > 0 else 0
    features.append(daily_spend)
    features.append(1 if daily_spend < 100 else 0)  # frugal
    features.append(1 if 100 <= daily_spend < 200 else 0)  # moderate
    features.append(1 if daily_spend >= 200 else 0)  # high spending
    
    # === INTERACTION FEATURES (limited) ===
    features.append(days * miles / 1000)  # normalized trip intensity
    features.append(days * receipts / 1000)  # normalized spending intensity
    features.append(miles * receipts / 10000)  # normalized distance-spending
    
    # === ROBUST RATIOS ===
    features.append((miles + receipts) / days if days > 0 else 0)  # combined intensity
    features.append(miles / (receipts + 100))  # mileage efficiency (smoothed)
    features.append(receipts / (miles + 100))  # spending per mile (smoothed)
    
    # === BUSINESS LOGIC FEATURES ===
    # Based on interview insights but generalized
    features.append(1 if days >= 5 and efficiency > 150 else 0)  # efficiency bonus
    features.append(1 if receipts >= 2000 and days >= 7 else 0)  # double penalty
    features.append(1 if days == 1 and miles < 150 else 0)  # single day trips
    features.append(1 if miles > 0 and receipts / miles > 2 else 0)  # high cost per mile
    
    return features

def main():
    print("🛡️  Robust Model Training (Anti-Overfitting)")
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
        
        features = create_robust_features(days, miles, receipts)
        X.append(features)
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset shape: {X.shape} ({X.shape[1]} robust features)")
    print()
    
    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different regularization levels
    configs = [
        {"name": "High Regularization", "n_estimators": 100, "max_depth": 4, "learning_rate": 0.1, "reg_alpha": 1.0, "reg_lambda": 1.0},
        {"name": "Medium Regularization", "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "reg_alpha": 0.5, "reg_lambda": 0.5},
        {"name": "Low Regularization", "n_estimators": 300, "max_depth": 8, "learning_rate": 0.05, "reg_alpha": 0.1, "reg_lambda": 0.1},
    ]
    
    best_model = None
    best_val_mae = float('inf')
    best_config = None
    
    for config in configs:
        print(f"Testing {config['name']}...")
        
        model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=config['reg_alpha'],
            reg_lambda=config['reg_lambda'],
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        # Calculate errors
        train_mae = np.mean(np.abs(train_pred - y_train))
        val_mae = np.mean(np.abs(val_pred - y_val))
        overfitting_gap = val_mae - train_mae
        
        print(f"  Train MAE: ${train_mae:.2f}")
        print(f"  Val MAE: ${val_mae:.2f}")
        print(f"  Overfitting gap: ${overfitting_gap:.2f}")
        
        # Track best model based on validation performance
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model
            best_config = config
        
        print()
    
    print(f"🏆 Best model: {best_config['name']} (Val MAE: ${best_val_mae:.2f})")
    print()
    
    # Cross-validation on best model
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"Cross-validation MAE: ${cv_mae:.2f} ± ${cv_std:.2f}")
    print()
    
    # Retrain on full dataset with best config
    print("Retraining on full dataset...")
    final_model = xgb.XGBRegressor(
        n_estimators=best_config['n_estimators'],
        max_depth=best_config['max_depth'],
        learning_rate=best_config['learning_rate'],
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=best_config['reg_alpha'],
        reg_lambda=best_config['reg_lambda'],
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    start_time = time.time()
    final_model.fit(X, y)
    train_time = time.time() - start_time
    
    # Final predictions
    final_pred = final_model.predict(X)
    final_mae = np.mean(np.abs(final_pred - y))
    exact_matches = np.sum(np.abs(final_pred - y) < 0.01)
    close_matches = np.sum(np.abs(final_pred - y) < 1.0)
    
    print(f"Final training MAE: ${final_mae:.2f}")
    print(f"Exact matches: {exact_matches}/1000 ({exact_matches/10:.1f}%)")
    print(f"Close matches: {close_matches}/1000 ({close_matches/10:.1f}%)")
    print(f"Training time: {train_time:.2f}s")
    print()
    
    # Feature importance
    print("Top 10 most important features:")
    feature_importance = final_model.feature_importances_
    top_features = sorted(enumerate(feature_importance), key=lambda x: x[1], reverse=True)[:10]
    for i, (idx, importance) in enumerate(top_features, 1):
        print(f"  {i}. Feature {idx}: {importance:.4f}")
    print()
    
    # Save model
    with open('robust_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    print("Robust model saved as robust_model.pkl")
    
    # Test prediction speed
    print("\nTesting prediction speed...")
    start_time = time.time()
    for _ in range(1000):
        _ = final_model.predict(X[:1])
    speed_time = time.time() - start_time
    predictions_per_sec = 1000 / speed_time
    
    print(f"Prediction speed: {predictions_per_sec:.0f} predictions/sec")
    
    print("\n🎯 SUMMARY:")
    print(f"• Robust features: {X.shape[1]}")
    print(f"• Expected validation MAE: ${cv_mae:.2f}")
    print(f"• Training MAE: ${final_mae:.2f}")
    print(f"• Should generalize well to private test cases")
    print(f"• Prediction speed: {predictions_per_sec:.0f}/sec")

if __name__ == "__main__":
    main()
