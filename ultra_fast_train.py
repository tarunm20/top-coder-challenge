#!/usr/bin/env python3
"""
Ultra-fast ML training for reimbursement prediction.
Optimized for speed with extensive feature engineering.
"""

import json
import numpy as np
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import time
import math

def create_extensive_features(days, miles, receipts):
    """Create extensive engineered features for maximum pattern capture"""
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
    print("Loading data...")
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Extract features and targets
    X = []
    y = []
    
    print("Creating extensive features...")
    for case in data:
        inp = case['input']
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']
        
        features = create_extensive_features(days, miles, receipts)
        X.append(features)
        y.append(case['expected_output'])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset shape: {X.shape} ({X.shape[1]} features per sample)")
    
    # Train on ALL data for maximum accuracy
    print("Training on ALL 1000 cases for maximum accuracy...")
    
    # Use DecisionTree optimized for speed and performance
    print("Training ultra-fast DecisionTree model with extensive features...")
    start_time = time.time()
    
    model = DecisionTreeRegressor(
        max_depth=15,     # Deeper for rich features
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42
    )
    
    model.fit(X, y)
    train_time = time.time() - start_time
    
    # Evaluate on training data
    train_pred = model.predict(X)
    train_mae = mean_absolute_error(y, train_pred)
    
    print(f"\nTraining completed in {train_time:.2f} seconds")
    print(f"Training MAE: ${train_mae:.2f}")
    
    # Test prediction speed
    print("\nTesting prediction speed...")
    start_time = time.time()
    for _ in range(1000):
        _ = model.predict(X[:1])
    speed_time = time.time() - start_time
    print(f"1000 predictions in {speed_time:.4f} seconds ({1000/speed_time:.0f} predictions/sec)")
    
    # Feature importance
    feature_importance = model.feature_importances_
    print(f"\nTop 10 most important features:")
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    # Save the model
    print("\nSaving model...")
    with open('ultra_fast_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model saved as ultra_fast_model.pkl")
    
    # Test a few predictions
    print("\nSample predictions on training data:")
    for i in range(10):
        actual = y[i]
        predicted = train_pred[i]
        error = abs(actual - predicted)
        print(f"  Actual: ${actual:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}")
    
    return model

if __name__ == "__main__":
    main()
