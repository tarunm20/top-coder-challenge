#!/usr/bin/env python3
"""
Ultra-fast batch results generation for private cases using robust model
"""

import json
import pickle
import math
import numpy as np
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
    print("🚀 Ultra-Fast Private Results Generation (ROBUST)")
    print("=" * 55)
    print()
    
    # Load robust model
    print("Loading robust XGBoost model...")
    start_time = time.time()
    try:
        with open('robust_model.pkl', 'rb') as f:
            model = pickle.load(f)
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.3f} seconds")
    except FileNotFoundError:
        print("ERROR: robust_model.pkl not found! Run robust_train.py first.")
        return
    
    # Load private cases
    print("Loading private test cases...")
    with open('private_cases.json', 'r') as f:
        private_cases = json.load(f)
    
    print(f"Processing {len(private_cases)} private cases...")
    print()
    
    # Prepare batch data
    start_time = time.time()
    X_batch = []
    
    for case in private_cases:
        days = case['trip_duration_days']
        miles = case['miles_traveled']
        receipts = case['total_receipts_amount']
        
        features = create_robust_features(days, miles, receipts)
        X_batch.append(features)
    
    X_batch = np.array(X_batch)
    
    feature_time = time.time() - start_time
    print(f"Feature engineering completed in {feature_time:.3f} seconds")
    
    # Batch prediction
    print("Running batch predictions...")
    start_time = time.time()
    predictions = model.predict(X_batch)
    prediction_time = time.time() - start_time
    
    predictions_per_sec = len(private_cases) / prediction_time
    print(f"Batch prediction completed in {prediction_time:.3f} seconds")
    print(f"Speed: {predictions_per_sec:.0f} predictions/sec")
    print()
    
    # Write results
    print("Writing results to private_results.txt...")
    with open('private_results.txt', 'w') as f:
        for prediction in predictions:
            f.write(f"{prediction:.2f}\n")
    
    total_time = load_time + feature_time + prediction_time
    
    print("✅ Results generation complete!")
    print()
    print(f"📊 Performance Summary:")
    print(f"  Cases processed: {len(private_cases)}")
    print(f"  Total time: {total_time:.3f} seconds")
    print(f"  Speed: {predictions_per_sec:.0f} predictions/sec")
    print(f"  Output file: private_results.txt")
    
    # Compare to shell script
    shell_time_estimate = len(private_cases) * 1.0  # Assume 1 second per call
    speedup = shell_time_estimate / total_time
    print(f"  Speed improvement: ~{speedup:.0f}x faster than individual calls")
    
    # Show sample predictions
    print()
    print("📋 Sample predictions:")
    for i in range(min(10, len(predictions))):
        case = private_cases[i]
        print(f"  Case {i+1}: {case['trip_duration_days']} days, {case['miles_traveled']} miles, ${case['total_receipts_amount']} → ${predictions[i]:.2f}")
    
    if len(predictions) > 10:
        print(f"  ... and {len(predictions) - 10} more cases")
    
    print()
    print("🎯 IMPORTANT: This uses the ROBUST model that should generalize well!")
    print("   Expected performance: realistic and stable on private test cases")
    print("   Ready for submission!")

if __name__ == "__main__":
    main()
