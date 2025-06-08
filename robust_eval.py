#!/usr/bin/env python3
"""
Ultra-fast batch evaluation using the robust (non-overfitted) model
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
    print(" Ultra-Fast Reimbursement System Evaluation (ROBUST)")
    print("=" * 60)
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
    
    # Load test cases
    print("Loading public test cases...")
    with open('public_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    print(f" Running evaluation against {len(test_cases)} test cases...")
    print()
    
    # Prepare batch data
    start_time = time.time()
    X_batch = []
    expected_outputs = []
    
    for case in test_cases:
        inp = case['input']
        days = inp['trip_duration_days']
        miles = inp['miles_traveled']
        receipts = inp['total_receipts_amount']
        expected = case['expected_output']
        
        features = create_robust_features(days, miles, receipts)
        X_batch.append(features)
        expected_outputs.append(expected)
    
    X_batch = np.array(X_batch)
    expected_outputs = np.array(expected_outputs)
    
    feature_time = time.time() - start_time
    print(f"Feature engineering completed in {feature_time:.3f} seconds")
    
    # Batch prediction
    print("Running batch predictions...")
    start_time = time.time()
    predictions = model.predict(X_batch)
    prediction_time = time.time() - start_time
    
    predictions_per_sec = len(test_cases) / prediction_time
    print(f"Batch prediction completed in {prediction_time:.3f} seconds")
    print(f"Speed: {predictions_per_sec:.0f} predictions/sec")
    print()
    
    # Calculate metrics
    print("Calculating evaluation metrics...")
    errors = np.abs(predictions - expected_outputs)
    
    exact_matches = np.sum(errors <= 0.01)
    close_matches = np.sum(errors <= 1.0)
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Calculate score (sum of errors)
    score = np.sum(errors)
    
    print(" Evaluation Complete!")
    print()
    print(" Results Summary:")
    print(f"  Total test cases: {len(test_cases)}")
    print(f"  Successful runs: {len(test_cases)}")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_matches/len(test_cases)*100:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_matches/len(test_cases)*100:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    print()
    print(f" Your Score: {score:.2f} (lower is better)")
    print()
    
    # Performance assessment
    if score < 100:
        print(" Excellent! Very close to perfect solution.")
    elif score < 500:
        print(" Good performance! Room for improvement.")
    elif score < 1000:
        print(" Moderate performance. Consider more feature engineering.")
    else:
        print(" Needs improvement. Review model and features.")
    
    print()
    print(" Tips for improvement:")
    
    # Show worst cases
    worst_indices = np.argsort(errors)[-5:]
    print("  Check these high-error cases:")
    for idx in worst_indices:
        case = test_cases[idx]
        inp = case['input']
        print(f"    Case {idx}: {inp['trip_duration_days']} days, {inp['miles_traveled']} miles, ${inp['total_receipts_amount']} receipts")
        print(f"      Expected: ${expected_outputs[idx]:.2f}, Got: ${predictions[idx]:.2f}, Error: ${errors[idx]:.2f}")
    
    total_time = load_time + feature_time + prediction_time
    print()
    print(f" Total evaluation time: {total_time:.3f} seconds")
    
    # Compare to shell script
    shell_time_estimate = len(test_cases) * 1.0  # Assume 1 second per call
    speedup = shell_time_estimate / total_time
    print(f"   Speed improvement: ~{speedup:.0f}x faster than calling run.sh {len(test_cases)} times")
    
    print()
    print(" Next steps:")
    print("  1. This is a ROBUST model that should generalize well")
    print("  2. Expected performance on private cases: similar to this")
    print("  3. Consider ensemble methods for marginal improvements")
    print("  4. Submit when ready!")

if __name__ == "__main__":
    main()
