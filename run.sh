#!/bin/bash

# Ultra-fast XGBoost reimbursement prediction
# Optimized for running 5k+ times with minimal overhead

python3 xgboost_predict.py "$1" "$2" "$3"
