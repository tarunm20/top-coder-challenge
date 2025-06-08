import json
import numpy as np

# Load the data
with open('public_cases.json', 'r') as f:
    data = json.load(f)

print(f'Total cases: {len(data)}')
print('\nFirst 5 cases:')
for i in range(5):
    case = data[i]
    inp = case['input']
    out = case['expected_output']
    print(f'  Days: {inp["trip_duration_days"]}, Miles: {inp["miles_traveled"]}, Receipts: ${inp["total_receipts_amount"]:.2f} -> ${out:.2f}')

# Basic statistics
days = [case['input']['trip_duration_days'] for case in data]
miles = [case['input']['miles_traveled'] for case in data]
receipts = [case['input']['total_receipts_amount'] for case in data]
outputs = [case['expected_output'] for case in data]

print(f'\nStatistics:')
print(f'Days: min={min(days)}, max={max(days)}, avg={np.mean(days):.1f}')
print(f'Miles: min={min(miles)}, max={max(miles)}, avg={np.mean(miles):.1f}')
print(f'Receipts: min=${min(receipts):.2f}, max=${max(receipts):.2f}, avg=${np.mean(receipts):.2f}')
print(f'Outputs: min=${min(outputs):.2f}, max=${max(outputs):.2f}, avg=${np.mean(outputs):.2f}')
