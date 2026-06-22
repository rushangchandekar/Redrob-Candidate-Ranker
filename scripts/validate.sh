#!/bin/bash
echo "=== Running submission validator ==="
python validate_submission.py submission.csv

echo ""
echo "=== Row count check ==="
# Get row count excluding header
ROW_COUNT=$(tail -n +2 submission.csv 2>/dev/null | wc -l)
# Strip whitespace
ROW_COUNT=$(echo $ROW_COUNT | tr -d '[:space:]')
echo "Rows in submission: $ROW_COUNT (expected: 100)"

if [ "$ROW_COUNT" -eq 100 ]; then
    echo "✅ Row count OK"
else
    echo "❌ Row count WRONG"
fi
