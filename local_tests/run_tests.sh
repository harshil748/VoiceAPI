#!/bin/bash
# Quick script to run local API tests

echo "Starting Local API Test"
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment not activated!"
    echo "Run: source tts/bin/activate"
    exit 1
fi

# Check if server is running
if ! curl -s http://localhost:8000/docs > /dev/null; then
    echo "API server is not running on localhost:8000"
    echo ""
    echo "Start the server first:"
    echo "python -m src.cli serve --port 8000"
    exit 1
fi

echo "API server is running"
echo ""

# Run tests
python local_tests/test_local_api.py

echo ""
echo "Tests complete!"
echo "Check outputs: local_tests/outputs/"