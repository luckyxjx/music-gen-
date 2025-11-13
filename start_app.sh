#!/bin/bash

# Script to start both backend and frontend

echo "=========================================="
echo "EMOPIA Music Generator - Full Stack Setup"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16 or higher."
    exit 1
fi

echo "✓ Python3 found: $(python3 --version)"
echo "✓ Node.js found: $(node --version)"
echo ""

# Start backend in background
echo "Starting backend API server..."
python3 api.py &
BACKEND_PID=$!
echo "✓ Backend started (PID: $BACKEND_PID)"
echo ""

# Wait for backend to initialize
echo "Waiting for backend to initialize (15 seconds)..."
sleep 15

# Start frontend
echo "Starting frontend development server..."
cd client
npm run dev

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
