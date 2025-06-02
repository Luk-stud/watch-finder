#!/bin/bash

echo "🚀 Starting Watch Finder - Modern Backend + Frontend"
echo "=================================================="

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Function to kill background processes on script exit
cleanup() {
    echo "🛑 Stopping services..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the modern backend
echo "🔧 Starting Modern Backend API..."
cd backend/api
python modern_app.py &
BACKEND_PID=$!
cd ../..

# Wait a moment for backend to start
sleep 3

# Check if backend is running
if curl -s http://localhost:5001/api/health > /dev/null; then
    echo "✅ Backend is running on http://localhost:5001"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start the frontend
echo "🎨 Starting Frontend..."
cd frontend-nextjs

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "🎉 Watch Finder is ready!"
echo "================================"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend:  http://localhost:5001"
echo "🔍 Health:   http://localhost:5001/api/health"
echo "📊 Stats:    http://localhost:5001/api/stats"
echo ""
echo "💡 Features:"
echo "  • Modern AI recommendation engine"
echo "  • FAISS vector similarity search"
echo "  • Smart seed selections"
echo "  • Real-time preference learning"
echo "  • Advanced diversity optimization"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for all background jobs
wait 