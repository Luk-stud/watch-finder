#!/bin/bash

echo "🎯 Starting WatchSwipe AI Frontend"
echo "=================================="

# Get the local network IP
NETWORK_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
if [ -z "$NETWORK_IP" ]; then
    NETWORK_IP="localhost"
fi

echo "🌐 Detected network IP: $NETWORK_IP"

# Check if .env file exists, create if needed
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    echo "VITE_API_URL=http://$NETWORK_IP:5001/api" > .env
    echo "✅ Environment configured for: http://$NETWORK_IP:5001/api"
else
    echo "📄 Using existing .env file"
    cat .env
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if backend is running on localhost
echo "🔍 Checking backend connection..."
if curl -s http://localhost:5001/api/health > /dev/null; then
    echo "✅ Backend is running on localhost:5001"
elif curl -s http://$NETWORK_IP:5001/api/health > /dev/null; then
    echo "✅ Backend is running on $NETWORK_IP:5001"
else
    echo "⚠️  Backend not detected. Make sure to start the backend first:"
    echo "   cd ../backend/api && python modern_app.py"
    echo ""
    echo "   The backend should be accessible at:"
    echo "   - http://localhost:5001/api/health"
    echo "   - http://$NETWORK_IP:5001/api/health"
    echo ""
    echo "   Press any key to continue anyway..."
    read -n 1
fi

echo ""
echo "🚀 Starting frontend development server..."
echo "   Frontend will auto-detect and connect to backend"
echo "   Backend API: http://$NETWORK_IP:5001/api"
echo "   CORS is configured for network access"
echo ""

# Start the development server
npm run dev 