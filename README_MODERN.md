# 🌟 Watch Finder - Modern AI Edition

A cutting-edge watch recommendation system featuring state-of-the-art machine learning algorithms, real-time preference learning, and advanced diversity optimization.

## 🚀 Features

### Modern Backend (v3.0.0)
- **Advanced ML Algorithms**: Multi-modal deep learning recommendations
- **FAISS Vector Search**: High-performance similarity search with 1536D embeddings
- **Smart Seed Selection**: Pre-computed diverse seed sets for optimal cold-start
- **Real-time Learning**: Dynamic preference clustering with transformer-based attention
- **Exploration/Exploitation**: Intelligent balance between personalization and discovery
- **Performance Monitoring**: Comprehensive metrics and diagnostics
- **JSON-Safe**: Robust serialization handling for production deployment

### Frontend Features
- **Modern UI**: React 18 + Next.js with Framer Motion animations
- **Responsive Design**: Mobile-first with gesture support
- **Real-time Feedback**: Instant recommendations based on user interactions
- **Series Discovery**: Explore watch collections and variants
- **Debug Insights**: See recommendation algorithms in action

## 📊 Dataset

- **3,320 Watches**: Comprehensive dataset from `final_scrape.csv`
- **Local Images**: 2,632 high-quality local images with CDN fallbacks
- **Rich Metadata**: 80+ specification fields per watch
- **Pre-normalized Embeddings**: 1536D vectors optimized for similarity search

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │ WatchCard   │ │ Navigation  │ │ Modern API Client  │ │
│  └─────────────┘ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│                Modern Backend API                      │
│  ┌─────────────────┐ ┌─────────────────────────────────┐ │
│  │ Flask Routes    │ │ Modern Session Manager         │ │
│  │ • /start-session│ │ • Async session handling       │ │
│  │ • /get-recs     │ │ • Real-time synchronization    │ │
│  │ • /health       │ │ • Performance tracking         │ │
│  │ • /diagnostics  │ └─────────────────────────────────┘ │
│  └─────────────────┘                                   │
└─────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│            Modern Recommendation Engine                │
│  ┌─────────────────┐ ┌─────────────────────────────────┐ │
│  │ FAISS Index     │ │ Preference Clustering           │ │
│  │ • HNSW search   │ │ • User profile learning         │ │
│  │ • 1536D vectors │ │ • Dynamic adaptation           │ │
│  └─────────────────┘ └─────────────────────────────────┘ │
│  ┌─────────────────┐ ┌─────────────────────────────────┐ │
│  │ Smart Seeds     │ │ Diversity Optimization         │ │
│  │ • Pre-computed  │ │ • Multi-objective balancing    │ │
│  │ • 20 seed sets  │ │ • Exploration strategies       │ │
│  └─────────────────┘ └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Easy Setup (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd watch_finder

# Run the one-command setup
./start_watch_finder.sh
```

This will:
1. Start the modern backend on `http://localhost:5001`
2. Start the frontend on `http://localhost:3000`
3. Automatically install dependencies if needed
4. Show you all the available endpoints

### Manual Setup

#### Backend
```bash
# Activate virtual environment
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Start modern backend
cd backend/api
python modern_app.py
```

#### Frontend
```bash
# Install dependencies
cd frontend-nextjs
npm install

# Start development server
npm run dev
```

## 📡 API Endpoints

### Core Endpoints
- `GET /api/health` - System health and metrics
- `GET /api/diagnostics` - Detailed system diagnostics
- `POST /api/start-session` - Initialize new recommendation session
- `POST /api/get-recommendations` - Get personalized recommendations
- `GET /api/stats` - Dataset and performance statistics

### Example API Usage

#### Start Session
```bash
curl -X POST http://localhost:5001/api/start-session \
  -H "Content-Type: application/json" \
  -d '{"num_seeds": 7}'
```

#### Get Recommendations
```bash
curl -X POST http://localhost:5001/api/get-recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your-session-id",
    "liked_indices": [273, 1421],
    "disliked_indices": [944],
    "current_candidates": [2380, 944, 273, 1913],
    "num_recommendations": 7,
    "exploration_factor": 0.3,
    "diversity_threshold": 0.7
  }'
```

## 🧠 Recommendation Algorithms

### Cold Start
- **Smart Seeds**: Pre-computed diverse selections using clustering
- **Style Exploration**: Representatives from different watch categories
- **Fallback Logic**: Robust handling when seeds unavailable

### Preference-Based
- **Preference Clustering**: Dynamic user preference modeling
- **Centroid Search**: FAISS-powered similarity search
- **Confidence Weighting**: Time-decayed preference strength

### Exploration
- **Style Clusters**: Discover unexplored watch categories
- **Diversity Optimization**: Balanced recommendations across brands/styles
- **Adaptive Exploration**: Context-aware discovery strategies

### Hybrid
- **Multi-Strategy**: Combines preference-based (70%) + exploration (30%)
- **Dynamic Balancing**: Adapts to user engagement level
- **Seamless Integration**: Unified experience across algorithms

## 🔧 Configuration

### Backend Settings
```python
# Modern Backend Configuration
USE_FAISS = True  # Enable high-performance search
SMART_SEEDS_PATH = "backend/data/precomputed_smart_seed_sets.pkl"
EMBEDDINGS_PATH = "backend/data/watch_embeddings.pkl"
METADATA_PATH = "backend/data/watch_metadata.pkl"
```

### Frontend Settings
```typescript
// Frontend Configuration
const CONFIG = {
  API_BASE_URL: 'http://localhost:5001/api',
  SWIPE_THRESHOLD: 50,
  DRAG_THRESHOLD: 100,
  ANIMATION_DURATION: 300,
  MAX_RETRIES: 3
};
```

## 📊 Performance Metrics

The system tracks comprehensive metrics:

- **Response Time**: Average API response time
- **Cache Hit Rate**: Similarity search optimization
- **User Satisfaction**: Engagement and interaction quality
- **Diversity Score**: Recommendation variety measurement
- **Exploration Rate**: User discovery progress

## 🐛 Debug Features

### Frontend Debug Badge
Each watch card shows:
- Algorithm used (`smart_seeds`, `preference_cluster`, `exploration`)
- Confidence score (0.0-1.0)
- Seed indicators for cold-start recommendations

### Backend Logging
```bash
# View detailed logs
tail -f backend/logs/recommendation_engine.log

# Key log patterns
🤖 Algorithm used: preference_based
🎯 Diversity score: 0.84
👤 User engagement: highly_engaged
💡 Exploration suggestions: Try dive watches, Explore luxury timepieces
```

## 🚀 Production Deployment

### Railway Deployment
The modern backend is optimized for Railway:
- FAISS-optional design with numpy fallback
- JSON serialization safeguards
- Environment-based configuration
- Health check endpoints

### Environment Variables
```bash
# Production settings
FAISS_ENABLED=false  # Disable FAISS on Railway if needed
API_PORT=5001
DEBUG_MODE=false
```

## 🔍 Troubleshooting

### Common Issues

1. **FAISS Installation**
   ```bash
   # Mac with Apple Silicon
   pip install faiss-cpu
   
   # If FAISS fails, the system automatically falls back to numpy
   ```

2. **JSON Serialization Errors**
   - The modern backend includes comprehensive numpy type conversion
   - All responses are JSON-safe by design

3. **Frontend Connection Issues**
   - Ensure backend is running on port 5001
   - Check CORS settings in modern_app.py

### Health Checks
```bash
# Check backend status
curl http://localhost:5001/api/health

# Check diagnostics
curl http://localhost:5001/api/diagnostics

# Check stats
curl http://localhost:5001/api/stats
```

## 📈 Future Enhancements

- **Graph Neural Networks**: Advanced collaborative filtering
- **Multi-modal Features**: Image + text embeddings
- **Real-time Analytics**: Live recommendation performance dashboard
- **A/B Testing**: Algorithm performance comparison
- **Mobile App**: Native iOS/Android applications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both frontend and backend
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using cutting-edge AI and modern web technologies** 