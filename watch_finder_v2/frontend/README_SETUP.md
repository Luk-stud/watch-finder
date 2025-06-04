# 🎯 WatchSwipe AI - Frontend Setup

This frontend has been updated to work with the modern AI-powered watch recommendation backend.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
npm install
```

### 2. Environment Setup
Create a `.env` file in the root directory:
```bash
VITE_API_URL=http://localhost:5001/api
```

### 3. Start the Backend
Ensure the modern backend is running on port 5001:
```bash
cd ../backend/api
python modern_app.py
```

### 4. Start the Frontend
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000` (or next available port).

## 🎯 Key Features

### ✨ Modern AI Integration
- **Smart Recommendations**: Uses advanced machine learning algorithms
- **Real-time Learning**: Adapts to your preferences as you swipe
- **Diversity Optimization**: Ensures varied recommendations
- **Confidence Scoring**: Shows AI confidence levels

### 🗃️ Rich Dataset
- **3,320+ Watches**: From luxury to everyday timepieces
- **Comprehensive Specs**: Movement, materials, complications
- **High-Quality Images**: Local and CDN image sources
- **Brand Information**: Direct links to brand websites

### 🎨 Enhanced UI
- **Algorithm Badges**: Shows which AI algorithm recommended each watch
- **Seed Indicators**: Identifies carefully selected seed watches
- **Real Image Display**: Shows actual watch photos
- **Detailed Specifications**: Comprehensive watch information
- **Series Exploration**: Browse watches from the same series

## 🔧 API Endpoints Used

### Session Management
- `POST /api/start-session` - Initialize recommendation session
- `POST /api/get-recommendations` - Get personalized recommendations
- `POST /api/add-feedback` - Record user feedback

### Data Exploration
- `GET /api/get-series` - Get watches from same series
- `GET /api/stats` - Backend statistics
- `GET /api/health` - Health check

## 🎛️ Configuration

### Backend Connection
The frontend automatically connects to:
- **Local Development**: `http://localhost:5001/api`
- **Production**: Set `VITE_API_URL` environment variable

### Image Sources
Watch images are loaded in priority order:
1. Local images (if available)
2. CDN URLs (fallback)
3. Generated placeholders (final fallback)

## 🔍 Debug Information

### AI Algorithm Insights
Each watch card shows:
- **Algorithm Used**: Which ML algorithm recommended it
- **Confidence Score**: AI confidence level (0.0-1.0)
- **Seed Status**: Whether it's a pre-selected seed watch

### Console Logging
Check browser console for:
- Session initialization details
- Recommendation algorithm insights
- User engagement metrics
- Performance statistics

## 🎪 User Flow

1. **Landing Page**: Introduction and start button
2. **AI Initialization**: Backend creates personalized session
3. **Smart Seeds**: Shows 7 carefully selected initial watches
4. **Adaptive Learning**: AI learns from your swipes
5. **Personalized Recommendations**: Gets more relevant over time
6. **Series Exploration**: Click series button to see related watches
7. **Detailed Specs**: Click info button for full specifications
8. **Liked Collection**: View all your favorited watches

## 🐛 Troubleshooting

### Backend Connection Issues
```bash
# Check if backend is running
curl http://localhost:5001/api/health

# Start backend if needed
cd ../backend/api && python modern_app.py
```

### TypeScript Errors
The codebase uses modern React + TypeScript. If you see import errors:
```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Environment Variables
If API URL is wrong, create `.env` file:
```bash
echo "VITE_API_URL=http://localhost:5001/api" > .env
```

## 📊 Data Structure

### Watch Object
```typescript
interface Watch {
  index: number;           // Unique identifier
  brand: string;          // Watch brand
  model: string;          // Model name
  specs: {               // Detailed specifications
    movement: string;
    case_material: string;
    diameter_mm: string;
    // ... 80+ additional fields
  };
  // AI enhancement fields
  algorithm?: string;     // ML algorithm used
  confidence?: number;    // AI confidence score
  is_seed?: boolean;     // Pre-selected seed watch
}
```

## 🚀 Development

### Project Structure
```
src/
├── components/         # React components
│   ├── WatchCard.tsx  # Main swipe card
│   ├── SpecsOverlay.tsx # Specifications modal
│   └── SeriesOverlay.tsx # Series browser
├── pages/             # Route pages
│   ├── Landing.tsx    # Welcome page
│   ├── Index.tsx      # Main swipe interface
│   └── LikedWatches.tsx # Favorites view
├── lib/               # Utilities
│   └── api.ts         # Backend API client
└── data/              # Data helpers
    └── watchData.ts   # Type definitions & utilities
```

### Adding New Features
1. Add API endpoints to `src/lib/api.ts`
2. Update types in `src/data/watchData.ts`
3. Create/modify components as needed
4. Update routing in `src/App.tsx`

## 🎯 Next Steps

### Planned Features
- [ ] User accounts and persistence
- [ ] Advanced filtering options
- [ ] Price range preferences
- [ ] Brand preference learning
- [ ] Export favorites list
- [ ] Social sharing features

### Performance Optimizations
- [ ] Image lazy loading
- [ ] Virtual scrolling for large lists
- [ ] Service worker for offline support
- [ ] Progressive web app features

---

**Ready to explore the world of watches with AI?** 🤖⌚

Start by running the backend, then launch this frontend, and let the AI learn your watch preferences! 