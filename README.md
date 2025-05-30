# Watch Finder - AI-Powered Visual Watch Discovery

A modern web application that uses AI visual similarity to help you discover watches based on your preferences. Powered by custom CLIP embeddings and intelligent recommendation algorithms.

## 🔥 Features

- **AI-Powered Discovery**: Visual similarity matching using 512-dimensional CLIP embeddings
- **Intuitive Swiping**: Tinder-style interface with touch/swipe gestures
- **Smart Recommendations**: Beam search algorithm with feedback weighting
- **Variant Detection**: Groups similar watch models automatically
- **Mobile-First Design**: Responsive, accessible UI with modern design
- **Comprehensive Database**: 4,994+ watches with detailed specifications

## 🏗️ Architecture

### Frontend
- **Vanilla JavaScript ES6+** - Modern modular architecture
- **Mobile-first CSS** - Responsive design with CSS custom properties
- **Accessibility Ready** - Screen reader support, keyboard navigation
- **Progressive Enhancement** - Works without JavaScript for basic functionality

### Backend
- **Flask API** - Lightweight Python backend
- **Precomputed Embeddings** - No ML model inference required
- **Memory Efficient** - ~50MB deployment size
- **Fast Response Times** - Direct embedding lookups

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Modern web browser

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/watch-finder.git
cd watch-finder
```

2. **Set up backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python api/app.py
```

3. **Serve frontend**
```bash
cd frontend
python -m http.server 8080
```

4. **Open your browser**
Navigate to `http://localhost:8080`

## 📂 Project Structure

```
watch-finder/
├── backend/
│   ├── api/
│   │   └── app.py              # Flask API server
│   ├── data/
│   │   ├── watch_embeddings.pkl  # Precomputed CLIP embeddings
│   │   ├── watch_metadata.pkl    # Watch specifications
│   │   └── variant_groups.pkl    # Variant clustering data
│   ├── requirements.txt        # Python dependencies
│   ├── Procfile               # Railway deployment config
│   └── railway.json           # Railway deployment settings
├── frontend/
│   ├── css/
│   │   └── styles.css         # Modern responsive styles
│   ├── js/
│   │   └── app.js             # Modular ES6+ application
│   ├── index.html             # Main application
│   └── debug.html             # Debug/testing page
├── watch_data/                # Raw watch data (JSON files)
└── README.md
```

## 🎯 API Endpoints

- `GET /api/health` - Health check
- `POST /api/start-session` - Initialize discovery session
- `POST /api/get-recommendations` - Get watch recommendations
- `POST /api/add-feedback` - Submit like/dislike feedback
- `GET /api/get-variants` - Get watch variants
- `GET /api/stats` - Database statistics

## 🛠️ Technical Details

### Embedding System
- **Custom CLIP Model**: Trained on watch images
- **512-dimensional vectors**: Semantic visual representation
- **Cosine similarity**: For recommendation matching
- **Beam search**: Intelligent exploration vs exploitation

### Performance Optimizations
- **Precomputed embeddings**: No real-time ML inference
- **Efficient similarity search**: Optimized vector operations
- **Minimal dependencies**: Lightweight deployment
- **Memory efficient**: Direct pickle loading

## 🚀 Deployment

### Frontend (Netlify)
1. Connect your GitHub repo to Netlify
2. Set build directory to `frontend`
3. Deploy automatically on push

### Backend (Railway.app)
1. Connect your GitHub repo to Railway
2. Set root directory to `backend`
3. Railway will detect Flask app automatically

Environment variables needed:
- `FLASK_ENV=production`
- `PORT=5000` (Railway sets this automatically)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Watch data sourced from various online retailers
- CLIP model architecture from OpenAI
- Modern UI/UX patterns from contemporary web design

## 📊 Stats

- **4,994 watches** in database
- **741 variant groups** detected
- **59.6% diversity improvement** through variant grouping
- **512-dimensional** CLIP embeddings
- **<50MB** deployment size 