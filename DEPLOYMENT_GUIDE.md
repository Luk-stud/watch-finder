# Deployment Guide: Modern Watch Finder

This guide covers deploying the modern Watch Finder application with the new Vite + React frontend on Netlify and the modern Python backend on Railway.

## 🚀 Backend Deployment (Railway)

### Prerequisites
- Railway account
- Git repository access

### Updated Configuration Files
- ✅ `railway.json` - Updated to use `backend/api/modern_app.py`
- ✅ `Procfile` - Updated for modern backend
- ✅ `requirements.txt` - All dependencies included
- ✅ `railway.env` - Production environment settings

### Deployment Steps

1. **Push Updated Backend Files**
   ```bash
   git add .
   git commit -m "Update backend for Railway deployment"
   git push
   ```

2. **Deploy to Railway**
   - Railway will automatically detect changes and redeploy
   - The modern backend will start at `backend/api/modern_app.py`
   - Port will be automatically set by Railway

3. **Environment Variables** (Set in Railway Dashboard)
   ```
   FLASK_ENV=production
   USE_FAISS=false
   ENABLE_CORS=true
   LOG_LEVEL=INFO
   ```

4. **Verify Deployment**
   - Check Railway logs for successful startup
   - Test health endpoint: `https://your-railway-app.railway.app/api/health`

## 🌐 Frontend Deployment (Netlify)

### Prerequisites
- Netlify account
- Git repository access to `wrist-swipe-match-main/` folder

### Updated Configuration Files
- ✅ `wrist-swipe-match-main/netlify.toml` - Vite build configuration
- ✅ Environment variables for production API URL

### Deployment Steps

1. **Create New Netlify Site**
   - Connect to your Git repository
   - Set **Base directory** to: `wrist-swipe-match-main`
   - Set **Build command** to: `npm run build`
   - Set **Publish directory** to: `dist`

2. **Environment Variables** (Set in Netlify Dashboard)
   ```
   VITE_API_URL=https://your-railway-app.railway.app/api
   ```

3. **Build Settings**
   ```
   Base directory: wrist-swipe-match-main
   Build command: npm run build
   Publish directory: wrist-swipe-match-main/dist
   ```

4. **Deploy**
   - Netlify will automatically build and deploy
   - The React app will connect to your Railway backend

## 🔧 Key Changes Made

### Backend (Railway)
- **Updated entry point**: Now uses `backend/api/modern_app.py`
- **Port handling**: Automatically uses Railway's PORT environment variable
- **Production settings**: Debug mode disabled in production
- **CORS configuration**: Properly configured for Netlify frontend
- **Modern algorithms**: Advanced recommendation engine active
- **Enhanced error handling**: Production-ready logging and monitoring

### Frontend (Netlify)
- **Build system**: Updated for Vite (dist folder instead of out)
- **Environment**: Production API URL points to Railway backend
- **Routing**: SPA routing properly configured
- **API integration**: Uses modern backend endpoints
- **Performance**: Optimized caching headers

## 🎯 Current Architecture

```
Frontend (Netlify)           Backend (Railway)
┌─────────────────────┐      ┌──────────────────────┐
│ Vite + React + TS   │ ───► │ Flask + Modern ML    │
│ shadcn/ui           │      │ CLIP Embeddings     │
│ TailwindCSS         │      │ Advanced Clustering  │
│ Network IP: Auto    │      │ Real-time Learning   │
└─────────────────────┘      └──────────────────────┘
  wrist-swipe-match-main       backend/api/modern_app.py
```

## 📊 Features Active

### Advanced ML Backend
- ✅ CLIP image embeddings (1536D)
- ✅ Multi-cluster preference learning
- ✅ Exploration/exploitation balance
- ✅ Real-time session management
- ✅ Diversity optimization
- ✅ Performance analytics

### Modern Frontend
- ✅ AI-powered recommendations
- ✅ Real watch images with fallbacks
- ✅ Comprehensive specifications (80+ fields)
- ✅ Series exploration
- ✅ Algorithm transparency
- ✅ Responsive design

## 🔍 Monitoring & Debugging

### Backend Monitoring
- Health endpoint: `/api/health`
- Metrics endpoint: `/api/metrics`
- Diagnostics endpoint: `/api/diagnostics`

### Frontend Debugging
- Console logs show AI algorithm insights
- Network tab shows API communication
- Performance metrics available

## 🚨 Troubleshooting

### Common Issues
1. **CORS errors**: Check Railway CORS configuration
2. **Build failures**: Verify Node.js version (18)
3. **API connection**: Check environment variable URLs
4. **Memory issues**: Railway may need memory increase for large datasets

### Backend Logs
```bash
# Railway logs will show:
INFO:models.modern_recommendation_engine:🚀 Modern Recommendation Engine initialized
INFO:models.modern_session_manager:🚀 Modern Session Manager initialized
```

### Frontend Network Errors
- Check if API URL is correct in Netlify environment variables
- Verify Railway backend is running and accessible

## 🎉 Success Verification

✅ **Backend deployed**: Railway shows green status  
✅ **Frontend deployed**: Netlify build successful  
✅ **API connection**: Health check returns 200  
✅ **Recommendations**: AI generates watch suggestions  
✅ **Images loading**: Watch images display correctly  
✅ **Session management**: User preferences tracked  

The modern Watch Finder is now production-ready! 🚀 