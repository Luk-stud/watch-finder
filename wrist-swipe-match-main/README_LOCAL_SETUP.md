# Local Development Setup

## API Configuration

The frontend automatically detects and connects to the backend API. Here are the configuration options:

### Environment-Based Configuration

The app uses different configurations based on where it's running:

- **Production** (Netlify/Vercel): Uses Railway backend URL
- **Local Development**: Uses `localhost:5001` with fallbacks
- **Network Development**: Uses `localhost:5001` first, then tries network IPs

### Configuration Files

1. **Runtime Config** (`public/config.json`):
   ```json
   {
     "apiUrl": "http://localhost:5001/api",
     "fallbackUrls": [
       "http://localhost:5001/api",
       "http://127.0.0.1:5001/api",
       "http://192.168.50.5:5001/api"
     ]
   }
   ```

2. **Environment Variables**:
   - Create `.env.local` (if needed):
     ```
     VITE_API_URL=http://localhost:5001/api
     ```

### Backend Setup

1. Make sure your backend is running:
   ```bash
   cd backend
   python api/modern_app.py
   ```

2. Backend should show:
   ```
   * Running on http://127.0.0.1:5001
   * Running on http://192.168.50.5:5001
   ```

### Frontend Setup

1. Start the frontend:
   ```bash
   cd wrist-swipe-match-main
   npm install  # or bun install
   npm run dev  # or bun dev
   ```

2. Open browser and check console for connection logs:
   ```
   🔧 Environment Configuration: {...}
   🔧 API Service initialized: {...}
   🌐 Making API request to: http://localhost:5001/api/health
   ```

### Troubleshooting

If the frontend can't connect to the backend:

1. **Check Backend IP**: The backend log shows which IPs it's listening on
2. **Update Fallback URLs**: Edit `src/config/environment.ts` if your network IP is different
3. **Connection Testing**: The app will automatically test multiple URLs and show results in console
4. **Manual Override**: Set `VITE_API_URL` environment variable

### Network Development

When running on network IP (e.g., accessing via `192.168.0.209:8080`):

1. The app will try `localhost:5001` first (most reliable)
2. If that fails, it tests all fallback URLs
3. Updates configuration to use the working URL

### Console Debugging

Enable debug logging by ensuring you're in development mode:
- Check console for `🔧 Environment Configuration`
- Look for connection test results: `🔍 Testing X API endpoints...`
- Successful connections show: `✅ http://localhost:5001/api/health - XXXms`
- Failed connections show: `❌ http://X.X.X.X:5001/api/health - (error message)`

### Quick Fix

If nothing works, manually update the IP in `src/config/environment.ts`:

```typescript
const DEVELOPMENT_CONFIG: EnvironmentConfig = {
  API_BASE_URL: 'http://YOUR_BACKEND_IP:5001/api',
  // ... rest of config
};
```

Replace `YOUR_BACKEND_IP` with the IP shown in your backend startup logs. 