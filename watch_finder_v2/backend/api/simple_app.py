"""
Simple Flask App for Railway Deployment Test
============================================

Minimal app to verify Railway deployment works before loading heavy ML dependencies.
"""

import os
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)

# Environment-specific CORS configuration
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
if ENVIRONMENT == 'production':
    # Production CORS - restrict to your Netlify domain
    NETLIFY_DOMAIN = os.getenv('NETLIFY_DOMAIN', 'your-netlify-app.netlify.app')
    CORS(app, origins=[
        f"https://{NETLIFY_DOMAIN}",
        f"https://www.{NETLIFY_DOMAIN}",
        "https://deploy-preview-*--your-netlify-app.netlify.app"
    ])
    print(f"🔒 Production CORS enabled for: {NETLIFY_DOMAIN}")
else:
    # Development CORS - allow all origins
    CORS(app)
    print("🔓 Development CORS enabled (all origins)")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'environment': ENVIRONMENT,
        'message': 'Simple Flask app running on Railway!'
    })

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify deployment."""
    return jsonify({
        'message': 'Railway deployment successful!',
        'environment': ENVIRONMENT,
        'python_version': os.sys.version,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        'message': 'Watch Finder Backend API',
        'status': 'running',
        'endpoints': ['/api/health', '/api/test']
    })

if __name__ == '__main__':
    print("🚀 Starting Simple Watch Finder API for Railway...")
    
    # Start the server
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🌐 Starting server on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug) 