"""
Basic tests for the Watch Finder backend.
"""

import pytest
from api.linucb_app import app
from models.linucb_engine import LinUCBEngine
from models.simple_session import SimpleSession

@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'active_sessions' in data

def test_session_creation(client):
    """Test session creation endpoint."""
    response = client.post('/api/session')
    assert response.status_code == 200
    data = response.get_json()
    assert 'session_id' in data
    assert 'recommendations' in data

def test_simple_session():
    """Test SimpleSession class."""
    session = SimpleSession()
    assert session.id is not None
    assert len(session.liked_watches) == 0
    assert len(session.shown_watches) == 0

def test_linucb_engine():
    """Test LinUCBEngine initialization."""
    engine = LinUCBEngine(dim=20, alpha=0.25)
    assert engine.dim == 20
    assert engine.alpha == 0.25 