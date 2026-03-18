import pytest
import os
import json
from app import create_app, db
from app.models import User, Role

@pytest.fixture
def app():
    # Setup a testing app with in-memory SQLite and testing flag
    app = create_app()
    app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "WTF_CSRF_ENABLED": False
    })

    with app.app_context():
        db.create_all()
        
        # Seed test roles
        admin_role = Role(name='Admin')
        db.session.add(admin_role)
        
        # Seed test admin user
        admin = User(username='testadmin', email='admin@test.com', role=admin_role)
        admin.set_password('password')
        db.session.add(admin)
        
        db.session.commit()
    
    yield app
    
    with app.app_context():
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

def login(client, username, password):
    return client.post('/auth/login', data=dict(
        username=username,
        password=password
    ), follow_redirects=True)

def test_api_fl_status_unauthorized(client):
    """Test that unauthorized users get redirected (or 401ed) from API."""
    response = client.get('/api/fl/status')
    # Because of Flask-Login, usually it redirects to login (302) or 401
    assert response.status_code in [302, 401]

def test_api_users_admin_only(client):
    """Test that an Admin can access the users list."""
    # Login as admin
    login(client, 'testadmin', 'password')
    
    response = client.get('/api/users')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert len(data) == 1
    assert data[0]['username'] == 'testadmin'

def test_api_fl_start_round(client):
    """Test starting an FL round as an admin."""
    login(client, 'testadmin', 'password')
    
    response = client.post('/api/fl/start_round')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'started' in data.get('message', '').lower()
