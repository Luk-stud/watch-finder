# Watch Finder v2

A modern watch recommendation system using LinUCB (Linear Upper Confidence Bound) algorithm for personalized watch suggestions.

## Project Structure

```
watch_finder_v2/
├── backend/
│   ├── api/           # Flask API endpoints
│   ├── models/        # LinUCB and session management
│   ├── data/          # Watch database and features
│   ├── tests/         # Unit and integration tests
│   ├── utils/         # Helper functions and utilities
│   └── requirements.txt
├── frontend/          # React frontend (to be implemented)
└── docs/             # Project documentation
```

## Backend Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Start the development server:
```bash
python -m flask run --port 5001
```

## API Endpoints

- `POST /api/session` - Create new session and get initial recommendations
- `GET /api/recommendations` - Get next batch of recommendations
- `POST /api/feedback` - Record like/dislike actions
- `GET /api/liked-watches` - Retrieve liked watches
- `GET /api/watch/<id>/details` - Get watch details

## Frontend Setup

(To be implemented)

## Development

- Backend uses Flask with Python 3.8+
- LinUCB implementation for contextual bandits
- Simple session management
- RESTful API design
- Comprehensive error handling

## Testing

```bash
cd backend
pytest
```

## License

MIT 