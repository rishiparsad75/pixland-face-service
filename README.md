# PixLand Face Recognition Microservice

Python Flask service using **DeepFace (ArcFace model)** for production-grade face recognition.

## Why Python / ArcFace?
- **L2-normalized 512D embeddings** — no false positives like face-api.js had
- **Euclidean distance ≤ 0.68 = match** — reliable, industry-standard threshold
- **RetinaFace detector** — handles glasses, lighting, angles far better

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check, returns `{status, model, detector}` |
| POST | `/extract` | Extract face embedding from image |
| POST | `/compare` | Compare two embeddings, returns similarity score |

### POST /extract
```json
// Request (JSON)
{ "image": "<base64_string>" }

// OR multipart form with field "image"

// Response
{
  "embedding": [512 floats],
  "embedding_dim": 512,
  "face_count": 1,
  "face_area": { "x": 0, "y": 0, "w": 100, "h": 100 },
  "model": "ArcFace"
}
```

### POST /compare
```json
// Request
{ "embedding1": [...], "embedding2": [...] }

// Response
{ "similarity": 0.85, "distance": 0.15, "is_match": true, "threshold": 0.55 }
```

## Setup & Run

```bash
# Install dependencies (first time only)
cd server/face_service
pip install -r requirements.txt

# Run the service (port 5001)
python app.py
```

## Architecture

```
Browser (Selfie) → Node.js API (port 5000) → Python Flask (port 5001)
                                                    ↓
                                           DeepFace / ArcFace
                                                    ↓
                                          512D L2-normalized embedding
                                                    ↓
                                          Cosine Similarity Matching
```

## Production Deployment
Can be deployed as:
- **Azure App Service** (separate from Node.js)
- **Azure Container Instance** (Docker)
- **Same VM** as Node.js (simplest)

> **Note**: Always start this service before Node.js. Node.js will check `/health` at startup.
