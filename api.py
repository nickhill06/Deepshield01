import sys
import os
import uuid
import shutil
import sqlite3
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# add src/ to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from predict_video import predict_video

# ── CREATE FASTAPI APP ────────────────────────────────────────────
app = FastAPI(
    title="DeepShield API",
    description="Deepfake Video Detection API",
    version="1.0.0"
)

# ── CORS MIDDLEWARE ───────────────────────────────────────────────
# This allows our React frontend to talk to this server
# Without this, browser will block the requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow all origins (frontend URL)
    allow_methods=["*"],       # allow GET, POST, etc.
    allow_headers=["*"],       # allow all headers
)

# ── SERVE STATIC FILES ────────────────────────────────────────────
# This makes our output images accessible via URL
os.makedirs("outputs/gradcam", exist_ok=True)
os.makedirs("outputs/plots",   exist_ok=True)
os.makedirs("uploads",         exist_ok=True)

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ── DATABASE SETUP ────────────────────────────────────────────────
def init_db():
    """
    Creates SQLite database with predictions table
    SQLite = simple file-based database, no setup needed
    """
    conn = sqlite3.connect("deepshield.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            TEXT PRIMARY KEY,
            filename      TEXT,
            verdict       TEXT,
            confidence    REAL,
            fake_percent  REAL,
            heatmap_url   TEXT,
            graph_url     TEXT,
            created_at    TEXT
        )
    """)

    conn.commit()
    conn.close()

# initialize database when server starts
init_db()


def save_prediction(id, filename, verdict, confidence, fake_percent, heatmap_url, graph_url):
    """Save prediction result to database"""
    conn = sqlite3.connect("deepshield.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        id, filename, verdict, confidence,
        fake_percent, heatmap_url, graph_url,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_all_predictions():
    """Get all predictions from database"""
    conn = sqlite3.connect("deepshield.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()

    # convert to list of dicts
    results = []
    for row in rows:
        results.append({
            "id":           row[0],
            "filename":     row[1],
            "verdict":      row[2],
            "confidence":   row[3],
            "fake_percent": row[4],
            "heatmap_url":  row[5],
            "graph_url":    row[6],
            "created_at":   row[7]
        })

    return results


# ── API ENDPOINTS ─────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — confirms server is running"""
    return {
        "message": "DeepShield API is running!",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /api/predict",
            "history": "GET /api/history",
            "docs":    "GET /docs"
        }
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main endpoint — receives video, runs detection, returns result

    UploadFile = the video file sent from frontend
    File(...)  = required field
    """

    # ── VALIDATE FILE ─────────────────────────────────────────────
    allowed_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    # ── SAVE UPLOADED VIDEO ───────────────────────────────────────
    # generate unique ID for this prediction
    prediction_id = str(uuid.uuid4())[:8]

    # save video to uploads/ folder
    upload_path = f"uploads/{prediction_id}{file_ext}"

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"\n📹 Received video: {file.filename}")
    print(f"   Saved to: {upload_path}")
    print(f"   Prediction ID: {prediction_id}")

    # ── RUN DEEPFAKE DETECTION ────────────────────────────────────
    try:
        verdict, fake_percent = predict_video(
            video_path=upload_path,
            model_path="outputs/models/deepshield.pth"
        )

        confidence = fake_percent if verdict == "FAKE" else (100 - fake_percent)

        # paths to output images
        heatmap_url = f"/outputs/gradcam/result.png"
        graph_url   = f"/outputs/gradcam/video_analysis.png"

        # save to database
        save_prediction(
            id=prediction_id,
            filename=file.filename,
            verdict=verdict,
            confidence=round(confidence, 2),
            fake_percent=round(fake_percent, 2),
            heatmap_url=heatmap_url,
            graph_url=graph_url
        )

        # clean up uploaded video
        os.remove(upload_path)

        # return result to frontend
        return JSONResponse({
            "id":           prediction_id,
            "filename":     file.filename,
            "verdict":      verdict,
            "confidence":   round(confidence, 2),
            "fake_percent": round(fake_percent, 2),
            "heatmap_url":  heatmap_url,
            "graph_url":    graph_url,
            "message":      f"Video analyzed successfully"
        })

    except Exception as e:
        # clean up on error
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
def history():
    """Returns all past predictions from database"""
    predictions = get_all_predictions()
    return {
        "total": len(predictions),
        "predictions": predictions
    }


@app.get("/api/history/{prediction_id}")
def get_prediction(prediction_id: str):
    """Returns one specific prediction by ID"""
    conn = sqlite3.connect("deepshield.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM predictions WHERE id = ?",
        (prediction_id,)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return {
        "id":           row[0],
        "filename":     row[1],
        "verdict":      row[2],
        "confidence":   row[3],
        "fake_percent": row[4],
        "heatmap_url":  row[5],
        "graph_url":    row[6],
        "created_at":   row[7]
    }


# ── RUN SERVER ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting DeepShield API server...")
    print("   API docs: http://localhost:8000/docs")
    print("   API URL:  http://localhost:8000")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True    # auto-restart when code changes
    )