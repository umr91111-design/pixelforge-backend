
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PixelForge AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://playful-squirrel-c30122.netlify.app"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "name": "PixelForge AI API",
        "status": "running",
        "message": "Backend is connected and ready for development."
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "pixelforge-api"}
