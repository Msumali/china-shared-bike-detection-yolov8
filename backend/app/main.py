from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

from app.api.routes import router
from app.core.config import settings

# Create FastAPI app
app = FastAPI(
    title="Bike Detection API",
    description="API for detecting shared bikes (Didi, Meituan, HelloRide) in videos using YOLOv8",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="outputs"), name="static")

# Include API routes
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Bike Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs"
    }

# @app.get("/health")
# async def health_check():
#     """Health check endpoint."""
#     return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )