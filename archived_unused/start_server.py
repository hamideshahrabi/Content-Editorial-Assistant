import uvicorn
import sys
import os

def start_server(port=8000):
    """Start the FastAPI server on the specified port."""
    print(f"Starting server on port {port}...")
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    # Get port from command line argument or use default
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_server(port) 