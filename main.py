import os
import uvicorn
from app.website import app

OWNER = "Bobinou"
os.environ["MINER_OWNER"] = OWNER

def main():
    print(f"Starting Conway GPU Miner for {OWNER} on http://127.0.0.1:5001")
    uvicorn.run("app.website:app", host="127.0.0.1", port=5001, log_level="warning")

if __name__ == "__main__":
    main()
