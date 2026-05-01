# DailyConway

A hybrid GPU/CPU Conway's Game of Life cryptocurrency-style hash miner written in Python, featuring a bespoke hardware-accelerated CUDA kernel via `CuPy` and JIT compiled CPU simulation paths via `Numba`. Ships with a completely interactive, real-time metrics Web UI using FastAPI and WebSockets.

## Project Structure
```text
DailyConway/
├── app/               # Main application backend package
│   ├── __init__.py
│   ├── logic.py       # Domain logic: NIST pulse, GoL CPU routines, Numba grids
│   ├── miner.py       # Hardware Orchestration: CuPy CUDA kernels and worker pool spawn
│   └── website.py     # Web UI server: FastAPI HTTP routes and WebSocket metrics pub/sub
├── tools/             # Utility scripts
│   └── json_to_rle.py # Format conversion utility 
├── main.py            # Primary configurable entry point 
├── pyproject.toml
└── README.md
```

## Installation

This project utilizes `uv` for lightning-fast package management and uses the standalone `nvidia` package distribution alongside `CuPy` to dynamically locate DLLs on Windows without needing a manual CUDA toolkit installation.

### Steps to Install

1.  Clone the repository and `cd` into the project directory.
2.  Run `uv sync` to install all dependencies (`cupy-cuda11x`, `numba`, etc.) directly from the locked sources.

```powershell
uv sync
```

## Configuration

To change the "Owner" string registered against winning hashes, you can directly update the `OWNER = "Bobinou"` constant at the top of `main.py`.

## Usage
To start the FastAPI server and the embedded real-time Web UI, run:

```powershell
uv run python main.py
```
This will launch the application, and you can access the dashboard at `http://127.0.0.1:5001/`.

### Web Dashboard Features
Once the dashboard is open, you can:
- **Control Mining:** Start and stop the mining workers dynamically using the UI controls.
- **Monitor Performance:** View real-time graphs displaying your current hash rate (H/s) and overall miner performance.
- **Track Discoveries:** See a live history table of "winning" hashes/grids as your hardware finds them.
- **Inspect Patterns:** Click the "Inspect" button next to any discovered hash to open a modal and view the raw Run Length Encoded (RLE) pattern of the Game of Life grid.

### Mining Configuration Parameters
The web interface exposes several hardware-level tuning parameters so you can maximize your GPU's efficiency:
- **Suite:** Selects the target complexity based on the current date. Choosing `4`, `6`, or `8` specifies how many trailing digits of today's date must be matched in the generated hash. `any` accepts matches of any of those lengths.
- **Blocks:** The number of CUDA blocks scheduled per kernel launch. Increasing this spreads the workload across more of the GPU's Streaming Multiprocessors (SMs).
- **Threads:** The number of CUDA threads executed within each block. 
- **Iters/thread (ipt):** *Iterations Per Thread*. This determines how many sequential Game of Life hashes each individual thread will evaluate internally before the kernel finishes and returns results to the CPU. A higher value reduces CPU-to-GPU overhead but can cause display unresponsiveness if set too high.

