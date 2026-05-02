import datetime
import hashlib
import json
import urllib.error
import urllib.request
import numpy as np
from numba import njit, prange

_VOID_GRID_ASCII = b"0" * 256

def _compute_void_hash(salt_bytes: bytes) -> str:
    """Compute the SHA-256 of an all-dead 16x16 grid combined with the given salt."""
    return hashlib.sha256(_VOID_GRID_ASCII + salt_bytes).hexdigest()

def fetch_nist_pulse():
    """Fetch the latest NIST beacon pulse for today's date."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    midnight_ms = int(datetime.datetime(today.year, today.month, today.day, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    url = f"https://beacon.nist.gov/beacon/2.0/pulse/time/{midnight_ms}"
    print(f"Fetching NIST pulse: {url}")
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read())
    pulse = data["pulse"]
    return pulse["uri"], pulse["outputValue"]

def get_target_strings(suite_choice: str) -> list[str]:
    """
    Generate target hash suffixes based on the user's selected difficulty.

    Args:
        suite_choice (str): The complexity of the challenge (e.g. '4', '6', '8', or 'any').

    Returns:
        list[str]: A list of target hexadecimal strings to match against.
    """
    today = datetime.datetime.now(datetime.timezone.utc).date()
    date_str = today.strftime("%Y%m%d")  # e.g. "20260501"
    if suite_choice == "any":
        return [date_str[-8:], date_str[-6:], date_str[-4:]]
    return [date_str[-int(suite_choice):]]

def b_str_from_bytes(b32: bytes) -> str:
    """Reproduce format(int.from_bytes(b32, 'big'), '0256b')."""
    return ''.join(f'{byte:08b}' for byte in b32)

def verify_hash(b_str: str, salt: str, expected_hex: str) -> bool:
    """Verify that a given binary string + salt produces the expected SHA-256 hash."""
    h = hashlib.sha256((b_str + salt).encode('ascii')).hexdigest()
    return h == expected_hex

@njit(cache=True, nogil=True)
def step_grid_numba(grid):
    n = np.empty_like(grid)
    for i in range(16):
        i_prev = 15 if i == 0 else i - 1
        i_next = 0 if i == 15 else i + 1
        for j in range(16):
            j_prev = 15 if j == 0 else j - 1
            j_next = 0 if j == 15 else j + 1

            neighbors = (grid[i_prev, j_prev] + grid[i_prev, j] + grid[i_prev, j_next] +
                         grid[i, j_prev]                       + grid[i, j_next] +
                         grid[i_next, j_prev] + grid[i_next, j] + grid[i_next, j_next])

            alive = grid[i, j]
            if alive:
                n[i, j] = 1 if (neighbors == 2 or neighbors == 3) else 0
            else:
                n[i, j] = 1 if neighbors == 3 else 0
    return n


@njit(cache=True, nogil=True)
def _play_gol_core(grid):
    """
    Run the full Game of Life loop inside Numba.

    Returns: (iterations, peak, stat_min, stat_max, delta, final_grid)
    where delta is the cycle length (1=STATIC, 2/3=FLICKER, 64=GLIDER, else UNKNOWN/VOID).
    """
    MAX_STATES = 16384
    # Store visited states as 4 uint64s per state (256 bits = 16x16).
    history = np.zeros((MAX_STATES, 4), dtype=np.uint64)
    freq = np.zeros((16, 16), dtype=np.int32)
    stat_min = 999
    stat_max = 0
    iteration = 0
    delta = 0

    while iteration < MAX_STATES:
        # Pack grid into 4 uint64s (4 rows of 16 bits each = 64 bits per word).
        w0 = np.uint64(0); w1 = np.uint64(0); w2 = np.uint64(0); w3 = np.uint64(0)
        for i in range(4):
            r = np.uint64(0)
            for j in range(16):
                r = (r << np.uint64(1)) | np.uint64(grid[i, j])
            w0 = (w0 << np.uint64(16)) | r
        for i in range(4, 8):
            r = np.uint64(0)
            for j in range(16):
                r = (r << np.uint64(1)) | np.uint64(grid[i, j])
            w1 = (w1 << np.uint64(16)) | r
        for i in range(8, 12):
            r = np.uint64(0)
            for j in range(16):
                r = (r << np.uint64(1)) | np.uint64(grid[i, j])
            w2 = (w2 << np.uint64(16)) | r
        for i in range(12, 16):
            r = np.uint64(0)
            for j in range(16):
                r = (r << np.uint64(1)) | np.uint64(grid[i, j])
            w3 = (w3 << np.uint64(16)) | r

        alive = 0
        for i in range(16):
            for j in range(16):
                alive += grid[i, j]

        # Check for cycle.
        for k in range(iteration):
            if (history[k, 0] == w0 and history[k, 1] == w1 and
                history[k, 2] == w2 and history[k, 3] == w3):
                delta = iteration - k
                return iteration, int(freq.max()), stat_min, stat_max, delta, grid

        history[iteration, 0] = w0
        history[iteration, 1] = w1
        history[iteration, 2] = w2
        history[iteration, 3] = w3

        if alive < stat_min:
            stat_min = alive
        if alive > stat_max:
            stat_max = alive
        for i in range(16):
            for j in range(16):
                freq[i, j] += grid[i, j]

        iteration += 1
        grid = step_grid_numba(grid)

    return iteration, int(freq.max()), stat_min, stat_max, 0, grid


@njit(cache=True, nogil=True, parallel=True)
def _play_gol_batch_core(grids, out_iters, out_peaks, out_mins, out_maxs, out_deltas, out_final):
    """
    Run _play_gol_core on a batch of grids in parallel.

    grids: (N, 16, 16) int8
    out_*: (N,) preallocated output arrays
    out_final: (N, 16, 16) int8 preallocated final-grid buffer
    """
    n = grids.shape[0]
    for k in prange(n):
        g = grids[k].copy()
        it, peak, smin, smax, delta, final_g = _play_gol_core(g)
        out_iters[k] = it
        out_peaks[k] = peak
        out_mins[k] = smin
        out_maxs[k] = smax
        out_deltas[k] = delta
        for i in range(16):
            for j in range(16):
                out_final[k, i, j] = final_g[i, j]


def play_game_of_life_batch(B_strs, public_salt: str):
    """
    Evaluate many boards in a single Numba call.

    B_strs: list[str] of 256-char binary strings.
    Returns: list[dict] with same shape as play_game_of_life_fast.
    """
    salt_bytes = public_salt.encode("ascii")
    void_hash = _compute_void_hash(salt_bytes)
    n = len(B_strs)
    if n == 0:
        return []

    grids = np.empty((n, 16, 16), dtype=np.int8)
    for k, b in enumerate(B_strs):
        arr = np.frombuffer(b.encode("ascii"), dtype=np.uint8).astype(np.int8) - 48
        grids[k] = arr.reshape(16, 16)

    out_iters = np.zeros(n, dtype=np.int64)
    out_peaks = np.zeros(n, dtype=np.int64)
    out_mins = np.zeros(n, dtype=np.int64)
    out_maxs = np.zeros(n, dtype=np.int64)
    out_deltas = np.zeros(n, dtype=np.int64)
    out_final = np.empty((n, 16, 16), dtype=np.int8)

    _play_gol_batch_core(grids, out_iters, out_peaks, out_mins, out_maxs, out_deltas, out_final)

    results = []
    for k in range(n):
        ascii_view = (out_final[k].reshape(-1) + 48).astype(np.uint8).tobytes()
        current_hash = hashlib.sha256(ascii_view + salt_bytes).hexdigest()
        delta = int(out_deltas[k])
        if delta == 1:
            terminus = "STATIC"
        elif delta == 2:
            terminus = "2-FLICKER"
        elif delta == 3:
            terminus = "3-FLICKER"
        elif delta == 64:
            terminus = "GLIDER"
        elif current_hash == void_hash:
            terminus = "VOID"
        else:
            terminus = "UNKNOWN"
        smin = int(out_mins[k])
        results.append({
            "iterations": int(out_iters[k]),
            "peak": int(out_peaks[k]),
            "min": smin if smin != 999 else 0,
            "max": int(out_maxs[k]),
            "terminusHash": current_hash,
            "terminus": terminus,
        })
    return results


def play_game_of_life_fast(B_str: str, public_salt: str) -> dict:
    """Play Conway's Game of Life on a 16x16 grid mapped from a 256-bit binary string."""
    salt_bytes = public_salt.encode("ascii")
    void_hash = _compute_void_hash(salt_bytes)
    grid = np.frombuffer(B_str.encode("ascii"), dtype=np.uint8) - 48
    grid = grid.reshape(16, 16).astype(np.int8)

    iteration, peak, stat_min, stat_max, delta, final_grid = _play_gol_core(grid)

    ascii_view = (final_grid.reshape(-1) + 48).astype(np.uint8).tobytes()
    current_hash = hashlib.sha256(ascii_view + salt_bytes).hexdigest()

    if delta == 1:
        terminus = "STATIC"
    elif delta == 2:
        terminus = "2-FLICKER"
    elif delta == 3:
        terminus = "3-FLICKER"
    elif delta == 64:
        terminus = "GLIDER"
    elif current_hash == void_hash:
        terminus = "VOID"
    else:
        terminus = "UNKNOWN"

    return {
        "iterations": iteration,
        "peak": peak,
        "min": stat_min if stat_min != 999 else 0,
        "max": stat_max,
        "terminusHash": current_hash,
        "terminus": terminus,
    }
