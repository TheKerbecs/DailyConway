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

@njit(parallel=True)
def step_grid_numba(grid):
    n = np.empty_like(grid)
    for i in prange(16):
        for j in range(16):
            i_prev = 15 if i == 0 else i - 1
            i_next = 0 if i == 15 else i + 1
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

def play_game_of_life_fast(B_str: str, public_salt: str) -> dict:
    """Play Conway's Game of Life on a 16x16 grid mapped from a 256-bit binary string."""
    salt_bytes = public_salt.encode("ascii")
    void_hash = _compute_void_hash(salt_bytes)
    grid = np.frombuffer(B_str.encode("ascii"), dtype=np.uint8) - 48
    grid = grid.reshape(16, 16).astype(np.int8)

    past_grids: dict = {}
    stat_min, stat_max = 999, 0
    freq = np.zeros((16, 16), dtype=np.int32)
    iteration = 0

    while True:
        grid_bytes = grid.tobytes()
        alive = int(grid.sum())

        if grid_bytes in past_grids:
            idx = past_grids[grid_bytes]
            delta = iteration - idx
            
            ascii_view = (grid.reshape(-1) + 48).astype(np.uint8).tobytes()
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
                "peak": int(freq.max()),
                "min": stat_min,
                "max": stat_max,
                "terminusHash": current_hash,
                "terminus": terminus,
            }

        past_grids[grid_bytes] = iteration
        if alive < stat_min:
            stat_min = alive
        if alive > stat_max:
            stat_max = alive
        freq += grid
        iteration += 1

        grid = step_grid_numba(grid)
