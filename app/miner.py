from __future__ import annotations
import asyncio
import concurrent.futures
import multiprocessing
import os
import sys
import time
import uuid
import numpy as np

# --- DLL discovery for pip-installed CUDA libs (must run before cupy import) ---
def _add_nvidia_dll_dirs():
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if spec is None or not spec.submodule_search_locations:
            return
        for root in spec.submodule_search_locations:
            for dirpath, _, filenames in os.walk(root):
                if any(f.lower().endswith(".dll") for f in filenames):
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(dirpath)
                        except OSError:
                            pass
                    os.environ["PATH"] = dirpath + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
_add_nvidia_dll_dirs()

import cupy as cp
from .logic import play_game_of_life_fast, b_str_from_bytes, fetch_nist_pulse, get_target_strings

KERNEL_SRC = r"""
typedef unsigned char       uint8_t;
typedef unsigned int        uint32_t;
typedef unsigned long long  uint64_t;

extern "C" {

__device__ __constant__ uint32_t K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

__device__ void sha256_block(uint32_t state[8], const uint8_t* block) {
    uint32_t w[64];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i*4] << 24) | ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) | (uint32_t)block[i*4+3];
    }
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
    state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
}

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }
__device__ __forceinline__ uint64_t xoro_next(uint64_t* s0, uint64_t* s1) {
    uint64_t a = *s0, b = *s1;
    uint64_t result = a + b;
    b ^= a;
    *s0 = rotl64(a, 24) ^ b ^ (b << 16);
    *s1 = rotl64(b, 37);
    return result;
}

__global__ void mine(
    const uint8_t* __restrict__ salt, int salt_len,
    const uint8_t* __restrict__ targets, const int* __restrict__ target_lens, int n_targets,
    uint64_t base_seed, int iters_per_thread,
    unsigned int* result_count, uint8_t* result_data, int max_results
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t s0 = base_seed ^ ((uint64_t)tid * 0x9E3779B97F4A7C15ULL + 1ULL);
    uint64_t s1 = (base_seed + 0xDEADBEEFCAFEBABEULL) ^ ((uint64_t)tid * 0xBF58476D1CE4E5B9ULL);
    if ((s0 | s1) == 0) s1 = 1;

    uint8_t msg[512];
    for (int i = 0; i < salt_len; i++) msg[256 + i] = salt[i];
    int total_len = 256 + salt_len;
    int padded_size = ((total_len + 9 + 63) / 64) * 64;
    msg[total_len] = 0x80;
    for (int i = total_len + 1; i < padded_size - 8; i++) msg[i] = 0;
    uint64_t bit_len = (uint64_t)total_len * 8ULL;
    msg[padded_size-8] = (uint8_t)(bit_len >> 56);
    msg[padded_size-7] = (uint8_t)(bit_len >> 48);
    msg[padded_size-6] = (uint8_t)(bit_len >> 40);
    msg[padded_size-5] = (uint8_t)(bit_len >> 32);
    msg[padded_size-4] = (uint8_t)(bit_len >> 24);
    msg[padded_size-3] = (uint8_t)(bit_len >> 16);
    msg[padded_size-2] = (uint8_t)(bit_len >> 8);
    msg[padded_size-1] = (uint8_t)(bit_len);

    const char hexchars[16] = {'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};

    for (int it = 0; it < iters_per_thread; it++) {
        uint64_t r0 = xoro_next(&s0, &s1);
        uint64_t r1 = xoro_next(&s0, &s1);
        uint64_t r2 = xoro_next(&s0, &s1);
        uint64_t r3 = xoro_next(&s0, &s1);

        #pragma unroll
        for (int i = 0; i < 64; i++) msg[i]      = '0' + (uint8_t)((r0 >> (63 - i)) & 1ULL);
        #pragma unroll
        for (int i = 0; i < 64; i++) msg[64 + i] = '0' + (uint8_t)((r1 >> (63 - i)) & 1ULL);
        #pragma unroll
        for (int i = 0; i < 64; i++) msg[128+ i] = '0' + (uint8_t)((r2 >> (63 - i)) & 1ULL);
        #pragma unroll
        for (int i = 0; i < 64; i++) msg[192+ i] = '0' + (uint8_t)((r3 >> (63 - i)) & 1ULL);

        uint32_t state[8] = {
            0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
            0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
        };
        for (int b = 0; b < padded_size; b += 64) sha256_block(state, msg + b);

        uint8_t hex[64];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint32_t v = state[i];
            hex[i*8+0] = hexchars[(v >> 28) & 0xfu];
            hex[i*8+1] = hexchars[(v >> 24) & 0xfu];
            hex[i*8+2] = hexchars[(v >> 20) & 0xfu];
            hex[i*8+3] = hexchars[(v >> 16) & 0xfu];
            hex[i*8+4] = hexchars[(v >> 12) & 0xfu];
            hex[i*8+5] = hexchars[(v >> 8 ) & 0xfu];
            hex[i*8+6] = hexchars[(v >> 4 ) & 0xfu];
            hex[i*8+7] = hexchars[ v        & 0xfu];
        }

        for (int t = 0; t < n_targets; t++) {
            int tlen = target_lens[t];
            const uint8_t* tgt = targets + t * 16;
            int last = 64 - tlen;
            for (int j = 0; j <= last; j++) {
                bool ok = true;
                #pragma unroll 8
                for (int k = 0; k < tlen; k++) {
                    if (hex[j+k] != tgt[k]) { ok = false; break; }
                }
                if (ok) {
                    unsigned int slot = atomicAdd(result_count, 1u);
                    if (slot < (unsigned int)max_results) {
                        uint8_t* out = result_data + (size_t)slot * 96;
                        #pragma unroll
                        for (int i = 0; i < 8; i++) out[i]    = (uint8_t)(r0 >> (56 - i*8));
                        #pragma unroll
                        for (int i = 0; i < 8; i++) out[8+i]  = (uint8_t)(r1 >> (56 - i*8));
                        #pragma unroll
                        for (int i = 0; i < 8; i++) out[16+i] = (uint8_t)(r2 >> (56 - i*8));
                        #pragma unroll
                        for (int i = 0; i < 8; i++) out[24+i] = (uint8_t)(r3 >> (56 - i*8));
                        #pragma unroll
                        for (int i = 0; i < 64; i++) out[32+i] = hex[i];
                    }
                    goto next_iter;
                }
            }
        }
        next_iter:;
    }
}

}  // extern C
"""

class GPUMiner:
    """
    Manages the GPU-accelerated mining process for Conway's Game of Life hashes.

    This class coordinates the execution of the custom CUDA kernel, manages memory
    transfers between the host and device, and delegates the evaluation of active
    Game of Life grids to a pool of CPU worker processes.
    """

    def __init__(self, callbacks):
        """
        Initialize the GPUMiner.

        Args:
            callbacks (dict): A dictionary of async callback functions. Expected keys are
                'on_log', 'on_rate', 'on_hit_count', 'on_match_history', and 'on_stop'.
        """
        self.callbacks = callbacks # dict with 'on_log', 'on_rate', 'on_hit_count', 'on_match_history', 'on_stop'
        self.is_running = False
        self._kernel_module = None
        self._kernel_fn = None
        self.max_results = 16384
    
    async def _log(self, msg):
        """
        Broadcast a log message using the configured 'on_log' callback.

        Args:
            msg (str): The log message to broadcast.
        """
        if 'on_log' in self.callbacks:
            await self.callbacks['on_log'](msg)

    def _compile(self):
        """
        Compile the CUDA kernel from source.
        """
        if self._kernel_fn is None:
            self._kernel_module = cp.RawModule(code=KERNEL_SRC, options=("-std=c++14",))
            self._kernel_fn = self._kernel_module.get_function("mine")

    async def _gol_worker(self, public_salt: str, pulse_uri: str, hit_queue: asyncio.Queue, process_pool: concurrent.futures.ProcessPoolExecutor):
        """
        Consume hashes from the queue and evaluate them using Conway's Game of Life.

        Args:
            public_salt (str): The daily salt used for hashing.
            pulse_uri (str): The URI of the NIST beacon pulse.
            hit_queue (asyncio.Queue): The queue providing raw GPU hits.
            process_pool (concurrent.futures.ProcessPoolExecutor): The pool execution context.
        """
        loop = asyncio.get_running_loop()
        while True:
            try:
                item = await hit_queue.get()
                if item is None:
                    return
                b_str, gpu_hex, hit_target, hit_idx, attempts_at_hit = item
                gol = await loop.run_in_executor(process_pool, play_game_of_life_fast, b_str, public_salt)
                match_id = uuid.uuid4().hex
                payload = {
                    "owner": os.environ.get("MINER_OWNER", "Unknown"),
                    "nist_pulse_id": pulse_uri,
                    "salt_value": public_salt,
                    "origin_hash": gpu_hex,
                    "suite": str(len(hit_target)),
                    "iterations": gol["iterations"],
                    "peak": gol["peak"],
                    "hash_index": hit_idx,
                    "attempts": int(1),
                    "bin": b_str,
                    "terminus_hash": gol["terminusHash"],
                }
                if 'on_match_history' in self.callbacks:
                    await self.callbacks['on_match_history'](match_id, gpu_hex, gol["iterations"], gol["peak"], payload)
            except asyncio.CancelledError:
                return
            except Exception as e:
                await self._log(f"GoL failed: {e}")

    async def run(self, target_suite: str, blocks: int, threads: int, ipt: int, n_workers: int):
        """
        Orchestrate the continuous mining loop.

        Args:
            target_suite (str): Selected target challenge string complexity (e.g. '4', '6', '8', or 'any').
            blocks (int): The number of CUDA blocks.
            threads (int): The number of threads per block.
            ipt (int): Iterations per thread (IPT). Hashes evaluated per thread.
            n_workers (int): Number of CPU workers for the process pool.
        
        Raises:
            asyncio.CancelledError: When the overarching async task is canceled.
            Exception: If an unhandled error occurs during GPU launch or CPU processing.
        """
        self.is_running = True
        hit_queue: asyncio.Queue = asyncio.Queue(maxsize=50000)
        worker_tasks = []
        process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
        
        try:
            await self._log("Fetching NIST pulse from logic module...")
            pulse_uri, salt = await asyncio.to_thread(fetch_nist_pulse)
            await self._log(f"Salt: {salt[:24]}... (len {len(salt)})")

            targets = get_target_strings(target_suite)
            await self._log(f"Targets: {targets}")

            await self._log("Compiling CUDA kernel...")
            await asyncio.to_thread(self._compile)

            salt_bytes = salt.encode("ascii")
            d_salt = cp.asarray(np.frombuffer(salt_bytes, dtype=np.uint8))

            target_arr = np.zeros((len(targets), 16), dtype=np.uint8)
            target_lens = np.zeros(len(targets), dtype=np.int32)
            for i, t in enumerate(targets):
                tb = t.encode("ascii")
                target_arr[i, : len(tb)] = np.frombuffer(tb, dtype=np.uint8)
                target_lens[i] = len(tb)
            d_targets = cp.asarray(target_arr.flatten())
            d_target_lens = cp.asarray(target_lens)

            d_result_count = cp.zeros(1, dtype=cp.uint32)
            d_result_data = cp.zeros(self.max_results * 96, dtype=cp.uint8)

            hashes_per_launch = blocks * threads * ipt
            await self._log(f"Grid: {blocks} x {threads} x {ipt} = {hashes_per_launch:,} hashes/launch")

            await self._log(f"Spawning {n_workers} concurrent CPU processes for Game of Life evaluations...")
            worker_tasks = [
                asyncio.create_task(self._gol_worker(salt, pulse_uri, hit_queue, process_pool))
                for _ in range(n_workers * 2)
            ]

            base_seed = int.from_bytes(os.urandom(8), "big")
            total_hashes = 0
            total_hits = 0
            dropped = 0
            seen_b = set()
            last_rate_t = time.time()
            last_rate_hashes = 0
            last_rate_hits = 0
            last_hit_broadcast = 0

            salt_len_arg = np.int32(len(salt_bytes))
            n_targets_arg = np.int32(len(targets))
            max_res_arg = np.int32(self.max_results)
            ipt_arg = np.int32(ipt)
            blocks_t = (blocks,)
            threads_t = (threads,)

            while self.is_running:
                if hit_queue.qsize() >= 49000:
                    await self._log(f"Queue full ({hit_queue.qsize()}), pausing GPU...")
                    while self.is_running and hit_queue.qsize() > 1000:
                        await asyncio.sleep(0.5)
                        now = time.time()
                        dt = now - last_rate_t
                        if dt >= 0.5:
                            hr = (total_hits - last_rate_hits) / dt
                            last_rate_t = now
                            last_rate_hits = total_hits
                            if 'on_rate' in self.callbacks:
                                await self.callbacks['on_rate'](0.0, hr, hit_queue.qsize(), dropped)
                            print(f"[rate] 0.0 M/s  total={total_hashes:,}  hits={total_hits}  q={hit_queue.qsize()}  dropped={dropped}")
                    if self.is_running:
                        await self._log("Queue drained, resuming GPU...")
                    else:
                        break

                d_result_count[:] = 0
                seed_arg = np.uint64(base_seed)
                base_seed = (base_seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF

                def _launch():
                    self._kernel_fn(
                        blocks_t, threads_t,
                        (d_salt, salt_len_arg,
                         d_targets, d_target_lens, n_targets_arg,
                         seed_arg, ipt_arg,
                         d_result_count, d_result_data, max_res_arg),
                    )
                    cp.cuda.Stream.null.synchronize()
                    n = int(d_result_count.get()[0])
                    if n == 0: return n, None
                    drained = min(n, self.max_results)
                    return n, d_result_data[: drained * 96].get()

                n_hits, host = await asyncio.to_thread(_launch)
                total_hashes += hashes_per_launch

                now = time.time()
                dt = now - last_rate_t
                if dt >= 0.5:
                    rate = (total_hashes - last_rate_hashes) / dt
                    hits_rate = (total_hits - last_rate_hits) / dt
                    last_rate_t = now
                    last_rate_hashes = total_hashes
                    last_rate_hits = total_hits
                    if 'on_rate' in self.callbacks:
                        await self.callbacks['on_rate'](rate, hits_rate, hit_queue.qsize(), dropped)
                    print(f"[rate] {rate/1e6:.1f} M/s  total={total_hashes:,}  hits={total_hits}  q={hit_queue.qsize()}  dropped={dropped}")

                if n_hits > 0 and host is not None:
                    drained = min(n_hits, self.max_results)
                    for i in range(drained):
                        rec = host[i * 96 : (i + 1) * 96]
                        b_bytes = bytes(rec[:32])
                        if b_bytes in seen_b: continue
                        seen_b.add(b_bytes)
                        gpu_hex = bytes(rec[32:96]).decode("ascii")
                        total_hits += 1

                        hit_target = None
                        hit_idx = -1
                        for t in targets:
                            j = gpu_hex.find(t)
                            if j != -1:
                                hit_target, hit_idx = t, j
                                break
                        if hit_target is None: continue

                        b_str = b_str_from_bytes(b_bytes)
                        try:
                            hit_queue.put_nowait((b_str, gpu_hex, hit_target, hit_idx, total_hashes))
                        except asyncio.QueueFull:
                            dropped += 1

                    if total_hits - last_hit_broadcast >= 10 or n_hits > 0:
                        last_hit_broadcast = total_hits
                        if 'on_hit_count' in self.callbacks:
                            await self.callbacks['on_hit_count'](total_hits)

            if hit_queue.qsize() > 0:
                await self._log(f"Stopping GPU, processing {hit_queue.qsize()} remaining items in queue...")
                while hit_queue.qsize() > 0:
                    await asyncio.sleep(0.5)
                    now = time.time()
                    dt = now - last_rate_t
                    if dt >= 0.5:
                        hr = (total_hits - last_rate_hits) / dt
                        last_rate_t = now
                        last_rate_hits = total_hits
                        if 'on_rate' in self.callbacks:
                            await self.callbacks['on_rate'](0.0, hr, hit_queue.qsize(), dropped)
                        print(f"[rate] 0.0 M/s  total={total_hashes:,}  hits={total_hits}  q={hit_queue.qsize()}  dropped={dropped}")

            await self._log(f"Stopped. {total_hashes:,} hashes, {total_hits} hits, {dropped} dropped.")
        except asyncio.CancelledError:
            await self._log("Mining cancelled.")
            raise
        except Exception as e:
            await self._log(f"Miner error: {e}")
            import traceback
            await self._log(traceback.format_exc())
            raise
        finally:
            self.is_running = False
            for _ in worker_tasks:
                try: hit_queue.put_nowait(None)
                except asyncio.QueueFull: pass
            for t in worker_tasks: t.cancel()
            process_pool.shutdown(wait=False)
            if 'on_stop' in self.callbacks:
                await self.callbacks['on_stop']()

    def stop(self):
        self.is_running = False
