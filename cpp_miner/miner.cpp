// miner.cpp — CUDA 16x16 toroidal Game of Life max-iteration miner
//
// Build:
//   nvcc -O3 -arch=sm_86 -std=c++17 miner.cu -o miner.exe
//   (adjust sm_86 to your GPU: sm_75=Turing, sm_80=Ampere, sm_89=Ada)
//
// Run:
//   .\miner.exe 2000 --min-density 0.3 --max-density 0.7                        

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <random>

namespace fs = std::filesystem;

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at line %d: %s\n", #call, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Grid: 16 rows stored as uint16_t, packed into two uint64_t for fast compare.
// row[i] lives in bits [i*16 .. i*16+15] of the two uint64_t words.
// ---------------------------------------------------------------------------
struct Grid {
    uint16_t r[16];
    __device__ __host__ bool operator==(const Grid& o) const {
        // compare as four 64-bit words for speed
        const uint64_t* a = reinterpret_cast<const uint64_t*>(r);
        const uint64_t* b = reinterpret_cast<const uint64_t*>(o.r);
        return a[0]==b[0] && a[1]==b[1] && a[2]==b[2] && a[3]==b[3];
    }
};

// ---------------------------------------------------------------------------
// Bitboard step — identical logic to CPU miner, works in registers on GPU.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint16_t rotl16(uint16_t x, int n) {
    return (uint16_t)((x << n) | (x >> (16 - n)));
}
__device__ __forceinline__ uint16_t rotr16(uint16_t x, int n) {
    return (uint16_t)((x >> n) | (x << (16 - n)));
}

__device__ __forceinline__ Grid step(Grid in) {
    Grid out;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int up = (i == 0) ? 15 : i - 1;
        int dn = (i == 15) ? 0  : i + 1;
        uint16_t U = in.r[up], C = in.r[i], D = in.r[dn];
        uint16_t UL = rotl16(U,1), UR = rotr16(U,1);
        uint16_t CL = rotl16(C,1), CR = rotr16(C,1);
        uint16_t DL = rotl16(D,1), DR = rotr16(D,1);

        uint16_t low=0, mid=0, hi=0;
        #define ADDBIT(x) { uint16_t cy=low&(x); low^=(x); uint16_t cy2=mid&cy; mid^=cy; hi|=cy2; }
        ADDBIT(UL) ADDBIT(U) ADDBIT(UR)
        ADDBIT(CL)            ADDBIT(CR)
        ADDBIT(DL) ADDBIT(D) ADDBIT(DR)
        #undef ADDBIT

        uint16_t n3 =  low &  mid & ~hi;
        uint16_t n2 = ~low &  mid & ~hi;
        out.r[i] = (uint16_t)(n3 | (n2 & C));
    }
    return out;
}

// ---------------------------------------------------------------------------
// xoshiro256** — one 64-bit output, full period, GPU-friendly.
// ---------------------------------------------------------------------------
struct Xoshiro {
    uint64_t s[4];

    __device__ void init(uint64_t seed) {
        // splitmix64 init
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            seed += 0x9E3779B97F4A7C15ULL;
            uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            s[i] = z ^ (z >> 31);
        }
    }

    __device__ __forceinline__ uint64_t next() {
        auto rotl = [](uint64_t x, int k) { return (x << k) | (x >> (64 - k)); };
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;    s[3] = rotl(s[3], 45);
        return result;
    }
};

// ---------------------------------------------------------------------------
// 35% density grid generation via per-bit threshold.
// THRESH = 0.35 * 2^64
// ---------------------------------------------------------------------------
// 100-wide bins covering iterations 0–31 999
static constexpr int HIST_BINS  = 320;
static constexpr int HIST_BIN_W = 100;

__device__ __forceinline__ Grid random_grid(Xoshiro& rng, uint64_t thresh) {
    Grid g;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint16_t row = 0;
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            if (rng.next() < thresh) row |= (uint16_t)(1 << j);
        g.r[i] = row;
    }
    return g;
}

// ---------------------------------------------------------------------------
// Result struct written to device buffer when a long-lived grid is found.
// ---------------------------------------------------------------------------
struct Result {
    Grid  grid;
    int   iterations;  // mu + lam
    int   lam;         // cycle length
    float density;     // per-game starting density parameter
};

// ---------------------------------------------------------------------------
// Kernel — each thread runs independent games via Brent's cycle detection.
// ---------------------------------------------------------------------------
__global__ void mine_kernel(
    uint64_t    seed_base,
    int         min_iters,
    int         max_iters,
    uint64_t    min_thresh,      // density range: [min_thresh, max_thresh]
    uint64_t    max_thresh,
    Result*     d_results,
    uint32_t*   d_count,
    uint32_t    max_results,
    int*        d_global_best,
    unsigned long long* d_histogram
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    Xoshiro rng;
    rng.init(seed_base + tid);

    // Pick a per-game density threshold uniformly in [min_thresh, max_thresh]
    uint64_t d_thresh;
    if (max_thresh == min_thresh) {
        d_thresh = min_thresh;
    } else {
        // Use upper 53 bits of RNG output for a clean double in [0,1)
        double frac = (double)(rng.next() >> 11) * (1.0 / (double)(1ULL << 53));
        d_thresh = min_thresh + (uint64_t)(frac * (double)(max_thresh - min_thresh));
    }
    float density = (float)((double)d_thresh / 18446744073709551615.0);

    Grid initial = random_grid(rng, d_thresh);

    // -----------------------------------------------------------------------
    // Brent's cycle detection — Phase 1: find cycle length (lam)
    // -----------------------------------------------------------------------
    Grid tortoise = initial;
    Grid hare     = step(initial);
    int  power    = 1;
    int  lam      = 1;
    int  p1_steps = 1;

    {
        bool z = true;
        #pragma unroll
        for (int i = 0; i < 16; ++i) if (hare.r[i]) { z = false; break; }
        if (z) {
            // died at step 1: iters=1, lam=1 (matches CPU miner)
            atomicAdd(&d_histogram[min(1 / HIST_BIN_W, HIST_BINS - 1)], 1ULL);
            if (1 >= min_iters) {
                uint32_t idx = atomicAdd(d_count, 1);
                if (idx < max_results) {
                    d_results[idx].grid       = initial;
                    d_results[idx].iterations = 1;
                    d_results[idx].lam        = 1;
                    d_results[idx].density    = density;
                }
            }
            return;
        }
    }

    bool found = false;
    while (p1_steps < max_iters) {
        if (hare == tortoise) { found = true; break; }

        if (power == lam) {
            tortoise = hare;
            power <<= 1;
            lam = 0;
        }

        hare = step(hare);
        ++p1_steps;
        ++lam;

        bool z = true;
        #pragma unroll
        for (int i = 0; i < 16; ++i) if (hare.r[i]) { z = false; break; }
        if (z) {
            // grid died: save as STATIC (lam=1), matches CPU miner
            int dead_iters = p1_steps;
            atomicAdd(&d_histogram[min(dead_iters / HIST_BIN_W, HIST_BINS - 1)], 1ULL);
            atomicMax(d_global_best, dead_iters);
            if (dead_iters >= min_iters) {
                uint32_t idx = atomicAdd(d_count, 1);
                if (idx < max_results) {
                    d_results[idx].grid       = initial;
                    d_results[idx].iterations = dead_iters;
                    d_results[idx].lam        = 1;
                    d_results[idx].density    = density;
                }
            }
            return;
        }
    }

    if (!found) return; // hit cap — skip this grid

    // -----------------------------------------------------------------------
    // Phase 2: find transient length (mu)
    // Advance hare by lam steps from start, then walk both until they meet.
    // -----------------------------------------------------------------------
    tortoise = initial;
    hare     = initial;
    for (int i = 0; i < lam; ++i) hare = step(hare);

    int mu = 0;
    while (!(tortoise == hare)) {
        tortoise = step(tortoise);
        hare     = step(hare);
        ++mu;
        if (mu >= max_iters) return; // safety
    }

    int iters = mu + lam -1; // matches CPU miner's definition exactly

    {
        int bin = iters / HIST_BIN_W;
        atomicAdd(&d_histogram[bin < HIST_BINS ? bin : HIST_BINS - 1], 1ULL);
    }

    atomicMax(d_global_best, iters);

    if (iters >= min_iters) {
        uint32_t idx = atomicAdd(d_count, 1);
        if (idx < max_results) {
            d_results[idx].grid       = initial;
            d_results[idx].iterations = iters;
            d_results[idx].lam        = lam;
            d_results[idx].density    = density;
        }
    }
}

// ---------------------------------------------------------------------------
// Host helpers: JSON output compatible with gol_miner.cpp format
// ---------------------------------------------------------------------------
static std::string grid_to_binstr(const Grid& g) {
    std::string s(256, '0');
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            if ((g.r[i] >> j) & 1) s[i*16+j] = '1';
    return s;
}

static std::string grid_to_json_lines(const Grid& g) {
    std::string out = "[\n";
    for (int i = 0; i < 16; ++i) {
        out += "    \"";
        for (int j = 0; j < 16; ++j) out += ((g.r[i] >> j) & 1) ? '1' : '0';
        out += (i == 15) ? "\"\n" : "\",\n";
    }
    out += "  ]";
    return out;
}

// ---------------------------------------------------------------------------
// Host-side step and stats — mirrors collect_stats in gol_miner.cpp
// ---------------------------------------------------------------------------
static uint16_t h_rotl16(uint16_t x, int n) { return (uint16_t)((x << n) | (x >> (16 - n))); }
static uint16_t h_rotr16(uint16_t x, int n) { return (uint16_t)((x >> n) | (x << (16 - n))); }

static Grid step_host(Grid in) {
    Grid out;
    for (int i = 0; i < 16; ++i) {
        int up = (i == 0) ? 15 : i - 1;
        int dn = (i == 15) ? 0  : i + 1;
        uint16_t U = in.r[up], C = in.r[i], D = in.r[dn];
        uint16_t UL = h_rotl16(U,1), UR = h_rotr16(U,1);
        uint16_t CL = h_rotl16(C,1), CR = h_rotr16(C,1);
        uint16_t DL = h_rotl16(D,1), DR = h_rotr16(D,1);
        uint16_t low=0, mid=0, hi=0;
        #define HADDBIT(x) { uint16_t cy=low&(x); low^=(x); uint16_t cy2=mid&cy; mid^=cy; hi|=cy2; }
        HADDBIT(UL) HADDBIT(U) HADDBIT(UR)
        HADDBIT(CL)             HADDBIT(CR)
        HADDBIT(DL) HADDBIT(D) HADDBIT(DR)
        #undef HADDBIT
        uint16_t n3 =  low &  mid & ~hi;
        uint16_t n2 = ~low &  mid & ~hi;
        out.r[i] = (uint16_t)(n3 | (n2 & C));
    }
    return out;
}

static int popcount16h(uint16_t x) {
    x = (uint16_t)(x - ((x >> 1) & 0x5555u));
    x = (uint16_t)((x & 0x3333u) + ((x >> 2) & 0x3333u));
    x = (uint16_t)((x + (x >> 4)) & 0x0F0Fu);
    return (int)((x * 0x0101u) >> 8);
}

struct HostStats {
    int         peak_freq;
    int         min_alive;
    int         max_alive;
    const char* terminus;
};

static HostStats collect_stats_host(Grid g, int total_iters, int lam) {
    int freq[16][16] = {};
    int min_alive = 256, max_alive = 0;
    for (int iter = 0; iter < total_iters; ++iter) {
        int alive = 0;
        for (int i = 0; i < 16; ++i) {
            alive += popcount16h(g.r[i]);
            for (int j = 0; j < 16; ++j) freq[i][j] += (g.r[i] >> j) & 1;
        }
        if (alive < min_alive) min_alive = alive;
        if (alive > max_alive) max_alive = alive;
        g = step_host(g);
    }
    int peak = 0;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            if (freq[i][j] > peak) peak = freq[i][j];

    const char* term;
    if      (lam ==  1) term = "STATIC";
    else if (lam ==  2) term = "2-FLICKER";
    else if (lam ==  3) term = "3-FLICKER";
    else if (lam == 64) term = "GLIDER";
    else                term = "OTHER";

    return { peak, min_alive, max_alive, term };
}

// ---------------------------------------------------------------------------
// Histogram helpers — same BIN=100 format as gol_miner.cpp / submit_gol.py
// ---------------------------------------------------------------------------
static void load_histogram(const fs::path& dir, std::vector<unsigned long long>& hist) {
    std::ifstream f(dir / "histogram.csv");
    if (!f.is_open()) return;
    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        size_t dash  = line.find('-');
        size_t comma = line.find(',');
        if (dash == std::string::npos || comma == std::string::npos) continue;
        int      bin   = std::stoi(line.substr(0, dash)) / HIST_BIN_W;
        uint64_t count = std::stoull(line.substr(comma + 1));
        if (bin >= 0 && bin < HIST_BINS) hist[bin] += count;
    }
    printf("  Loaded existing histogram from histogram.csv\n");
}

static void write_histogram(const fs::path& dir, const std::vector<unsigned long long>& hist) {
    std::ofstream f(dir / "histogram.csv", std::ios::trunc);
    f << "iterations_bin,count\n";
    for (int b = 0; b < (int)hist.size(); ++b)
        if (hist[b])
            f << (b * HIST_BIN_W) << "-" << (b * HIST_BIN_W + HIST_BIN_W - 1)
              << "," << hist[b] << "\n";
}

static void save_json(const fs::path& p, const Grid& g, int iters, int lam,
                      float density, const HostStats& st, uint64_t total_searched) {
    std::ofstream f(p, std::ios::trunc);
    char dbuf[16]; snprintf(dbuf, sizeof(dbuf), "%.4f", density);
    f << "{\n";
    f << "  \"iterations\": " << iters             << ",\n";
    f << "  \"peak\": "        << st.peak_freq      << ",\n";
    f << "  \"min\": "         << st.min_alive       << ",\n";
    f << "  \"max\": "         << st.max_alive       << ",\n";
    f << "  \"cycle_len\": "   << lam               << ",\n";
    f << "  \"terminus\": \""  << st.terminus       << "\",\n";
    f << "  \"density\": "     << dbuf              << ",\n";
    f << "  \"attempts\": "    << total_searched    << ",\n";
    f << "  \"bin\": \""       << grid_to_binstr(g) << "\",\n";
    f << "  \"start_grid\": "  << grid_to_json_lines(g) << "\n";
    f << "}\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    int min_iters   = 1800;
    double min_density = 0.35;
    double max_density = 0.35;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--min-density") == 0 && i + 1 < argc)
            min_density = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--max-density") == 0 && i + 1 < argc)
            max_density = std::atof(argv[++i]);
        else
            min_iters = std::atoi(argv[i]); // positional = min_iters
    }
    if (max_density < min_density) max_density = min_density;

    uint64_t min_thresh = (uint64_t)(min_density * 18446744073709551615.0);
    uint64_t max_thresh = (uint64_t)(max_density * 18446744073709551615.0);

    int max_iters  = 32000;
    int tpb        = 256;    // threads per block
    int blocks     = 32768/2;   // blocks per launch
    uint32_t max_results = 4096; // device result buffer size

    fs::path out_dir = "gol_output";
    fs::create_directories(out_dir);

    // Device allocations
    Result*  d_results;
    uint32_t* d_count;
    int*      d_best;
    unsigned long long* d_histogram;
    CHECK_CUDA(cudaMalloc(&d_results,   max_results * sizeof(Result)));
    CHECK_CUDA(cudaMalloc(&d_count,     sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_best,      sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_histogram, HIST_BINS * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_count,     0, sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_best,      0, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_histogram, 0, HIST_BINS * sizeof(unsigned long long)));

    const uint64_t PHI = 0x9E3779B97F4A7C15ULL;
    int global_best = 0;
    uint64_t total_searched = 0;
    uint64_t total_saved    = 0;
    uint64_t launch         = 0;

    std::vector<unsigned long long> h_histogram(HIST_BINS, 0);
    load_histogram(out_dir, h_histogram);
    // Upload pre-existing histogram counts to device so they accumulate correctly
    CHECK_CUDA(cudaMemcpy(d_histogram, h_histogram.data(),
                          HIST_BINS * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    printf("=========================================\n");
    printf("  GoL CUDA Miner\n");
    printf("  threads/block: %d   blocks: %d\n", tpb, blocks);
    printf("  grids/launch : %d\n", tpb * blocks);
    printf("  min save     : %d iters\n", min_iters);
    if (min_density == max_density)
        printf("  density      : %.4f (fixed)\n", min_density);
    else
        printf("  density      : %.4f – %.4f (random per game)\n", min_density, max_density);
    printf("  output dir   : %s\n", out_dir.string().c_str());
    printf("=========================================\n");
    printf("Press Ctrl+C to stop.\n\n");

    auto t0 = std::chrono::steady_clock::now();
    uint64_t grids_per_launch = (uint64_t)tpb * blocks;

    // Staging buffer
    std::vector<Result> host_results(max_results);

    std::random_device rd;
    uint64_t global_seed = ((uint64_t)rd() << 32) ^ rd();

    while (true) {
        uint64_t seed = global_seed + launch * grids_per_launch;

        // Reset result counter each launch (results buffer reused)
        CHECK_CUDA(cudaMemset(d_count, 0, sizeof(uint32_t)));

        mine_kernel<<<blocks, tpb>>>(seed, min_iters, max_iters,
                                     min_thresh, max_thresh,
                                     d_results, d_count, max_results, d_best,
                                     d_histogram);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Read back count + best
        uint32_t found = 0;
        int dev_best   = 0;
        CHECK_CUDA(cudaMemcpy(&found,    d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&dev_best, d_best,  sizeof(int),      cudaMemcpyDeviceToHost));

        // Copy device histogram to host (it accumulates permanently on device)
        CHECK_CUDA(cudaMemcpy(h_histogram.data(), d_histogram,
                              HIST_BINS * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        total_searched += grids_per_launch;
        ++launch;

        if (found > 0) {
            uint32_t n = found < max_results ? found : max_results;
            CHECK_CUDA(cudaMemcpy(host_results.data(), d_results, n * sizeof(Result), cudaMemcpyDeviceToHost));

            for (uint32_t i = 0; i < n; ++i) {
                const Result& res = host_results[i];

                bool is_best = (res.iterations > global_best);
                if (is_best) global_best = res.iterations;

                char name[128];
                if (is_best) {
                    snprintf(name, sizeof(name), "best_%05d.json", res.iterations);
                } else {
                    // t0 = GPU device 0; n = sequential saved count
                    snprintf(name, sizeof(name), "top_%05d_t0_n%llu.json",
                             res.iterations, (unsigned long long)total_saved);
                }
                HostStats st = collect_stats_host(res.grid, res.iterations, res.lam);
                save_json(out_dir / name, res.grid, res.iterations, res.lam, res.density, st, grids_per_launch);
                ++total_saved;
            }
        }

        if (dev_best > global_best) global_best = dev_best;

        // Write histogram.csv every 30 seconds
        {
            static auto last_hist_write = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_hist_write).count() >= 30) {
                write_histogram(out_dir, h_histogram);
                last_hist_write = now;
            }
        }

        // Progress every launch
        auto el = std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::steady_clock::now() - t0).count();
        uint64_t rate = (el > 0) ? total_searched / (uint64_t)el : 0;
        printf("\r[%llds] rate=%8llu g/s  total=%12llu  max=%d  saved=%llu   ",
               (long long)el, (unsigned long long)rate,
               (unsigned long long)total_searched,
               global_best,
               (unsigned long long)total_saved);
        fflush(stdout);
    }

    cudaFree(d_results);
    cudaFree(d_count);
    cudaFree(d_best);
    cudaFree(d_histogram);
    return 0;
}
