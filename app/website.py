import asyncio
import json
import time
import urllib.request
import urllib.error
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from .miner import GPUMiner

app = FastAPI(title="Conway GPU Miner Web UI")

class AppState:
    """
    Holds the application state of the Conway Web UI server.
    
    Attributes:
        miner: An active instance of the GPUMiner.
        miner_task: The background asyncio Task orchestrating the mining loops.
        active_connections: List of active WebSocket sessions broadcasting metrics.
        pending_payloads: Dictionary holding cached payload metrics for manually queued submissions.
    """
    def __init__(self):
        self.miner = None
        self.miner_task = None
        self.active_connections = []
        self.pending_payloads = {}  # match_id -> payload

state = AppState()

# -- Web UI HTML --
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Conway GPU Miner UI</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; display: flex; flex-direction: column; height: 100vh; box-sizing: border-box; overflow: hidden; }
        .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; flex-shrink: 0; }
        .controls { display: flex; gap: 15px; align-items: center; background: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; flex-wrap: wrap; }
        select, button, input { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 8px 16px; border-radius: 6px; font-size: 14px; cursor: pointer; outline: none; transition: 0.2s; }
        select:hover, button:hover { background: #30363d; }
        button.start { background: #238636; color: white; border-color: #2ea043; font-weight: 600; }
        button.start:hover { background: #2ea043; }
        button.stop { background: #da3633; color: white; border-color: #f85149; font-weight: 600; }
        button.stop:hover { background: #f85149; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }

        .main-content { display: flex; gap: 20px; flex: 1; min-height: 0; flex-wrap: wrap; overflow-y: auto; padding-right: 5px; }
        .panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; display: flex; flex-direction: column; min-width: 320px; }
        .chart-panel { flex: 2; }
        .log-panel { flex: 1; min-height: 300px; }

        .stats-bar { display: flex; justify-content: space-around; margin-bottom: 15px; background: #0d1117; padding: 10px; border-radius: 6px; border: 1px solid #30363d; }
        .stat-box { text-align: center; }
        .stat-val { font-size: 24px; font-weight: bold; color: #58a6ff; }
        .stat-label { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;}

        canvas { width: 100% !important; height: 100% !important; display: block; }
        #logs { flex: 1; background: #0d1117; color: #79c0ff; border: 1px solid #30363d; border-radius: 6px; padding: 10px; font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; font-size: 13px; overflow-y: auto; white-space: pre-wrap; resize: none; width: 100%; box-sizing: border-box; }
    </style>
</head>
<body>
    <div class="header">
        <h2>GPU Conway Miner</h2>
        <div class="controls">
            <label>Suite:
                <select id="suiteSelect">
                    <option value="4">4</option>
                    <option value="6">6</option>
                    <option value="8" selected>8</option>
                    <option value="any">Any (4, 6, 8)</option>
                </select>
            </label>
            <label>Blocks: <input type="number" id="blocks" value="2048" min="32" max="65535" style="width: 80px;"></label>
            <label>Threads: <input type="number" id="threads" value="256" min="32" max="1024" step="32" style="width: 80px;"></label>
            <label>Iters/thread: <input type="number" id="ipt" value="256" min="1" max="4096" style="width: 80px;"></label>
            <label>CPU Workers: <input type="number" id="workers" value="4" min="1" max="64" style="width: 60px;"></label>
            <button id="btnStart" class="start" onclick="sendCommand('start')">Start</button>
            <button id="btnStop" class="stop" onclick="sendCommand('stop')" disabled>Stop</button>
        </div>
    </div>

    <div class="main-content">
        <div class="panel chart-panel">
            <div class="stats-bar">
                <div class="stat-box">
                    <div class="stat-val" id="hashRate">0.00 M</div>
                    <div class="stat-label">Hashes / Sec</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val" id="totalSuccesses" style="color: #3fb950;">0</div>
                    <div class="stat-label">Submissions</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val" style="color: #f0883e;"><span id="totalHits">0</span> <span style="font-size: 14px; font-weight: normal; color: #8b949e;" id="hitsRate">(0.0/s)</span></div>
                    <div class="stat-label">Total Hits</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val" id="bestIters" style="color: #d2a8ff;">0</div>
                    <div class="stat-label">Best Iterations</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val" id="queueSize" style="color: #e3b341;">0</div>
                    <div class="stat-label">Queue</div>
                </div>
            </div>
            <div style="display: flex; gap: 15px; flex-wrap: wrap; flex: 0 0 auto;">
                <div style="flex: 1 1 250px; height: 350px; position: relative; min-width: 0;"><canvas id="hashChart"></canvas></div>
                <div style="flex: 1 1 250px; height: 350px; position: relative; min-width: 0;"><canvas id="iterHistChart"></canvas></div>
            </div>
            <div style="flex: 1; margin-top: 15px; background: #0d1117; border: 1px solid #30363d; border-radius: 6px; overflow-y: auto; min-height: 200px; max-height: 400px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 12px; text-align: left;">
                    <thead style="background: #21262d; position: sticky; top: 0;">
                        <tr>
                            <th style="padding: 8px;">#</th>
                            <th style="padding: 8px;">Origin Hash</th>
                            <th style="padding: 8px; cursor: pointer; user-select: none;" onclick="sortHistory('iterations')">Iters <span id="sortInd-iterations">▼</span></th>
                            <th style="padding: 8px; cursor: pointer; user-select: none;" onclick="sortHistory('peak')">Peak <span id="sortInd-peak"></span></th>
                            <th style="padding: 8px; cursor: pointer; user-select: none;" onclick="sortHistory('time')">Time <span id="sortInd-time"></span></th>
                            <th style="padding: 8px;">Action</th>
                        </tr>
                    </thead>
                    <tbody id="historyTableBody"></tbody>
                </table>
            </div>
        </div>
        <div class="panel log-panel">
            <h3 style="margin-top: 0; margin-bottom: 10px; font-size: 16px; color: #8b949e;">Mining Console</h3>
            <textarea id="logs" readonly></textarea>
        </div>
    </div>

    <!-- Inspect Modal -->
    <div id="inspectModal" style="display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; align-items: center; justify-content: center;">
        <div style="background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; width: 600px; max-width: 90%;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="margin: 0; color: #c9d1d9;">Inspect Match</h3>
                <button onclick="document.getElementById('inspectModal').style.display='none'" style="background: transparent; border: none; font-size: 18px; color: #8b949e; cursor: pointer;">&times;</button>
            </div>
            <div>
                <label style="font-size: 12px; color: #8b949e;">RLE Format</label>
                <textarea id="rleOutput" readonly style="width: 100%; height: 160px; background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 8px; margin-bottom: 10px; font-family: ui-monospace, monospace; resize: none; box-sizing: border-box;"></textarea>
                <label style="font-size: 12px; color: #8b949e;">Submission Data</label>
                <textarea id="jsonOutput" readonly style="width: 100%; height: 160px; background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 8px; font-family: ui-monospace, monospace; resize: none; box-sizing: border-box;"></textarea>
            </div>
        </div>
    </div>

    <script>
        function computeRLE(binStr) {
            if (!binStr || binStr.length !== 256) return "Invalid binary string";
            let rleParts = [];
            for (let r = 0; r < 16; r++) {
                let row = binStr.substring(r * 16, r * 16 + 16);
                let count = 0;
                let currentChar = row[0];
                let rowRle = "";
                for (let i = 0; i < 16; i++) {
                    if (row[i] === currentChar) {
                        count++;
                    } else {
                        let c = currentChar === '0' ? 'b' : 'o';
                        rowRle += (count > 1 ? count : "") + c;
                        currentChar = row[i];
                        count = 1;
                    }
                }
                if (currentChar === '1') {
                    rowRle += (count > 1 ? count : "") + 'o';
                }
                rleParts.push(rowRle);
            }
            let rleStr = rleParts.join('$') + "!";
            let lines = [];
            for (let i = 0; i < rleStr.length; i += 70) {
                lines.push(rleStr.substring(i, i + 70));
            }
            return "x = 16, y = 16, rule = B3/S23:T16,16\\n" + lines.join('\\n');
        }

        function inspectMatch(matchId) {
            const match = historyData.find(d => d.match_id === matchId);
            if (!match) return;
            
            document.getElementById('rleOutput').value = computeRLE(match.bin);
            document.getElementById('jsonOutput').value = JSON.stringify({
                match_id: match.match_id,
                origin_hash: match.origin_hash,
                iterations: match.iterations,
                peak: match.peak,
                bin: match.bin
            }, null, 2);
            document.getElementById('inspectModal').style.display = 'flex';
        }

        const ws = new WebSocket(`ws://${location.host}/ws`);
        const logsEl = document.getElementById('logs');

        const ctx = document.getElementById('hashChart').getContext('2d');
        const hashChart = new Chart(ctx, {
            type: 'line',
            data: { labels: Array(30).fill(''), datasets: [{ label: 'GPU Hash Rate', data: Array(30).fill(0), borderColor: '#f0883e', backgroundColor: 'rgba(240, 136, 62, 0.15)', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 }] },
            options: { responsive: true, maintainAspectRatio: false, animation: { duration: 0 }, scales: { x: { display: false }, y: { beginAtZero: true, grid: { color: '#30363d' }, ticks: { color: '#8b949e', callback: v => (v / 1e6).toFixed(0) + 'M' } } }, plugins: { legend: { display: false }, title: { display: true, text: 'Hash Rate', color: '#8b949e'} } }
        });

        let iterHistData = {};
        const histCtx = document.getElementById('iterHistChart').getContext('2d');
        const iterHistChart = new Chart(histCtx, {
            type: 'bar',
            data: { labels: [], datasets: [{ data: [], backgroundColor: '#58a6ff', barPercentage: 0.8, categoryPercentage: 0.8 }] },
            options: { responsive: true, maintainAspectRatio: false, animation: { duration: 0 }, scales: { x: { ticks: { color: '#8b949e', maxTicksLimit: 8 }, grid: { display: false } }, y: { type: 'logarithmic', beginAtZero: true, grid: { color: '#30363d' }, ticks: { color: '#8b949e' } } }, plugins: { legend: { display: false }, title: { display: true, text: 'Iteration Distribution (100s)', color: '#8b949e'} } }
        });

        function logToConsole(msg) {
            logsEl.value += msg + '\\n';
            const lines = logsEl.value.split('\\n');
            if (lines.length > 2000) logsEl.value = lines.slice(-2000).join('\\n');
            logsEl.scrollTop = logsEl.scrollHeight;
        }

        function sendCommand(cmd) {
            const sv = document.getElementById('suiteSelect').value;
            ws.send(JSON.stringify({ command: cmd, suite: sv === 'any' ? 'any' : parseInt(sv), blocks: parseInt(document.getElementById('blocks').value), threads: parseInt(document.getElementById('threads').value), ipt: parseInt(document.getElementById('ipt').value), workers: parseInt(document.getElementById('workers').value) }));
        }

        function submitMatch(matchId) { ws.send(JSON.stringify({ command: 'submit_match', match_id: matchId })); }

        let totalHits = 0, successes = 0, bestIters = 0, historyData = [], sortKey = 'iterations', sortDir = -1;

        function sortHistory(key) {
            if (sortKey === key) sortDir = -sortDir; else { sortKey = key; sortDir = -1; }
            ['iterations','peak','time'].forEach(k => { document.getElementById('sortInd-' + k).innerText = (k === sortKey) ? (sortDir === -1 ? '▼' : '▲') : ''; });
            renderHistory();
        }

        function renderHistory() {
            const sorted = historyData.slice().sort((a, b) => (a[sortKey] - b[sortKey]) * sortDir);
            const tbody = document.getElementById('historyTableBody');
            tbody.innerHTML = '';
            let i = 1;
            for (const d of sorted) {
                const row = document.createElement('tr');
                row.style.borderBottom = "1px solid #30363d";
                const tStr = new Date(d.time).toLocaleTimeString();
                
                let actionBtn = d.uploaded ? `<span style="color: #3fb950; font-size: 11px;">Sent</span>` : `<button onclick="submitMatch('${d.match_id}')" style="background: #1f6feb; color: white; border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 11px;">Send</button>`;
                let inspectBtn = `<button onclick="inspectMatch('${d.match_id}')" style="background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 11px; margin-left: 5px;">Inspect</button>`;
                let actionCell = `<div style="display: flex; align-items: center;">${actionBtn}${inspectBtn}</div>`;
                
                row.innerHTML = `<td style="padding: 8px; color: #8b949e;">${i++}</td><td style="padding: 8px; font-family: monospace; color: #c9d1d9;">${d.origin_hash.substring(0, 24)}…</td><td style="padding: 8px; color: #d2a8ff; font-weight: bold;">${d.iterations}</td><td style="padding: 8px;">${d.peak}</td><td style="padding: 8px; color: #8b949e;">${tStr}</td><td style="padding: 8px;">${actionCell}</td>`;
                tbody.appendChild(row);
            }
            document.getElementById('kept').innerText = historyData.length;
        }

        let renderPending = false;
        function scheduleRender() { if (renderPending) return; renderPending = true; setTimeout(() => { renderPending = false; renderHistory(); }, 250); }

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'clear_history') { historyData = []; iterHistData = {}; renderHistory(); }
            else if (data.type === 'log') logToConsole(data.message);
            else if (data.type === 'rate') {
                document.getElementById('hashRate').innerText = (data.rate / 1e6).toFixed(2) + ' M';
                document.getElementById('hitsRate').innerText = '(' + (data.hits_rate || 0).toFixed(1) + '/s)';
                document.getElementById('queueSize').innerText = data.queue || 0;

                hashChart.data.datasets[0].data.shift(); hashChart.data.datasets[0].data.push(data.rate); hashChart.update();
                
                const buckets = Object.keys(iterHistData).map(Number).sort((a,b) => a-b);
                if (buckets.length > 0) {
                    iterHistChart.data.labels = []; iterHistChart.data.datasets[0].data = [];
                    for (let b = Math.max(0, buckets[0]); b <= buckets[buckets.length - 1]; b += 100) { iterHistChart.data.labels.push(b); iterHistChart.data.datasets[0].data.push(iterHistData[b] || 0); }
                    iterHistChart.update();
                }
            } else if (data.type === 'hit_count') { totalHits = data.total; document.getElementById('totalHits').innerText = totalHits; }
            else if (data.type === 'submission_success') { successes++; document.getElementById('totalSuccesses').innerText = successes; }
            else if (data.type === 'match_history') {
                if (data.iterations > bestIters) { bestIters = data.iterations; document.getElementById('bestIters').innerText = bestIters; }
                let bucket = Math.floor(data.iterations / 100) * 100;
                iterHistData[bucket] = (iterHistData[bucket] || 0) + 1;
                historyData.push({ match_id: data.match_id, origin_hash: data.origin_hash, iterations: data.iterations, peak: data.peak, uploaded: data.uploaded, time: data.time || Date.now(), bin: data.bin });
                if (historyData.length > 500) {
                    const uploaded = historyData.filter(d => d.uploaded), others = historyData.filter(d => !d.uploaded);
                    others.sort((a, b) => b.iterations !== a.iterations ? b.iterations - a.iterations : (b.peak !== a.peak ? b.peak - a.peak : b.time - a.time));
                    historyData = uploaded.concat(others.slice(0, Math.max(50, 500 - uploaded.length)));
                }
                scheduleRender();
            } else if (data.type === 'match_update') {
                const e = historyData.find(d => d.match_id === data.match_id);
                if (e) { e.uploaded = data.uploaded; scheduleRender(); }
            } else if (data.type === 'state_update') {
                const r = data.is_running;
                ['btnStart','btnStop','suiteSelect','blocks','threads','ipt','workers'].forEach(id => document.getElementById(id).disabled = id==='btnStop' ? !r : r);
            }
        };

        ws.onopen = () => logToConsole("Connected to GPU backend");
        ws.onclose = () => logToConsole("Disconnected. Refresh to reconnect.");
    </script>
</body>
</html>
"""

async def broadcast(data: dict) -> None:
    """
    Transmit raw dictionary payload simultaneously to all active websocket clients.
    Automatically purges any dead or disconnected sockets.

    Args:
        data (dict): The serialized dictionary data to broadcast.
    """
    msg = json.dumps(data)
    dead = []
    for conn in state.active_connections:
        try:
            await conn.send_text(msg)
        except Exception:
            dead.append(conn)
    for d in dead:
        state.active_connections.remove(d)

async def broadcast_log(msg: str):
    print(msg)
    await broadcast({"type": "log", "message": msg})

async def broadcast_state():
    is_running = state.miner is not None and getattr(state.miner, 'is_running', False)
    await broadcast({"type": "state_update", "is_running": is_running})

async def submit_payload(payload: dict) -> bool:
    def post_data():
        req = urllib.request.Request(
            "https://lifehashes.net/dailychallenge/save_glyph.php",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode()
            return json.loads(body) if body.strip() else {"success": False, "error": "Empty response"}

    try:
        res = await asyncio.to_thread(post_data)
        if res.get("success"):
            await broadcast_log(f"Success! Glyph ID {res.get('id', '?')} registered.")
            await broadcast({"type": "submission_success"})
            return True
        await broadcast_log(f"Server rejected: {res}")
        return False
    except Exception as e:
        await broadcast_log(f"Upload failed: {e}")
        return False

@app.get("/", response_class=HTMLResponse)
def get_ui():
    return HTML

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.active_connections.append(websocket)
    await broadcast_state()
    try:
        while True:
            msg = await websocket.receive_json()
            cmd = msg.get("command")

            if cmd == "start":
                if state.miner and state.miner.is_running: continue
                target_suite = msg.get("suite", "any")
                blocks = max(32, int(msg.get("blocks", 1024)))
                threads = max(32, int(msg.get("threads", 256)))
                ipt = max(1, int(msg.get("ipt", 64)))
                workers = max(1, int(msg.get("workers", 4)))
                
                async def on_log(m: str): await broadcast_log(m)
                async def on_rate(r: float, hr: float, q: int, d: int): await broadcast({"type": "rate", "rate": r, "hits_rate": hr, "queue": q, "dropped": d})
                async def on_hit_count(total: int): await broadcast({"type": "hit_count", "total": total})
                async def on_match_history(mid: str, h: str, iters: int, peak: int, p: dict):
                    state.pending_payloads[mid] = p
                    await broadcast({"type": "match_history", "match_id": mid, "origin_hash": h, "iterations": iters, "peak": peak, "uploaded": False, "time": int(time.time() * 1000), "bin": p.get("bin")})
                async def on_stop():
                    state.miner = None
                    await broadcast_state()
                
                state.miner = GPUMiner(callbacks={
                    'on_log': on_log, 'on_rate': on_rate, 'on_hit_count': on_hit_count, 
                    'on_match_history': on_match_history, 'on_stop': on_stop
                })
                
                state.miner.is_running = True
                await broadcast_state()
                state.miner_task = asyncio.create_task(state.miner.run(target_suite, blocks, threads, ipt, workers))

            elif cmd == "stop":
                if state.miner:
                    state.miner.stop()
                await broadcast_state()

            elif cmd == "submit_match":
                mid = msg.get("match_id")
                payload = state.pending_payloads.get(mid)
                if not payload:
                    await broadcast_log(f"Manual submit: match not found.")
                    continue
                await broadcast_log(f"Submitting match {mid[:8]}...")
                ok = await submit_payload(payload)
                if ok:
                    state.pending_payloads.pop(mid, None)
                    await broadcast({"type": "match_update", "match_id": mid, "uploaded": True})

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.active_connections:
            state.active_connections.remove(websocket)

if __name__ == "__main__":
    print("Starting GPU miner web UI on http://127.0.0.1:5001")
    uvicorn.run("app.website:app", host="127.0.0.1", port=5001, log_level="warning")