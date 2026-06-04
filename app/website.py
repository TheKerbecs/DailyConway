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
        self.match_buffer = []
        self.match_flush_task = None
        self.custom_mode = False

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
            <label id="suiteLabel">Suite:
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
            <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;" title="Custom search: define hex-string constraints (front/back/any). All must match. No NIST salt. No submission, only Save File.">
                <input type="checkbox" id="customMode" style="width: 16px; height: 16px; cursor: pointer;" onchange="onCustomToggle()">
                <span>Custom Search</span>
            </label>
            <button id="btnStart" class="start" onclick="sendCommand('start')">Start</button>
            <button id="btnStop" class="stop" onclick="sendCommand('stop')" disabled>Stop</button>
        </div>
    </div>

    <div id="customControls" style="display:none; background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px 15px; margin-bottom:15px; flex-shrink:0;">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
            <strong style="color:#c9d1d9; font-size:14px;">Custom search mode:</strong>
            <label style="display:flex; align-items:center; gap:4px; cursor:pointer;">
                <input type="radio" name="customKind" value="groups" checked onchange="onKindChange()"> <span>Groups (AND of OR)</span>
            </label>
            <label style="display:flex; align-items:center; gap:4px; cursor:pointer;">
                <input type="radio" name="customKind" value="regex" onchange="onKindChange()"> <span>Regex</span>
            </label>
        </div>

        <div id="groupsPanel">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; flex-wrap:wrap; gap:8px;">
                <span style="color:#8b949e; font-size:12px;">Groups are AND-ed; rows within a group are OR-ed.</span>
                <button onclick="addGroup()" style="background:#1f6feb; color:white; border:none; padding:6px 12px; border-radius:4px; cursor:pointer; font-size:12px;">+ AND group</button>
            </div>
            <div id="groupsList" style="display:flex; flex-direction:column; gap:10px;"></div>
            <div id="exprPreview" style="margin-top:10px; font-family:ui-monospace,monospace; font-size:12px; color:#8b949e;"></div>
        </div>

        <div id="regexPanel" style="display:none;">
            <div style="display:flex; flex-direction:column; gap:8px;">
                <label style="display:flex; flex-direction:column; gap:4px;">
                    <span style="font-size:12px; color:#8b949e;">Pattern (Python <code>re</code> syntax)</span>
                    <input id="regexPattern" type="text" placeholder="e.g. ^513.*(b0b1400|626f62696e6f75)" style="width:100%; background:#0d1117; color:#c9d1d9; border:1px solid #30363d; padding:8px; border-radius:4px; font-family:ui-monospace,monospace;">
                </label>
                <div style="display:flex; gap:10px; flex-wrap:wrap;">
                    <label style="display:flex; flex-direction:column; gap:4px; flex:0 0 120px;">
                        <span style="font-size:12px; color:#8b949e;">Flags (i s m x)</span>
                        <input id="regexFlags" type="text" maxlength="4" placeholder="e.g. i" style="background:#0d1117; color:#c9d1d9; border:1px solid #30363d; padding:8px; border-radius:4px; font-family:ui-monospace,monospace;">
                    </label>
                    <label style="display:flex; flex-direction:column; gap:4px; flex:1; min-width:200px;" title="GPU pre-filter: a literal lowercase hex substring (1-16) that MUST appear somewhere in the hash. The full regex runs on the CPU on candidates. Required because regex on the GPU is impractical at 100M+ hashes/sec.">
                        <span style="font-size:12px; color:#8b949e;">GPU pre-filter anchor (lowercase hex, 1-16) <span style="color:#f85149;">*required</span></span>
                        <input id="regexAnchor" type="text" maxlength="16" placeholder="e.g. b0b1400" style="background:#0d1117; color:#c9d1d9; border:1px solid #30363d; padding:8px; border-radius:4px; font-family:ui-monospace,monospace;">
                    </label>
                </div>
                <div style="background:#0d1117; border:1px solid #30363d; border-radius:6px; padding:10px;">
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
                        <span style="color:#8b949e; font-size:12px;">Live tester — paste a 64-char hex hash:</span>
                        <span id="regexTestStatus" style="font-family:ui-monospace,monospace; font-size:12px;">—</span>
                    </div>
                    <textarea id="regexTestInput" rows="2" placeholder="51300000000b0b1400..." style="width:100%; background:#161b22; color:#c9d1d9; border:1px solid #30363d; padding:8px; border-radius:4px; font-family:ui-monospace,monospace; resize:vertical; box-sizing:border-box;"></textarea>
                    <div id="regexTestDetail" style="margin-top:6px; font-family:ui-monospace,monospace; font-size:12px; color:#8b949e; word-break:break-all;"></div>
                </div>
            </div>
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
            <h3 style="margin-top: 0; margin-bottom: 10px; font-size: 16px; color: #8b949e;">Console</h3>
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
            const customMode = document.getElementById('customMode').checked;
            const payload = {
                command: cmd,
                suite: sv === 'any' ? 'any' : parseInt(sv),
                blocks: parseInt(document.getElementById('blocks').value),
                threads: parseInt(document.getElementById('threads').value),
                ipt: parseInt(document.getElementById('ipt').value),
                workers: parseInt(document.getElementById('workers').value),
                custom_mode: customMode,
            };
            if (customMode && cmd === 'start') {
                if (getKind() === 'regex') {
                    const r = collectRegex();
                    if (!r) return;
                    payload.regex = r;
                } else {
                    const groups = collectGroups();
                    if (!groups) return;
                    payload.groups = groups;
                }
            }
            ws.send(JSON.stringify(payload));
        }

        function getKind() {
            const el = document.querySelector('input[name="customKind"]:checked');
            return el ? el.value : 'groups';
        }

        function onKindChange() {
            const kind = getKind();
            document.getElementById('groupsPanel').style.display = (kind === 'groups') ? '' : 'none';
            document.getElementById('regexPanel').style.display  = (kind === 'regex')  ? '' : 'none';
            if (kind === 'regex' && !document.getElementById('regexPattern').value) {
                document.getElementById('regexPattern').value = '^513.*(b0b1400|626f62696e6f75)';
                document.getElementById('regexFlags').value = '';
                document.getElementById('regexAnchor').value = 'b0b1400';
                runRegexTest();
            }
        }

        function pyFlagsToJs(f) {
            // Map Python re flags to JS RegExp flags. 'x' (verbose) is not supported in JS.
            let out = '';
            for (const ch of (f || '').toLowerCase()) {
                if (ch === 'i') out += 'i';
                else if (ch === 's') out += 's';
                else if (ch === 'm') out += 'm';
                // 'x' silently dropped for the JS-side tester.
            }
            return out;
        }

        function runRegexTest() {
            const pat = document.getElementById('regexPattern').value;
            const flags = document.getElementById('regexFlags').value;
            const anchor = document.getElementById('regexAnchor').value.trim().toLowerCase();
            const input = document.getElementById('regexTestInput').value.trim().toLowerCase();
            const status = document.getElementById('regexTestStatus');
            const detail = document.getElementById('regexTestDetail');
            detail.innerHTML = '';

            if (!pat) {
                status.innerText = '— (no pattern)';
                status.style.color = '#8b949e';
                return;
            }
            let re;
            try {
                re = new RegExp(pat, pyFlagsToJs(flags));
            } catch (e) {
                status.innerText = 'invalid regex';
                status.style.color = '#f85149';
                detail.innerText = String(e.message || e);
                return;
            }
            if (!input) {
                status.innerText = '— (paste a hash)';
                status.style.color = '#8b949e';
                return;
            }
            const anchorOk = !anchor || input.indexOf(anchor) !== -1;
            const m = re.exec(input);
            if (m && anchorOk) {
                status.innerText = '✓ matches';
                status.style.color = '#3fb950';
                detail.innerHTML = `match=<span style="color:#79c0ff;">${m[0]}</span> at index ${m.index}` +
                    (m.length > 1 ? `<br>groups: ${JSON.stringify(m.slice(1))}` : '');
            } else if (m && !anchorOk) {
                status.innerText = '✗ regex matches but anchor missing';
                status.style.color = '#f0883e';
                detail.innerText = `Anchor '${anchor}' not found in hash. The GPU would never emit this hash.`;
            } else {
                status.innerText = '✗ no match';
                status.style.color = '#f85149';
            }
        }

        function collectRegex() {
            const pattern = document.getElementById('regexPattern').value;
            const flags = document.getElementById('regexFlags').value.toLowerCase().replace(/[^ismx]/g, '');
            const anchor = document.getElementById('regexAnchor').value.trim().toLowerCase();
            if (!pattern) { alert('Regex pattern cannot be empty.'); return null; }
            if (pattern.length > 256) { alert('Regex pattern exceeds 256 chars.'); return null; }
            if (!/^[0-9a-f]{1,16}$/.test(anchor)) {
                alert('GPU pre-filter anchor must be 1-16 lowercase hex chars.');
                return null;
            }
            try { new RegExp(pattern, pyFlagsToJs(flags)); }
            catch (e) { alert('Invalid regex: ' + (e.message || e)); return null; }
            return { pattern, flags, anchor };
        }

        function onCustomToggle() {
            const on = document.getElementById('customMode').checked;
            document.getElementById('customControls').style.display = on ? 'block' : 'none';
            document.getElementById('suiteLabel').style.display = on ? 'none' : '';
            if (on && document.getElementById('groupsList').children.length === 0) {
                // Default example: 513 AND (b0b1400 OR 626f62696e6f75)
                addGroup([{ position: 'front', value: '513' }]);
                addGroup([{ position: 'any',   value: 'b0b1400' },
                         { position: 'any',   value: '626f62696e6f75' }]);
            }
            updateExprPreview();
        }

        function addGroup(initialRows) {
            const list = document.getElementById('groupsList');
            const groupCount = list.querySelectorAll('.group-card').length;
            const card = document.createElement('div');
            card.className = 'group-card';
            card.style.cssText = 'background:#0d1117; border:1px solid #30363d; border-radius:6px; padding:10px;';
            card.innerHTML = `
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
                    <span style="color:#79c0ff; font-size:12px; font-weight:bold; letter-spacing:0.5px;">${groupCount === 0 ? 'GROUP 1' : 'AND GROUP ' + (groupCount + 1)} <span style="color:#8b949e; font-weight:normal;">(rows OR-ed)</span></span>
                    <div style="display:flex; gap:6px;">
                        <button class="g-add-or" style="background:#238636; color:white; border:none; padding:4px 10px; border-radius:4px; cursor:pointer; font-size:11px;">+ OR row</button>
                        <button class="g-remove" style="background:#da3633; color:white; border:none; padding:4px 10px; border-radius:4px; cursor:pointer; font-size:11px;">Remove group</button>
                    </div>
                </div>
                <div class="g-rows" style="display:flex; flex-direction:column; gap:4px;"></div>
            `;
            list.appendChild(card);
            card.querySelector('.g-add-or').addEventListener('click', () => { addConstraintRow(card); updateExprPreview(); });
            card.querySelector('.g-remove').addEventListener('click', () => { card.remove(); refreshGroupLabels(); updateExprPreview(); });
            const rows = initialRows && initialRows.length ? initialRows : [{}];
            for (const r of rows) addConstraintRow(card, r.position, r.value);
        }

        function refreshGroupLabels() {
            const cards = document.querySelectorAll('#groupsList .group-card');
            cards.forEach((c, i) => {
                const span = c.querySelector('span');
                if (!span) return;
                span.innerHTML = `${i === 0 ? 'GROUP 1' : 'AND GROUP ' + (i + 1)} <span style="color:#8b949e; font-weight:normal;">(rows OR-ed)</span>`;
            });
        }

        function addConstraintRow(card, pos, val) {
            const rows = card.querySelector('.g-rows');
            const row = document.createElement('div');
            row.className = 'constraint-row';
            row.style.cssText = 'display:flex; gap:8px; align-items:center;';
            row.innerHTML = `
                <span style="color:#6e7681; font-size:11px; min-width:24px;">${rows.children.length === 0 ? '' : 'OR'}</span>
                <select class="c-pos" style="background:#21262d; color:#c9d1d9; border:1px solid #30363d; padding:6px; border-radius:4px;">
                    <option value="front">Front (starts with)</option>
                    <option value="back">Back (ends with)</option>
                    <option value="any">Any (contains)</option>
                </select>
                <input class="c-val" type="text" maxlength="16" placeholder="hex (max 16, lowercase)" style="flex:1; min-width:150px; background:#0d1117; color:#c9d1d9; border:1px solid #30363d; padding:6px 8px; border-radius:4px; font-family:ui-monospace,monospace;">
                <button class="c-remove" style="background:#6e7681; color:white; border:none; padding:6px 10px; border-radius:4px; cursor:pointer; font-size:11px;">×</button>
            `;
            rows.appendChild(row);
            const sel = row.querySelector('.c-pos');
            const inp = row.querySelector('.c-val');
            if (pos) sel.value = pos;
            if (val) inp.value = val;
            inp.addEventListener('input', () => {
                inp.value = inp.value.toLowerCase().replace(/[^0-9a-f]/g, '').slice(0, 16);
                updateExprPreview();
            });
            sel.addEventListener('change', updateExprPreview);
            row.querySelector('.c-remove').addEventListener('click', () => {
                row.remove();
                // Refresh "OR" labels in this group.
                rows.querySelectorAll('.constraint-row').forEach((r, i) => {
                    const lbl = r.querySelector('span');
                    if (lbl) lbl.innerText = i === 0 ? '' : 'OR';
                });
                updateExprPreview();
            });
        }

        function collectGroups() {
            const cards = document.querySelectorAll('#groupsList .group-card');
            const groups = [];
            for (const card of cards) {
                const rows = card.querySelectorAll('.constraint-row');
                const group = [];
                for (const r of rows) {
                    const position = r.querySelector('.c-pos').value;
                    const value = r.querySelector('.c-val').value.trim().toLowerCase();
                    if (!value) continue;
                    if (!/^[0-9a-f]{1,16}$/.test(value)) {
                        alert(`Invalid constraint "${value}". Must be 1-16 lowercase hex chars.`);
                        return null;
                    }
                    group.push({ position, value });
                }
                if (group.length > 0) groups.push(group);
            }
            if (groups.length === 0) {
                alert('Add at least one constraint or disable Custom Search.');
                return null;
            }
            return groups;
        }

        function updateExprPreview() {
            const groups = [];
            const cards = document.querySelectorAll('#groupsList .group-card');
            for (const card of cards) {
                const rows = card.querySelectorAll('.constraint-row');
                const parts = [];
                for (const r of rows) {
                    const position = r.querySelector('.c-pos').value;
                    const value = r.querySelector('.c-val').value.trim().toLowerCase();
                    if (!value) continue;
                    parts.push(`${position}:${value}`);
                }
                if (parts.length === 0) continue;
                groups.push(parts.length === 1 ? parts[0] : '(' + parts.join(' OR ') + ')');
            }
            const el = document.getElementById('exprPreview');
            el.innerText = groups.length ? 'Expression: ' + groups.join(' AND ') : '';
        }

        function submitMatch(matchId) { ws.send(JSON.stringify({ command: 'submit_match', match_id: matchId })); }

        function saveMatchFile(matchId) {
            const match = historyData.find(d => d.match_id === matchId);
            if (!match) return;
            const data = {
                match_id: match.match_id,
                origin_hash: match.origin_hash,
                iterations: match.iterations,
                peak: match.peak,
                bin: match.bin,
                rle: computeRLE(match.bin),
            };
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `conway_${match.origin_hash.substring(0, 16)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

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
                
                let actionBtn;
                if (d.custom) {
                    actionBtn = `<button onclick="saveMatchFile('${d.match_id}')" style="background: #6e40c9; color: white; border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 11px;">Save File</button>`;
                } else if (d.uploaded) {
                    actionBtn = `<span style="color: #3fb950; font-size: 11px;">Sent</span>`;
                } else {
                    actionBtn = `<button onclick="submitMatch('${d.match_id}')" style="background: #1f6feb; color: white; border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 11px;">Send</button>`;
                }
                let inspectBtn = `<button onclick="inspectMatch('${d.match_id}')" style="background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 11px; margin-left: 5px;">Inspect</button>`;
                let actionCell = `<div style="display: flex; align-items: center;">${actionBtn}${inspectBtn}</div>`;
                
                row.innerHTML = `<td style="padding: 8px; color: #8b949e;">${i++}</td><td style="padding: 8px; font-family: monospace; color: #c9d1d9;">${d.origin_hash.substring(0, 24)}…</td><td style="padding: 8px; color: #d2a8ff; font-weight: bold;">${d.iterations}</td><td style="padding: 8px;">${d.peak}</td><td style="padding: 8px; color: #8b949e;">${tStr}</td><td style="padding: 8px;">${actionCell}</td>`;
                tbody.appendChild(row);
            }
            document.getElementById('kept') && (document.getElementById('kept').innerText = historyData.length);
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
                historyData.push({ match_id: data.match_id, origin_hash: data.origin_hash, iterations: data.iterations, peak: data.peak, uploaded: data.uploaded, time: data.time || Date.now(), bin: data.bin, custom: !!data.custom });
                if (historyData.length > 500) {
                    const uploaded = historyData.filter(d => d.uploaded), others = historyData.filter(d => !d.uploaded);
                    others.sort((a, b) => b.iterations !== a.iterations ? b.iterations - a.iterations : (b.peak !== a.peak ? b.peak - a.peak : b.time - a.time));
                    historyData = uploaded.concat(others.slice(0, Math.max(50, 500 - uploaded.length)));
                }
                scheduleRender();
            } else if (data.type === 'match_history_batch') {
                const now = Date.now();
                const agg = data.agg;
                if (agg) {
                    if (agg.max_iters > bestIters) bestIters = agg.max_iters;
                    if (agg.hist_delta) {
                        for (const k in agg.hist_delta) {
                            iterHistData[k] = (iterHistData[k] || 0) + agg.hist_delta[k];
                        }
                    }
                } else {
                    for (const m of data.items) {
                        if (m.iterations > bestIters) bestIters = m.iterations;
                        const bucket = Math.floor(m.iterations / 100) * 100;
                        iterHistData[bucket] = (iterHistData[bucket] || 0) + 1;
                    }
                }
                for (const m of data.items) {
                    historyData.push({ match_id: m.match_id, origin_hash: m.origin_hash, iterations: m.iterations, peak: m.peak, uploaded: m.uploaded, time: m.time || now, bin: m.bin, custom: !!m.custom });
                }
                document.getElementById('bestIters').innerText = bestIters;
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
                ['btnStart','btnStop','suiteSelect','blocks','threads','ipt','workers','customMode'].forEach(id => document.getElementById(id).disabled = id==='btnStop' ? !r : r);
                document.querySelectorAll('input[name="customKind"]').forEach(el => el.disabled = r);
                ['regexPattern','regexFlags','regexAnchor'].forEach(id => { const el = document.getElementById(id); if (el) el.disabled = r; });
            }
        };

        ws.onopen = () => logToConsole("Connected to GPU backend");
        ws.onclose = () => logToConsole("Disconnected. Refresh to reconnect.");

        // Live regex tester: re-run on any input change in the regex panel.
        ['regexPattern', 'regexFlags', 'regexAnchor', 'regexTestInput'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('input', runRegexTest);
        });
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


async def _match_flusher():
    """
    Coalesce buffered match-history items and broadcast them at a bounded rate.

    Sends at most ~5 messages/sec. Each message contains the top-N items by
    iterations (rest are dropped from the UI feed but remain in pending_payloads),
    plus aggregate stats so the histogram and best-iters counter stay accurate.
    """
    TOP_N = 50
    INTERVAL = 0.2
    try:
        while True:
            await asyncio.sleep(INTERVAL)
            buf = state.match_buffer
            if not buf:
                continue
            state.match_buffer = []
            total = len(buf)
            max_iters = 0
            hist_delta: dict = {}
            for m in buf:
                it = m["iterations"]
                if it > max_iters:
                    max_iters = it
                bucket = (it // 100) * 100
                hist_delta[bucket] = hist_delta.get(bucket, 0) + 1
            if total > TOP_N:
                buf.sort(key=lambda m: m["iterations"], reverse=True)
                items = buf[:TOP_N]
            else:
                items = buf
            await broadcast({
                "type": "match_history_batch",
                "items": items,
                "agg": {"count": total, "max_iters": max_iters, "hist_delta": hist_delta},
            })
    except asyncio.CancelledError:
        return

async def broadcast_log(msg: str):
    print(msg)
    await broadcast({"type": "log", "message": msg})

async def broadcast_state():
    is_running = state.miner is not None and getattr(state.miner, 'is_running', False)
    await broadcast({"type": "state_update", "is_running": is_running})

async def submit_payload(payload: dict) -> bool:
    def post_data():
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-GB,en;q=0.8",
            "Origin": "https://lifehashes.net",
            "Referer": "https://lifehashes.net/dailychallenge/",
            "sec-ch-ua": '"Brave";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1"
        }
        req = urllib.request.Request(
            "https://lifehashes.net/dailychallenge/save_glyph.php",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
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
                custom_mode = bool(msg.get("custom_mode", False))
                groups = msg.get("groups") or []
                regex_payload = msg.get("regex") or None
                use_regex = bool(custom_mode and regex_payload and regex_payload.get("pattern"))
                if custom_mode:
                    if use_regex:
                        if groups:
                            await broadcast_log("Custom mode: provide either groups OR regex, not both.")
                            continue
                        pattern = str(regex_payload.get("pattern", ""))
                        flags_str = str(regex_payload.get("flags", "")).lower()
                        anchor = str(regex_payload.get("anchor", "")).lower().strip()
                        if not pattern or len(pattern) > 256:
                            await broadcast_log("Regex pattern must be 1-256 chars.")
                            continue
                        if any(ch not in "ismx" for ch in flags_str if ch.strip()):
                            await broadcast_log("Regex flags: only i, s, m, x allowed.")
                            continue
                        if not anchor or len(anchor) > 16 or any(ch not in "0123456789abcdef" for ch in anchor):
                            await broadcast_log("Regex anchor must be 1-16 lowercase hex chars.")
                            continue
                        try:
                            import re as _re
                            _re.compile(pattern)
                        except _re.error as e:
                            await broadcast_log(f"Invalid regex: {e}")
                            continue
                    else:
                        if not isinstance(groups, list) or not groups:
                            await broadcast_log("Custom mode requires at least one group (or a regex).")
                            continue
                        if len(groups) > 16:
                            await broadcast_log("At most 16 groups supported.")
                            continue
                        bad = False
                        total = 0
                        for gid, g in enumerate(groups):
                            if not isinstance(g, list) or not g:
                                await broadcast_log(f"Group {gid} must be a non-empty list.")
                                bad = True; break
                            for c in g:
                                v = str(c.get("value", "")).lower()
                                p = str(c.get("position", "any")).lower()
                                if p not in ("front", "back", "any"):
                                    await broadcast_log(f"Invalid position {p!r}.")
                                    bad = True; break
                                if not v or len(v) > 16 or any(ch not in "0123456789abcdef" for ch in v):
                                    await broadcast_log(f"Invalid constraint value {v!r}.")
                                    bad = True; break
                                total += 1
                            if bad: break
                        if bad: continue
                        if total > 64:
                            await broadcast_log("At most 64 constraints supported in total.")
                            continue
                state.custom_mode = custom_mode
                
                async def on_log(m: str): await broadcast_log(m)
                async def on_rate(r: float, hr: float, q: int, d: int): await broadcast({"type": "rate", "rate": r, "hits_rate": hr, "queue": q, "dropped": d})
                async def on_hit_count(total: int): await broadcast({"type": "hit_count", "total": total})
                async def on_match_history(mid: str, h: str, iters: int, peak: int, p: dict):
                    state.pending_payloads[mid] = p
                    state.match_buffer.append({"match_id": mid, "origin_hash": h, "iterations": iters, "peak": peak, "uploaded": False, "time": int(time.time() * 1000), "bin": p.get("bin"), "custom": state.custom_mode})
                async def on_match_history_batch(items):
                    now_ms = int(time.time() * 1000)
                    for mid, h, iters, peak, p in items:
                        state.pending_payloads[mid] = p
                        state.match_buffer.append({"match_id": mid, "origin_hash": h, "iterations": iters, "peak": peak, "uploaded": False, "time": now_ms, "bin": p.get("bin"), "custom": state.custom_mode})
                async def on_stop():
                    state.miner = None
                    if state.match_flush_task:
                        state.match_flush_task.cancel()
                        state.match_flush_task = None
                    await broadcast_state()
                
                state.miner = GPUMiner(callbacks={
                    'on_log': on_log, 'on_rate': on_rate, 'on_hit_count': on_hit_count, 
                    'on_match_history': on_match_history, 'on_match_history_batch': on_match_history_batch, 'on_stop': on_stop
                })
                
                state.miner.is_running = True
                await broadcast_state()
                if state.match_flush_task is None or state.match_flush_task.done():
                    state.match_flush_task = asyncio.create_task(_match_flusher())
                state.miner_task = asyncio.create_task(
                    state.miner.run(
                        target_suite, blocks, threads, ipt, workers,
                        custom_mode,
                        [] if use_regex else groups,
                        regex_payload if use_regex else None,
                    )
                )

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
                if str(payload.get("nist_pulse_id", "")).startswith("custom:"):
                    await broadcast_log("Submit blocked: custom-mode matches cannot be submitted, use Save File instead.")
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