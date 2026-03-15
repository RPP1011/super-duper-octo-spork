/* Game Viewer — Canvas renderer + WebSocket client + controls */

const canvas = document.getElementById('battlefield');
const ctx = canvas.getContext('2d');

// --- State ---
let gameState = null;
let roomWidth = 30;
let roomDepth = 20;
let isPaused = false;
let isRunning = false;
let currentFormation = 'Hold';
let currentFocusTarget = null;
let allEvents = [];

// --- Personality presets ---
const PRESETS = {
  aggressive:  { aggression: 0.9, risk_tolerance: 0.8, discipline: 0.3, control_bias: 0.2, altruism: 0.3, patience: 0.2 },
  defensive:   { aggression: 0.2, risk_tolerance: 0.2, discipline: 0.9, control_bias: 0.5, altruism: 0.8, patience: 0.9 },
  balanced:    { aggression: 0.5, risk_tolerance: 0.5, discipline: 0.5, control_bias: 0.5, altruism: 0.5, patience: 0.5 },
  cc_heavy:    { aggression: 0.4, risk_tolerance: 0.4, discipline: 0.7, control_bias: 0.9, altruism: 0.5, patience: 0.6 },
};

// --- WebSocket ---
let ws = null;

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    handleStateMessage(msg);
  };

  ws.onclose = () => {
    setTimeout(connectWS, 2000);
  };

  ws.onerror = () => ws.close();
}

function handleStateMessage(msg) {
  gameState = msg;

  if (msg.type === 'state') {
    roomWidth = msg.room_width || 30;
    roomDepth = msg.room_depth || 20;
    isRunning = true;
  } else if (msg.type === 'done') {
    isRunning = false;
  }

  // Accumulate events
  if (msg.recent_events) {
    for (const ev of msg.recent_events) {
      allEvents.push(ev);
    }
    // Cap at 500
    if (allEvents.length > 500) allEvents = allEvents.slice(-500);
  }

  renderCanvas();
  updateUnitTable();
  updateEventLog();
  updateStatusSummary();
  updateFocusButtons();
  updateControlState();
}

// --- Canvas rendering ---

function renderCanvas() {
  if (!gameState) return;
  const units = gameState.units || [];

  const W = canvas.width;
  const H = canvas.height;
  const pad = 30;

  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = '#1a1a2e';
  ctx.lineWidth = 0.5;
  const scaleX = (W - pad * 2) / roomWidth;
  const scaleY = (H - pad * 2) / roomDepth;
  for (let x = 0; x <= roomWidth; x += 5) {
    const px = pad + x * scaleX;
    ctx.beginPath(); ctx.moveTo(px, pad); ctx.lineTo(px, H - pad); ctx.stroke();
  }
  for (let y = 0; y <= roomDepth; y += 5) {
    const py = pad + y * scaleY;
    ctx.beginPath(); ctx.moveTo(pad, py); ctx.lineTo(W - pad, py); ctx.stroke();
  }

  // Units
  const radius = Math.min(scaleX, scaleY) * 0.6;

  for (const u of units) {
    const px = pad + u.position[0] * scaleX;
    const py = pad + u.position[1] * scaleY;
    const isHero = u.team === 'Hero';
    const baseColor = isHero ? '#3498db' : '#e74c3c';
    const fillColor = isHero ? '#2471a3' : '#c0392b';

    // Casting indicator
    if (u.is_casting) {
      ctx.beginPath();
      ctx.arc(px, py, radius + 6, 0, Math.PI * 2);
      ctx.strokeStyle = '#f1c40f';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Focus indicator
    if (u.id === currentFocusTarget) {
      ctx.beginPath();
      ctx.arc(px, py, radius + 10, 0, Math.PI * 2);
      ctx.strokeStyle = '#e67e22';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Body
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = baseColor;
    ctx.lineWidth = 2;
    ctx.stroke();

    // HP bar
    const barW = radius * 2.2;
    const barH = 4;
    const barX = px - barW / 2;
    const barY = py - radius - 10;
    const hpPct = u.hp / u.max_hp;
    ctx.fillStyle = '#333';
    ctx.fillRect(barX, barY, barW, barH);
    ctx.fillStyle = hpPct > 0.5 ? '#2ecc71' : hpPct > 0.25 ? '#f39c12' : '#e74c3c';
    ctx.fillRect(barX, barY, barW * hpPct, barH);

    // ID
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(u.id, px, py);

    // Role label
    ctx.fillStyle = '#aaa';
    ctx.font = '9px sans-serif';
    ctx.fillText(u.role, px, py + radius + 10);
  }

  // Title
  ctx.fillStyle = '#666';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  const tick = gameState.tick || 0;
  const typeLabel = gameState.type === 'done' ? `DONE — Winner: ${gameState.winner}` : `Tick ${tick}`;
  ctx.fillText(typeLabel, 8, 8);
}

// --- Unit table ---

function updateUnitTable() {
  const tbody = document.querySelector('#unit-table tbody');
  if (!gameState) return;

  const units = gameState.units || [];
  let html = '';
  for (const u of units) {
    const cls = u.team === 'Hero' ? 'hero' : 'enemy';
    const status = u.is_casting ? 'CASTING' :
      u.control_remaining_ms > 0 ? 'CC' : '';
    html += `<tr class="${cls}">` +
      `<td>${u.id}</td>` +
      `<td>${u.team}</td>` +
      `<td>${u.role}</td>` +
      `<td>${u.hp}/${u.max_hp}</td>` +
      `<td>${u.position[0].toFixed(1)},${u.position[1].toFixed(1)}</td>` +
      `<td>${status}</td>` +
      `</tr>`;
  }

  // For done state, show final stats
  if (gameState.type === 'done') {
    html += `<tr><td colspan="6" style="text-align:center;color:#f1c40f;padding:8px;">` +
      `WINNER: ${gameState.winner} | Hero alive: ${gameState.hero_alive} | Enemy alive: ${gameState.enemy_alive}` +
      `</td></tr>`;
  }

  tbody.innerHTML = html;
}

// --- Event log ---

function updateEventLog() {
  const log = document.getElementById('event-log');
  // Only render last 100
  const recent = allEvents.slice(-100);
  let html = '';
  for (const ev of recent) {
    const cls = `ev-${ev.kind}`;
    let text = `T${ev.tick}: `;
    if (ev.kind === 'damage') {
      text += `Unit ${ev.unit_id} dealt ${ev.amount} damage to Unit ${ev.target_id}`;
    } else if (ev.kind === 'heal') {
      text += `Unit ${ev.unit_id} healed Unit ${ev.target_id} for ${ev.amount}`;
    } else if (ev.kind === 'death') {
      text += `Unit ${ev.unit_id} died`;
    } else if (ev.kind === 'control') {
      text += `Unit ${ev.unit_id} CC'd Unit ${ev.target_id}`;
    } else {
      text += `${ev.kind} (${ev.unit_id})`;
    }
    html += `<div class="${cls}">${text}</div>`;
  }
  log.innerHTML = html;
  log.scrollTop = log.scrollHeight;
}

// --- Status summary (text for Claude's get_page_text) ---

function updateStatusSummary() {
  const el = document.getElementById('status-summary');
  if (!gameState) {
    el.textContent = 'No game running';
    return;
  }

  if (gameState.type === 'done') {
    el.textContent = `GAME OVER | Winner: ${gameState.winner} | Tick: ${gameState.tick} | ` +
      `Hero alive: ${gameState.hero_alive} (${gameState.hero_hp_total} HP) | ` +
      `Enemy alive: ${gameState.enemy_alive} (${gameState.enemy_hp_total} HP)`;
    return;
  }

  const units = gameState.units || [];
  const heroes = units.filter(u => u.team === 'Hero');
  const enemies = units.filter(u => u.team === 'Enemy');
  const heroAvgHp = heroes.length ? Math.round(heroes.reduce((s, u) => s + u.hp_pct, 0) / heroes.length * 100) : 0;
  const enemyAvgHp = enemies.length ? Math.round(enemies.reduce((s, u) => s + u.hp_pct, 0) / enemies.length * 100) : 0;
  const statusLabel = isPaused ? 'PAUSED' : 'RUNNING';

  el.textContent = `Tick ${gameState.tick} | ${statusLabel} | ` +
    `Hero: ${heroes.length} alive (${heroAvgHp}% avg HP) | ` +
    `Enemy: ${enemies.length} alive (${enemyAvgHp}% avg HP) | ` +
    `Formation: ${currentFormation} | Focus: ${currentFocusTarget ?? 'none'}`;
}

// --- Focus target buttons ---

function updateFocusButtons() {
  if (!gameState || !gameState.units) return;
  const container = document.getElementById('focus-buttons');
  const enemies = (gameState.units || []).filter(u => u.team === 'Enemy');

  // Keep clear button, rebuild enemy buttons
  let html = '<button id="btn-clear-focus">CLEAR FOCUS</button>';
  for (const e of enemies) {
    const active = currentFocusTarget === e.id ? ' active' : '';
    html += `<button class="focus-btn${active}" data-target="${e.id}">FOCUS ${e.id} (${e.role})</button>`;
  }
  container.innerHTML = html;

  // Re-bind
  document.getElementById('btn-clear-focus').onclick = () => {
    currentFocusTarget = null;
    sendDecision();
    updateFocusButtons();
  };
  for (const btn of container.querySelectorAll('.focus-btn')) {
    btn.onclick = () => {
      currentFocusTarget = parseInt(btn.dataset.target);
      sendDecision();
      updateFocusButtons();
    };
  }
}

// --- Control state ---

function updateControlState() {
  const statusEl = document.getElementById('game-status');
  const pauseBtn = document.getElementById('btn-pause');
  const stepBtn = document.getElementById('btn-step');

  if (gameState && gameState.type === 'done') {
    statusEl.textContent = `DONE — ${gameState.winner} wins`;
    statusEl.style.color = '#f1c40f';
    pauseBtn.disabled = true;
    stepBtn.disabled = true;
    isRunning = false;
  } else if (isRunning) {
    statusEl.textContent = isPaused ? 'PAUSED' : 'RUNNING';
    statusEl.style.color = isPaused ? '#f39c12' : '#2ecc71';
    pauseBtn.disabled = false;
    pauseBtn.textContent = isPaused ? 'RESUME' : 'PAUSE';
    stepBtn.disabled = !isPaused;
  } else {
    statusEl.textContent = 'IDLE';
    statusEl.style.color = '#aaa';
    pauseBtn.disabled = true;
    stepBtn.disabled = true;
  }
}

// --- API helpers ---

async function apiPost(path, body = {}) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return res.json();
}

async function sendDecision() {
  const decision = {
    squad_overrides: {
      Hero: {
        mode: currentFormation,
        focus_target: currentFocusTarget,
      }
    }
  };
  await apiPost('/api/decision', decision);
}

// --- Event bindings ---

document.getElementById('btn-start').onclick = async () => {
  const scenario = document.getElementById('scenario-select').value;
  const seedEl = document.getElementById('seed-input');
  const intervalEl = document.getElementById('interval-input');
  const seed = seedEl.value ? parseInt(seedEl.value) : undefined;
  const interval = parseInt(intervalEl.value) || 30;

  allEvents = [];
  currentFormation = 'Hold';
  currentFocusTarget = null;
  isPaused = false;

  // Reset formation button states
  document.querySelectorAll('.formation-btn').forEach(b => b.classList.remove('active'));
  document.querySelector('.formation-btn[data-mode="Hold"]').classList.add('active');

  const goap = document.getElementById('goap-checkbox').checked;
  const body = { scenario, decision_interval: interval, goap };
  if (seed !== undefined) body.seed = seed;
  await apiPost('/api/start', body);
  isRunning = true;
  updateControlState();
};

document.getElementById('btn-pause').onclick = async () => {
  if (isPaused) {
    await apiPost('/api/resume');
    isPaused = false;
  } else {
    await apiPost('/api/pause');
    isPaused = true;
  }
  updateControlState();
};

document.getElementById('btn-step').onclick = async () => {
  await apiPost('/api/step');
};

document.getElementById('speed-select').onchange = (ev) => {
  apiPost('/api/speed', { interval_ms: parseInt(ev.target.value) });
};

// Formation buttons
document.querySelectorAll('.formation-btn').forEach(btn => {
  btn.onclick = () => {
    currentFormation = btn.dataset.mode;
    document.querySelectorAll('.formation-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    sendDecision();
  };
});

// Preset buttons
document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.onclick = async () => {
    const preset = PRESETS[btn.dataset.preset];
    if (!gameState || !gameState.units) return;
    const heroes = gameState.units.filter(u => u.team === 'Hero');
    const personalityUpdates = {};
    for (const h of heroes) {
      personalityUpdates[String(h.id)] = { ...preset };
    }
    await apiPost('/api/decision', {
      personality_updates: personalityUpdates,
      squad_overrides: {
        Hero: {
          mode: currentFormation,
          focus_target: currentFocusTarget,
        }
      }
    });
  };
});

// --- Load scenarios ---

async function loadScenarios() {
  const res = await fetch('/api/scenarios');
  const data = await res.json();
  const select = document.getElementById('scenario-select');
  select.innerHTML = '';
  for (const s of data.scenarios) {
    const opt = document.createElement('option');
    opt.value = s;
    opt.textContent = s.replace('.toml', '');
    select.appendChild(opt);
  }
}

// --- Init ---
loadScenarios();
connectWS();
updateControlState();
