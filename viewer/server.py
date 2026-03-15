"""Game viewer server — bridges browser UI to sim_bridge subprocess."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

GAME_ROOT = Path(__file__).resolve().parent.parent
SIM_BINARY = GAME_ROOT / "target" / "debug" / "sim_bridge"
SCENARIOS_DIR = GAME_ROOT / "scenarios"


class SimSession:
    """Manages a sim_bridge subprocess and auto-stepping."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.state: dict | None = None
        self.done: bool = False
        self.paused: bool = False
        self.step_interval: float = 0.5  # seconds between auto-steps
        self._step_task: asyncio.Task | None = None
        self._ws_clients: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def start(self, scenario: str, seed: int | None = None,
                    decision_interval: int = 30, ticks: int = 3200,
                    goap: bool = False) -> dict:
        await self.close()
        self.done = False
        self.paused = False

        scenario_path = SCENARIOS_DIR / scenario
        if not scenario_path.exists():
            return {"error": f"Scenario not found: {scenario}"}

        self.proc = subprocess.Popen(
            [str(SIM_BINARY)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )

        init_msg = {
            "type": "init",
            "scenario": str(scenario_path),
            "ticks": ticks,
            "decision_interval": decision_interval,
            "goap": goap,
        }
        if seed is not None:
            init_msg["seed"] = seed

        self._send(init_msg)
        self.state = self._recv()

        if self.state and self.state.get("type") == "done":
            self.done = True

        return self.state or {"error": "No response from sim"}

    def _send(self, msg: dict):
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()

    def _recv(self) -> dict | None:
        if self.proc and self.proc.stdout:
            line = self.proc.stdout.readline()
            if line:
                return json.loads(line)
        return None

    async def step(self, decision: dict | None = None) -> dict | None:
        """Send a decision (or no-op) and get next state."""
        if self.done or not self.proc:
            return self.state

        async with self._lock:
            wire = {"type": "decision"}
            if decision:
                if "personality_updates" in decision:
                    wire["personality_updates"] = decision["personality_updates"]
                if "squad_overrides" in decision:
                    wire["squad_overrides"] = decision["squad_overrides"]

            loop = asyncio.get_event_loop()
            self._send(wire)
            self.state = await loop.run_in_executor(None, self._recv)

            if self.state and self.state.get("type") == "done":
                self.done = True

            await self._broadcast(self.state)
            return self.state

    async def send_decision(self, decision: dict) -> dict | None:
        """Send a decision with squad/personality overrides."""
        return await self.step(decision)

    async def _broadcast(self, msg: dict | None):
        if not msg:
            return
        dead = []
        for ws in self._ws_clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._ws_clients.remove(ws)

    def add_ws(self, ws: WebSocket):
        self._ws_clients.append(ws)

    def remove_ws(self, ws: WebSocket):
        if ws in self._ws_clients:
            self._ws_clients.remove(ws)

    async def start_auto_step(self):
        self.paused = False
        if self._step_task is None or self._step_task.done():
            self._step_task = asyncio.create_task(self._auto_step_loop())

    async def _auto_step_loop(self):
        while not self.done:
            if self.paused:
                await asyncio.sleep(0.1)
                continue
            await self.step()
            await asyncio.sleep(self.step_interval)

    async def close(self):
        if self._step_task and not self._step_task.done():
            self._step_task.cancel()
            try:
                await self._step_task
            except asyncio.CancelledError:
                pass
            self._step_task = None

        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        self.state = None
        self.done = False


session = SimSession()


# --- Static files ---

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


# --- REST API ---

@app.get("/api/scenarios")
async def list_scenarios():
    files = sorted(
        str(p.relative_to(SCENARIOS_DIR))
        for p in SCENARIOS_DIR.rglob("*.toml")
        if not p.name.startswith(".")
    )
    return {"scenarios": files}


@app.post("/api/start")
async def start_game(body: dict):
    scenario = body.get("scenario", "basic_4v4.toml")
    seed = body.get("seed")
    interval = body.get("decision_interval", 30)
    ticks = body.get("ticks", 3200)
    goap = body.get("goap", False)
    state = await session.start(scenario, seed, interval, ticks, goap=goap)
    await session.start_auto_step()
    # Broadcast initial state to WebSocket clients
    await session._broadcast(state)
    return state


@app.post("/api/decision")
async def send_decision(body: dict):
    result = await session.send_decision(body)
    return result or {"error": "No active session"}


@app.post("/api/step")
async def step_game():
    result = await session.step()
    return result or {"error": "No active session"}


@app.post("/api/pause")
async def pause_game():
    session.paused = True
    return {"status": "paused", "state": session.state}


@app.post("/api/resume")
async def resume_game():
    session.paused = False
    return {"status": "running"}


@app.post("/api/speed")
async def set_speed(body: dict):
    ms = body.get("interval_ms", 500)
    session.step_interval = max(50, ms) / 1000.0
    return {"interval_ms": int(session.step_interval * 1000)}


# --- WebSocket ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session.add_ws(ws)
    # Send current state if available
    if session.state:
        await ws.send_json(session.state)
    try:
        while True:
            # Keep connection alive; client doesn't send meaningful data
            await ws.receive_text()
    except WebSocketDisconnect:
        session.remove_ws(ws)
