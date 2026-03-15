#!/usr/bin/env python3
"""Dashboard server for IMPALA training monitoring + scenario replay."""
import csv
import json
import os
import re
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent.parent.parent
GENERATED = ROOT / "generated"

# ── Episode index cache ──────────────────────────────────────────────────────
# Avoids re-parsing 1GB+ JSONL files on every request.
# Maps (exp, phase) -> { mtime, index: [{offset, scenario, outcome, reward, ticks, n_steps, ability_names}] }
_index_cache = {}


def _build_index(ep_path):
    """Build a lightweight index of episode metadata + byte offsets."""
    key = str(ep_path)
    mtime = ep_path.stat().st_mtime
    if key in _index_cache and _index_cache[key]["mtime"] == mtime:
        return _index_cache[key]["index"]

    print(f"  Indexing {ep_path.name} ({ep_path.stat().st_size / 1e6:.0f} MB)...")
    t0 = time.time()
    index = []
    with open(ep_path, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            # Fast partial parse: extract only top-level fields without parsing steps
            # We use a regex to grab scenario/outcome/reward/ticks without full JSON parse
            try:
                # Parse just enough to get metadata. Use raw string scanning for speed.
                line_str = line.decode("utf-8", errors="replace")
                # Extract fields at the start of the JSON object (before "steps")
                scenario = _extract_str(line_str, "scenario")
                outcome = _extract_str(line_str, "outcome")
                reward = _extract_num(line_str, "reward")
                ticks = _extract_int(line_str, "ticks")
                # Count steps by counting "tick": occurrences (each step has one)
                n_steps = line_str.count('"tick":') - 1  # subtract the top-level "ticks"
                if n_steps < 0:
                    n_steps = 0
                # Extract unit_ability_names (small, at start of JSON)
                abn_match = re.search(r'"unit_ability_names"\s*:\s*(\{[^}]*\})', line_str)
                ability_names = json.loads(abn_match.group(1)) if abn_match else {}

                index.append({
                    "offset": offset,
                    "length": len(line),
                    "scenario": scenario,
                    "outcome": outcome,
                    "reward": reward,
                    "ticks": ticks,
                    "n_steps": n_steps,
                    "unit_ability_names": ability_names,
                })
            except Exception as e:
                print(f"  Warning: failed to index line at offset {offset}: {e}")
                continue

    _index_cache[key] = {"mtime": mtime, "index": index}
    print(f"  Indexed {len(index)} episodes in {time.time() - t0:.1f}s")
    return index


def _extract_str(s, key):
    m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', s)
    return m.group(1) if m else ""


def _extract_num(s, key):
    m = re.search(rf'"{key}"\s*:\s*([-\d.eE+]+)', s)
    return float(m.group(1)) if m else 0.0


def _extract_int(s, key):
    m = re.search(rf'"{key}"\s*:\s*(\d+)', s)
    return int(m.group(1)) if m else 0


def _read_episode_at(ep_path, offset, length):
    """Read and parse a single episode by byte offset."""
    with open(ep_path, "rb") as f:
        f.seek(offset)
        line = f.read(length)
    return json.loads(line)


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/api/curriculum":
            exp = qs.get("exp", [None])[0]
            if exp:
                self._json_response(self._get_curriculum(exp))
            else:
                self._error(400, "Need exp param")
        elif path == "/api/experiments":
            self._json_response(self._list_experiments())
        elif path == "/api/training":
            exp = qs.get("exp", [None])[0]
            phase = qs.get("phase", [None])[0]
            if exp and phase:
                self._json_response(self._get_training(exp, phase))
            else:
                self._error(400, "Need exp and phase params")
        elif path == "/api/logs":
            exp = qs.get("exp", [None])[0]
            if exp:
                self._json_response(self._parse_log(exp))
            else:
                self._error(400, "Need exp param")
        elif path == "/api/episodes":
            exp = qs.get("exp", [None])[0]
            phase = qs.get("phase", [None])[0]
            scenario = qs.get("scenario", [None])[0]
            limit = int(qs.get("limit", ["20"])[0])
            if exp and phase:
                self._json_response(self._get_episodes(exp, phase, scenario, limit))
            else:
                self._error(400, "Need exp and phase params")
        elif path == "/api/episode":
            exp = qs.get("exp", [None])[0]
            phase = qs.get("phase", [None])[0]
            idx = int(qs.get("idx", ["0"])[0])
            if exp and phase:
                self._json_response(self._get_episode(exp, phase, idx))
            else:
                self._error(400, "Need exp and phase params")
        elif path == "/api/attention":
            exp = qs.get("exp", [None])[0]
            phase = qs.get("phase", [None])[0]
            idx = int(qs.get("idx", ["0"])[0])
            weights = qs.get("weights", [None])[0]
            if exp and phase:
                self._json_response(self._get_attention(exp, phase, idx, weights))
            else:
                self._error(400, "Need exp, phase params")
        elif path == "/api/scenarios":
            exp = qs.get("exp", [None])[0]
            phase = qs.get("phase", [None])[0]
            if exp and phase:
                self._json_response(self._get_scenarios(exp, phase))
            else:
                self._error(400, "Need exp and phase params")
        else:
            super().do_GET()

    def _json_response(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, msg):
        body = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _list_experiments(self):
        experiments = []
        for d in sorted(GENERATED.iterdir()):
            if d.is_dir() and d.name.startswith("curriculum_"):
                phases = []
                for p in sorted(d.iterdir()):
                    if not p.is_dir():
                        continue
                    # Match "phase*" (IMPALA) or any drill subdir with training data
                    if p.name.startswith("phase") or (p / "training.csv").exists() or (p / "episodes.jsonl").exists():
                        has_csv = (p / "training.csv").exists()
                        has_eps = (p / "episodes.jsonl").exists()
                        phases.append({
                            "name": p.name,
                            "has_csv": has_csv,
                            "has_episodes": has_eps,
                        })
                if phases:
                    experiments.append({"name": d.name, "phases": phases})
        return experiments

    def _get_curriculum(self, exp):
        """Return combined drill progress for a curriculum experiment."""
        exp_dir = GENERATED / exp
        if not exp_dir.is_dir():
            return {"error": f"Experiment {exp} not found", "drills": []}

        drills = []
        for d in sorted(exp_dir.iterdir()):
            if not d.is_dir():
                continue
            csv_path = d / "training.csv"
            if not csv_path.exists():
                continue

            # Parse training CSV for this drill
            rows = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({k: _num(v) for k, v in row.items()})

            # Determine phase from drill name (e.g. "1.1_reach_static" -> 1)
            phase = 0
            try:
                phase = int(d.name.split(".")[0])
            except (ValueError, IndexError):
                pass

            best_wr = max((r.get("win_rate", 0) for r in rows), default=0)
            latest_wr = rows[-1].get("win_rate", 0) if rows else 0
            n_iters = len(rows)

            # Check for best.pt — passed if best.pt exists AND no later drill is training
            # (greedy eval passes at 100/100 but stochastic training WR may be lower)
            passed = (d / "best.pt").exists()
            passed_iter = n_iters if passed else None  # approximate: last iter is when it passed

            drills.append({
                "name": d.name,
                "phase": phase,
                "n_iters": n_iters,
                "best_win_rate": best_wr,
                "latest_win_rate": latest_wr,
                "passed": passed,
                "passed_iter": passed_iter,
                "rows": rows,
            })

        return {"experiment": exp, "drills": drills}

    def _get_training(self, exp, phase):
        csv_path = GENERATED / exp / phase / "training.csv"
        if not csv_path.exists():
            return {"error": "No training.csv found", "rows": []}
        rows = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: _num(v) for k, v in row.items()})
        return {"experiment": exp, "phase": phase, "rows": rows}

    def _parse_log(self, exp):
        """Parse per-epoch metrics from log file."""
        log_path = Path(f"/tmp/{exp}.log")
        if not log_path.exists():
            return {"error": f"No log at {log_path}", "epochs": []}

        epochs = []
        current_iter = 0
        current_phase = ""
        with open(log_path) as f:
            for line in f:
                m = re.match(r"=== Phase (\d+):", line)
                if m:
                    current_phase = f"phase{m.group(1)}"

                m = re.match(r"\s+Iter (\d+):", line)
                if m:
                    current_iter = int(m.group(1))

                m = re.match(
                    r"\s+Epoch (\d+)/(\d+) \((\d+) steps\): "
                    r"pg=([-\d.]+) vl=([-\d.]+) ent=([-\d.]+) kl=([-\d.]+) rew=([-\d.]+)",
                    line,
                )
                if m:
                    epochs.append({
                        "phase": current_phase,
                        "iter": current_iter,
                        "epoch": int(m.group(1)),
                        "max_epochs": int(m.group(2)),
                        "grad_steps": int(m.group(3)),
                        "pg": float(m.group(4)),
                        "vl": float(m.group(5)),
                        "ent": float(m.group(6)),
                        "kl": float(m.group(7)),
                        "rew": float(m.group(8)),
                    })

                m = re.match(r"\s+EVAL:.*Win rate: ([\d.]+)%", line)
                if m:
                    epochs.append({
                        "phase": current_phase,
                        "iter": current_iter,
                        "epoch": 0,
                        "type": "eval",
                        "eval_win": float(m.group(1)) / 100,
                    })

        return {"experiment": exp, "epochs": epochs}

    def _get_episodes(self, exp, phase, scenario, limit):
        ep_path = GENERATED / exp / phase / "episodes.jsonl"
        if not ep_path.exists():
            return {"error": "No episodes.jsonl", "episodes": []}

        index = _build_index(ep_path)
        summaries = []
        for i, entry in enumerate(index):
            if scenario and entry["scenario"] != scenario:
                continue
            summaries.append({
                "idx": i,
                "scenario": entry["scenario"],
                "outcome": entry["outcome"],
                "reward": entry["reward"],
                "ticks": entry["ticks"],
                "n_steps": entry["n_steps"],
                "unit_ability_names": entry["unit_ability_names"],
            })
            if len(summaries) >= limit:
                break
        return {"experiment": exp, "phase": phase, "episodes": summaries}

    def _get_episode(self, exp, phase, idx):
        ep_path = GENERATED / exp / phase / "episodes.jsonl"
        if not ep_path.exists():
            return {"error": "No episodes.jsonl"}

        index = _build_index(ep_path)
        if idx < 0 or idx >= len(index):
            return {"error": f"Episode {idx} not found (have {len(index)})"}

        entry = index[idx]
        return _read_episode_at(ep_path, entry["offset"], entry["length"])

    def _get_scenarios(self, exp, phase):
        ep_path = GENERATED / exp / phase / "episodes.jsonl"
        if not ep_path.exists():
            return {"scenarios": []}

        index = _build_index(ep_path)
        counts = {}
        for entry in index:
            name = entry["scenario"]
            if name not in counts:
                counts[name] = {"wins": 0, "losses": 0, "timeouts": 0}
            o = entry["outcome"]
            if o == "Victory":
                counts[name]["wins"] += 1
            elif o == "Defeat":
                counts[name]["losses"] += 1
            else:
                counts[name]["timeouts"] += 1

        scenarios = []
        for name, c in sorted(counts.items()):
            total = c["wins"] + c["losses"] + c["timeouts"]
            scenarios.append({
                "name": name,
                "wins": c["wins"],
                "losses": c["losses"],
                "timeouts": c["timeouts"],
                "total": total,
                "win_rate": c["wins"] / total if total > 0 else 0,
            })
        return {"scenarios": scenarios}

    def _get_attention(self, exp, phase, idx, weights_path):
        """Extract action distributions from recorded episode data.

        Uses the log-probs already stored in each step (lp_move, lp_combat, lp_pointer)
        plus the action mask to reconstruct what the model was "thinking".
        No model loading required — just episode data analysis.
        """
        ep_path = GENERATED / exp / phase / "episodes.jsonl"
        if not ep_path.exists():
            return {"error": "No episodes.jsonl"}

        index = _build_index(ep_path)
        if idx < 0 or idx >= len(index):
            return {"error": f"Episode {idx} not found"}

        episode = _read_episode_at(ep_path, index[idx]["offset"], index[idx]["length"])
        import math

        results = []
        for step in episode["steps"]:
            step_data = {
                "tick": step["tick"],
                "unit_id": step["unit_id"],
            }

            # Move distribution: 9-way (0-7 dirs + 8 stay)
            # We have lp_move for the chosen action; reconstruct relative probs
            # from action mask and move_dir
            move_dir = step.get("move_dir")
            lp_move = step.get("lp_move")
            if move_dir is not None and lp_move is not None:
                step_data["move_prob"] = math.exp(lp_move)
                step_data["move_dir"] = move_dir

            # Combat distribution
            combat_type = step.get("combat_type")
            lp_combat = step.get("lp_combat")
            if combat_type is not None and lp_combat is not None:
                step_data["combat_prob"] = math.exp(lp_combat)
                step_data["combat_type"] = combat_type

            # Pointer / target
            target_idx = step.get("target_idx")
            lp_pointer = step.get("lp_pointer")
            if target_idx is not None and lp_pointer is not None:
                step_data["target_prob"] = math.exp(lp_pointer)
                step_data["target_idx"] = target_idx

            # Entity-level analysis: distances from self to each entity
            entities = step.get("entities", [])
            types = step.get("entity_types", [])
            if entities and len(entities) > 0:
                self_ent = entities[0]
                distances = []
                for j, e in enumerate(entities):
                    if j >= len(types):
                        break
                    dist = math.sqrt((self_ent[5] - e[5])**2 + (self_ent[6] - e[6])**2) * 20
                    hp = e[0]
                    atk_range = e[13] * 10
                    etype = types[j]
                    distances.append({
                        "idx": j,
                        "type": "self" if etype == 0 else "enemy" if etype == 1 else "ally",
                        "dist": round(dist, 2),
                        "hp": round(hp, 3),
                        "atk_range": round(atk_range, 1),
                        "in_range": dist <= (self_ent[13] * 10) if etype == 1 else None,
                    })
                step_data["entities"] = distances

            # Action mask analysis
            mask = step.get("mask", [])
            if mask:
                step_data["mask"] = mask

            results.append(step_data)

        return results

    def log_message(self, format, *args):
        if "api/" in (args[0] if args else ""):
            return  # quiet API calls
        super().log_message(format, *args)


def _num(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    server = ThreadedHTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"Dashboard: http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
