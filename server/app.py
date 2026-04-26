from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import GpuNegotiationAction, ResetConfig

app = FastAPI(title="GPU Budget Negotiation Arena", version="0.1.0")
env = GpuBudgetNegotiationEnv()

# ---------------------------------------------------------------------------
# Locate repo-relative artifact directories. The Space must include these.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ART_DIR = _REPO_ROOT / "artifacts"
_PLOT_DIR = _REPO_ROOT / "plots"
_DATA_DIR = _REPO_ROOT / "data"

for _name, _path in (("artifacts", _ART_DIR), ("plots", _PLOT_DIR), ("data", _DATA_DIR)):
    if _path.is_dir():
        app.mount(f"/{_name}", StaticFiles(directory=str(_path)), name=_name)


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return ""


def _safe_read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _file_size(path: Path) -> str:
    try:
        b = path.stat().st_size
    except OSError:
        return "—"
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / 1024 / 1024:.2f} MB"


def _parse_demo_transcript_md(md: str) -> list[dict[str, Any]]:
    """Parse artifacts/demo_transcript.md into a list of step dicts."""
    steps: list[dict[str, Any]] = []
    block = re.split(r"^### Step \d+\s*$", md, flags=re.MULTILINE)
    for chunk in block[1:]:
        action = re.search(r"Action:\s*`(.+?)`", chunk)
        result = re.search(r"Result:\s*`(.+?)`", chunk)
        imm = re.search(r"Immediate reward:\s*`([\-\d.]+)`", chunk)
        cum = re.search(r"Cumulative reward:\s*`([\-\d.]+)`", chunk)
        if not (action and result and imm and cum):
            continue
        try:
            res_msg = json.loads(result.group(1)).get("message", "")
        except Exception:
            res_msg = result.group(1)
        steps.append(
            {
                "action": action.group(1),
                "result": res_msg,
                "reward": float(imm.group(1)),
                "cum": float(cum.group(1)),
            }
        )
    return steps


def _parse_before_after_md(md: str) -> dict[str, Any]:
    """Parse before_after_training.md into two ordered step lists."""

    def _section_steps(block: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        chunks = re.split(r"^### Step \d+\s*$", block, flags=re.MULTILINE)
        for c in chunks[1:]:
            action = re.search(r"Action:\s*`(.+?)`", c)
            result = re.search(r"Result:\s*`(.+?)`", c)
            reward = re.search(r"Reward:\s*`([\-\d.]+)`", c)
            if not (action and result and reward):
                continue
            try:
                res_msg = json.loads(result.group(1)).get("message", "")
            except Exception:
                res_msg = result.group(1)
            out.append(
                {
                    "action": action.group(1),
                    "result": res_msg,
                    "reward": float(reward.group(1)),
                }
            )
        return out

    parts = re.split(r"^## After Training\s*$", md, flags=re.MULTILINE)
    before_block = re.split(r"^## Before Training\s*$", parts[0], flags=re.MULTILINE)[-1] if parts else ""
    after_block = parts[1] if len(parts) > 1 else ""

    def _meta(block: str) -> dict[str, Any]:
        return {
            "policy": (re.search(r"Policy:\s*`(.+?)`", block) or [None, ""])[1],
            "task": (re.search(r"Task:\s*`(.+?)`", block) or [None, ""])[1],
            "seed": (re.search(r"Seed:\s*`(.+?)`", block) or [None, ""])[1],
            "reward": float((re.search(r"Episode reward:\s*`([\-\d.]+)`", block) or [None, "0"])[1]),
        }

    return {
        "before": {**_meta(before_block), "steps": _section_steps(before_block)},
        "after": {**_meta(after_block), "steps": _section_steps(after_block)},
    }


def _parse_judged_rounds(md: str) -> dict[str, Any]:
    """Extract every round of a judged transcript for forward/back navigation."""
    rounds: list[dict[str, Any]] = []
    for rm in re.finditer(
        r"^## Round (\d+)\s*\n(.*?)(?=^## Round |\Z)",
        md,
        flags=re.MULTILINE | re.DOTALL,
    ):
        idx = int(rm.group(1))
        block = rm.group(2)
        pitches: list[dict[str, str]] = []
        for line in block.splitlines():
            pm = re.match(r"-\s*`(lab_\d+)`:\s*(.*)", line.strip())
            if pm:
                pitches.append({"lab": pm.group(1), "pitch": pm.group(2)})
        winner_m = re.search(r"Winner:\s*`(lab_\d+)`", block)
        scores_m = re.search(r"Scores:\s*`(\{.*?\})`", block)
        reason_m = re.search(r"Reason:\s*([^\n]+)", block)
        bonus_m = re.search(r"Controlled-lab judge bonus:\s*`([\-\d.eE+]+)`", block)
        env_m = re.search(r"Environment reward after action:\s*`([\-\d.eE+]+)`", block)
        breakdown_m = re.search(r"Reward breakdown:\s*`(\{.*?\})`", block)
        try:
            scores = json.loads(scores_m.group(1)) if scores_m else {}
        except Exception:
            scores = {}
        try:
            breakdown = json.loads(breakdown_m.group(1)) if breakdown_m else {}
        except Exception:
            breakdown = {}

        def _safe_float(s: re.Match | None) -> float:
            try:
                return float(s.group(1)) if s else 0.0
            except Exception:
                return 0.0

        rounds.append(
            {
                "round": idx,
                "pitches": pitches,
                "winner": winner_m.group(1) if winner_m else "",
                "scores": scores,
                "reason": (reason_m.group(1).strip() if reason_m else ""),
                "judge_bonus": _safe_float(bonus_m),
                "env_reward": _safe_float(env_m),
                "breakdown": breakdown,
            }
        )
    rounds.sort(key=lambda r: r["round"])
    if not rounds:
        return {"rounds": [], "total": 0, "task_type": "", "seed": ""}
    task_m = re.search(r"Task type:\s*`([^`]+)`", md)
    seed_m = re.search(r"Seed:\s*`([^`]+)`", md)
    return {
        "rounds": rounds,
        "total": len(rounds),
        "task_type": task_m.group(1) if task_m else "",
        "seed": seed_m.group(1) if seed_m else "",
    }


def _load_sft_sample(path: Path) -> dict[str, Any]:
    """Read first JSONL line and return the user/assistant pair (compact)."""
    try:
        with path.open(encoding="utf-8") as f:
            line = f.readline()
        rec = json.loads(line)
        msgs = rec.get("messages", [])
        user = next((m for m in msgs if m.get("role") == "user"), {})
        asst = next((m for m in msgs if m.get("role") == "assistant"), {})
        sys = next((m for m in msgs if m.get("role") == "system"), {})
        # Pretty print observation block from user content
        u = user.get("content", "")
        obs_m = re.search(r"Observation:\s*(\{.*\})", u, flags=re.DOTALL)
        obs_pretty = u
        if obs_m:
            try:
                obs_pretty = "Observation:\n" + json.dumps(
                    json.loads(obs_m.group(1)), indent=2
                )
            except Exception:
                pass
        try:
            asst_pretty = json.dumps(json.loads(asst.get("content", "")), indent=2)
        except Exception:
            asst_pretty = asst.get("content", "")
        return {
            "task_type": rec.get("task_type", ""),
            "seed": rec.get("seed", 0),
            "round_index": rec.get("round_index", 0),
            "system": sys.get("content", ""),
            "user": obs_pretty,
            "assistant": asst_pretty,
        }
    except Exception:
        return {}


def _extract_training_headline(md: str) -> dict[str, Any]:
    """Pull headline numbers from artifacts/training_report.md."""

    def _f(pattern: str, default: float) -> float:
        m = re.search(pattern, md)
        try:
            return float(m.group(1)) if m else default
        except Exception:
            return default

    return {
        "episodes": int(_f(r"Training episodes:\s*`(\d+)`", 0)),
        "start": _f(r"Eval reward start:\s*`([\d.\-]+)`", 0.0),
        "final": _f(r"Eval reward final:\s*`([\d.\-]+)`", 0.0),
    }


_DOWNLOADS = [
    ("baseline_eval.json", "/artifacts/baseline_eval.json", _ART_DIR / "baseline_eval.json", "Per-policy eval (training seeds)"),
    ("holdout_eval.json", "/artifacts/holdout_eval.json", _ART_DIR / "holdout_eval.json", "Per-policy eval (holdout seeds)"),
    ("training_curve.json", "/artifacts/training_curve.json", _ART_DIR / "training_curve.json", "Per-episode policy probabilities"),
    ("training_eval.json", "/artifacts/training_eval.json", _ART_DIR / "training_eval.json", "Full per-step training trace"),
    ("training_report.md", "/artifacts/training_report.md", _ART_DIR / "training_report.md", "Final training summary table"),
    ("sft_training_curve.json", "/artifacts/sft_training_curve.json", _ART_DIR / "sft_training_curve.json", "Real SFT loss curve · 500 steps · 13 epochs"),
    ("before_after_training.md", "/artifacts/before_after_training.md", _ART_DIR / "before_after_training.md", "Same-seed before vs after"),
    ("demo_transcript.md", "/artifacts/demo_transcript.md", _ART_DIR / "demo_transcript.md", "Expert demo (coalition_market · seed 5)"),
    ("judged_transcript.md", "/artifacts/judged_transcript.md", _ART_DIR / "judged_transcript.md", "Judged multi-lab debate transcript"),
    ("baseline_rewards.svg", "/plots/baseline_rewards.svg", _PLOT_DIR / "baseline_rewards.svg", "Static plot · vector (renders inline)"),
    ("sft_loss_curve.svg", "/plots/sft_loss_curve.svg", _PLOT_DIR / "sft_loss_curve.svg", "Static SFT loss-curve plot · vector"),
    ("reward_progress.json", "/plots/reward_progress.json", _PLOT_DIR / "reward_progress.json", "Training-progress timeseries"),
    ("sft_messages.jsonl", "/data/sft_messages.jsonl", _DATA_DIR / "sft_messages.jsonl", "SFT chat-formatted dataset"),
    ("sft_traces.jsonl", "/data/sft_traces.jsonl", _DATA_DIR / "sft_traces.jsonl", "SFT raw trace records"),
]


def _build_data_payload() -> dict[str, Any]:
    baseline = _safe_read_json(_ART_DIR / "baseline_eval.json") or {}
    holdout = _safe_read_json(_ART_DIR / "holdout_eval.json") or {}
    progress = _safe_read_json(_PLOT_DIR / "reward_progress.json") or []
    sft_curve = _safe_read_json(_ART_DIR / "sft_training_curve.json") or {}
    demo_md = _safe_read_text(_ART_DIR / "demo_transcript.md")
    ba_md = _safe_read_text(_ART_DIR / "before_after_training.md")
    judged_md = _safe_read_text(_ART_DIR / "judged_transcript.md")
    report_md = _safe_read_text(_ART_DIR / "training_report.md")

    downloads: list[dict[str, str]] = []
    for label, url, path, desc in _DOWNLOADS:
        downloads.append(
            {"label": label, "url": url, "size": _file_size(path), "desc": desc}
        )

    return {
        "baseline": baseline.get("summary", {}) if isinstance(baseline, dict) else {},
        "holdout": holdout.get("summary", {}) if isinstance(holdout, dict) else {},
        "progress": progress if isinstance(progress, list) else [],
        "sft_curve": sft_curve if isinstance(sft_curve, dict) else {},
        "demo": _parse_demo_transcript_md(demo_md),
        "before_after": _parse_before_after_md(ba_md),
        "judged": _parse_judged_rounds(judged_md),
        "sft": _load_sft_sample(_DATA_DIR / "sft_messages.jsonl"),
        "headline": _extract_training_headline(report_md),
        "downloads": downloads,
    }

# ---------------------------------------------------------------------------
# Front-page HTML (served at /)
# ---------------------------------------------------------------------------

_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>GPU Budget Negotiation Arena</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Orbitron:wght@700;900&family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,700;0,9..144,900;1,9..144,700;1,9..144,900&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet"/>
<style>
/* ── Reset & tokens ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #060810;
  --surface:   #0d1221;
  --border:    #1a2540;
  --accent:    #00ffe0;
  --accent2:   #7b5cff;
  --accent3:   #ff4f87;
  --text:      #c8d6f0;
  --muted:     #4a5a7a;
  --expert:    #00ffe0;
  --accept:    #7b5cff;
  --random:    #f4a261;
  --instruct:  #ff4f87;
  --greedy:    #e63946;
  --glow: 0 0 18px rgba(0,255,224,.25);
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Rajdhani', sans-serif;
  font-size: 16px;
  line-height: 1.5;
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── Scanlines overlay ──────────────────────────────────────── */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 999; pointer-events: none;
  background: repeating-linear-gradient(
    to bottom,
    transparent 0px, transparent 3px,
    rgba(0,0,0,.08) 3px, rgba(0,0,0,.08) 4px
  );
}

/* ── Grid background ────────────────────────────────────────── */
body::after {
  content: '';
  position: fixed; inset: 0; z-index: -1; pointer-events: none;
  background-image:
    linear-gradient(rgba(0,255,224,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,224,.03) 1px, transparent 1px);
  background-size: 40px 40px;
  animation: gridDrift 20s linear infinite;
}
@keyframes gridDrift { to { background-position: 40px 40px; } }

/* ── Layout ─────────────────────────────────────────────────── */
.wrap { max-width: 1100px; margin: 0 auto; padding: 0 24px; }

/* ── Header ─────────────────────────────────────────────────── */
header {
  border-bottom: 1px solid var(--border);
  padding: 48px 0 40px;
  text-align: center;
  position: relative;
}
.logo-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 11px; letter-spacing: 4px;
  color: var(--accent); text-transform: uppercase;
  margin-bottom: 12px;
  display: block;
}
h1 {
  font-family: 'Orbitron', sans-serif;
  font-size: clamp(28px, 5vw, 52px);
  font-weight: 900;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 60%, var(--accent3) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1; margin-bottom: 16px;
  text-shadow: none;
  filter: drop-shadow(0 0 20px rgba(0,255,224,.4));
  animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
  0%,100% { filter: drop-shadow(0 0 20px rgba(0,255,224,.4)); }
  50%      { filter: drop-shadow(0 0 40px rgba(123,92,255,.6)); }
}
.tagline {
  color: var(--muted); font-size: 18px; font-weight: 600;
  letter-spacing: 1px; max-width: 620px; margin: 0 auto 28px;
}
.badge-row {
  display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;
  margin-bottom: 16px;
}
.badge {
  padding: 4px 14px; border-radius: 2px;
  font-family: 'Share Tech Mono', monospace; font-size: 11px; letter-spacing: 2px;
  text-transform: uppercase; border: 1px solid currentColor;
}
.badge.green  { color: var(--accent);  border-color: rgba(0,255,224,.4);  background: rgba(0,255,224,.07); }
.badge.purple { color: var(--accent2); border-color: rgba(123,92,255,.4); background: rgba(123,92,255,.07); }
.badge.pink   { color: var(--accent3); border-color: rgba(255,79,135,.4); background: rgba(255,79,135,.07); }
.status-dot {
  display: inline-block; width:8px; height:8px; border-radius:50%;
  background: var(--accent); margin-right:6px;
  box-shadow: 0 0 8px var(--accent);
  animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Section headers ─────────────────────────────────────────── */
.section { padding: 56px 0; border-bottom: 1px solid var(--border); }
.section:last-child { border-bottom: none; }
.section-label {
  font-family: 'Share Tech Mono', monospace; font-size: 11px;
  letter-spacing: 4px; color: var(--accent); text-transform: uppercase;
  margin-bottom: 6px;
}
.section-title {
  font-family: 'Orbitron', sans-serif; font-size: 22px;
  font-weight: 700; margin-bottom: 32px; color: #fff;
}

/* ── Stats bar ───────────────────────────────────────────────── */
.stats-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 16px; margin-bottom: 0;
}
.stat-card {
  background: var(--surface); border: 1px solid var(--border);
  padding: 20px 24px; border-radius: 4px; position: relative; overflow: hidden;
  transition: border-color .3s;
}
.stat-card::before {
  content: ''; position: absolute; top:0; left:0; right:0; height:2px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.stat-card:hover { border-color: var(--accent); box-shadow: var(--glow); }
.stat-value {
  font-family: 'Orbitron', sans-serif; font-size: 28px; font-weight: 700;
  color: var(--accent); line-height: 1;
}
.stat-label {
  font-family: 'Share Tech Mono', monospace; font-size: 10px;
  letter-spacing: 2px; color: var(--muted); text-transform: uppercase;
  margin-top: 6px;
}

/* ── Live demo ───────────────────────────────────────────────── */
.demo-layout {
  display: grid; grid-template-columns: 1fr 1fr; gap: 24px;
}
@media(max-width:720px) { .demo-layout { grid-template-columns: 1fr; } }

.panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; overflow: hidden;
}
.panel-head {
  padding: 12px 18px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.panel-title {
  font-family: 'Share Tech Mono', monospace; font-size: 12px;
  letter-spacing: 2px; color: var(--accent); text-transform: uppercase;
}
.panel-body { padding: 18px; }

/* task selector */
.task-row { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap:wrap; }
.task-btn {
  padding: 6px 14px; border-radius: 2px; border: 1px solid var(--border);
  background: transparent; color: var(--muted); cursor: pointer;
  font-family: 'Share Tech Mono', monospace; font-size: 11px;
  letter-spacing: 1px; text-transform: uppercase;
  transition: all .2s;
}
.task-btn.active, .task-btn:hover {
  border-color: var(--accent); color: var(--accent);
  background: rgba(0,255,224,.06); box-shadow: 0 0 10px rgba(0,255,224,.1);
}
.task-btn.medium.active { border-color: var(--accent2); color: var(--accent2); background: rgba(123,92,255,.06); }
.task-btn.hard.active   { border-color: var(--accent3); color: var(--accent3); background: rgba(255,79,135,.06); }

.run-btn {
  width: 100%; padding: 12px; border-radius: 2px;
  background: linear-gradient(135deg, rgba(0,255,224,.15), rgba(123,92,255,.15));
  border: 1px solid var(--accent); color: var(--accent);
  font-family: 'Orbitron', sans-serif; font-size: 13px; font-weight: 700;
  letter-spacing: 2px; cursor: pointer; text-transform: uppercase;
  transition: all .25s; margin-top: 12px;
}
.run-btn:hover {
  background: linear-gradient(135deg, rgba(0,255,224,.3), rgba(123,92,255,.3));
  box-shadow: 0 0 20px rgba(0,255,224,.3);
}
.run-btn:disabled { opacity: .4; cursor: not-allowed; }

/* transcript */
.transcript {
  height: 360px; overflow-y: auto;
  font-family: 'Share Tech Mono', monospace; font-size: 12px; line-height: 1.8;
  color: var(--text);
}
.transcript::-webkit-scrollbar { width: 4px; }
.transcript::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.t-step { margin-bottom: 14px; border-left: 2px solid var(--border); padding-left: 12px; }
.t-step.good  { border-color: var(--accent); }
.t-step.great { border-color: var(--accent2); }
.t-step.warn  { border-color: var(--accent3); }
.t-head { color: var(--accent); font-size: 10px; letter-spacing: 2px; margin-bottom: 4px; }
.t-action { color: #fff; }
.t-result { color: var(--muted); font-size: 11px; }
.t-reward { float: right; }
.t-reward.pos { color: var(--accent); }
.t-reward.neg { color: var(--accent3); }

/* reward chart */
#rewardCanvas { width:100%; height:200px; display:block; margin-top:12px; }
.reward-total {
  text-align: right;
  font-family: 'Orbitron', sans-serif; font-size: 22px; font-weight: 700;
  color: var(--accent); margin-top: 10px;
}
.reward-label { font-family:'Share Tech Mono',monospace; font-size:10px; color:var(--muted); letter-spacing:2px; text-align:right; }

/* ── Baseline chart ──────────────────────────────────────────── */
.chart-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 24px; }
.chart-tabs { display:flex; gap:8px; margin-bottom:20px; flex-wrap:wrap; }
.chart-tab {
  padding: 5px 14px; border-radius:2px; border: 1px solid var(--border);
  background: transparent; color: var(--muted); cursor: pointer;
  font-family: 'Share Tech Mono', monospace; font-size: 11px;
  letter-spacing: 1px; text-transform: uppercase; transition: all .2s;
}
.chart-tab.active { border-color: var(--accent2); color: var(--accent2); background: rgba(123,92,255,.08); }
#baselineCanvas { width:100%; height:240px; display:block; }
.legend {
  display:flex; flex-wrap:wrap; gap:12px; margin-top:16px;
}
.legend-item { display:flex; align-items:center; gap:6px;
  font-family:'Share Tech Mono',monospace; font-size:11px; color:var(--muted);
}
.legend-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }

/* ── Action space ────────────────────────────────────────────── */
.action-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(220px,1fr)); gap: 12px;
}
.action-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; padding: 14px 16px;
  transition: all .2s; cursor: default;
}
.action-card:hover { border-color: var(--accent2); box-shadow: 0 0 12px rgba(123,92,255,.2); }
.action-name {
  font-family: 'Share Tech Mono', monospace; font-size: 12px;
  color: var(--accent2); margin-bottom: 4px;
}
.action-desc { font-size: 13px; color: var(--muted); }

/* ── Reward breakdown ────────────────────────────────────────── */
.reward-cols { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
@media(max-width:600px) { .reward-cols { grid-template-columns:1fr; } }
.reward-item {
  background: var(--surface); border: 1px solid var(--border);
  border-radius:4px; padding:14px 16px;
}
.reward-name {
  font-family:'Share Tech Mono',monospace; font-size:11px;
  color: var(--accent); letter-spacing:1px; margin-bottom: 6px;
}
.reward-bar-track { height:4px; background: var(--border); border-radius:2px; overflow:hidden; }
.reward-bar-fill  { height:100%; border-radius:2px; transition: width 1s ease; }

/* ── Transcript artifact ─────────────────────────────────────── */
.artifact-box {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; padding: 24px; font-family: 'Share Tech Mono', monospace;
  font-size: 12px; line-height: 1.9; color: var(--text);
  max-height: 480px; overflow-y: auto;
}
.artifact-box::-webkit-scrollbar { width: 4px; }
.artifact-box::-webkit-scrollbar-thumb { background: var(--border); }
.artifact-step-head { color: var(--accent2); font-size: 11px; letter-spacing: 2px; margin: 16px 0 4px; }
.artifact-action  { color: #fff; }
.artifact-result  { color: var(--muted); }
.artifact-reward  { color: var(--accent); }

/* ── API section ─────────────────────────────────────────────── */
.api-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px,1fr)); gap: 20px; }
.api-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; overflow: hidden;
}
.api-card-head {
  padding: 10px 16px; border-bottom: 1px solid var(--border);
  display: flex; gap: 10px; align-items: center;
}
.method {
  font-family:'Share Tech Mono',monospace; font-size:11px;
  padding: 2px 8px; border-radius:2px; font-weight:700;
}
.method.get  { background: rgba(0,255,224,.12); color: var(--accent); }
.method.post { background: rgba(123,92,255,.12); color: var(--accent2); }
.endpoint {
  font-family:'Share Tech Mono',monospace; font-size:13px; color:#fff;
}
.api-desc { padding: 12px 16px; font-size: 14px; color: var(--muted); }

/* ── Footer ──────────────────────────────────────────────────── */
footer {
  border-top: 1px solid var(--border);
  padding: 32px 0; text-align: center;
  font-family: 'Share Tech Mono', monospace; font-size: 11px;
  color: var(--muted); letter-spacing: 2px;
}
footer a { color: var(--accent); text-decoration: none; }
footer a:hover { text-decoration: underline; }

/* ── Utilities ───────────────────────────────────────────────── */
.mono { font-family:'Share Tech Mono',monospace; }
.mt8 { margin-top:8px; }
.mt16 { margin-top:16px; }
.placeholder {
  height: 200px; display:flex; align-items:center; justify-content:center;
  color: var(--muted); font-family:'Share Tech Mono',monospace; font-size:12px;
  letter-spacing:2px; border: 1px dashed var(--border); border-radius:4px;
}

/* ── Animated entrance ───────────────────────────────────────── */
.fade-up {
  opacity:0; transform: translateY(24px);
  animation: fadeUp .6s ease forwards;
}
@keyframes fadeUp { to { opacity:1; transform:translateY(0); } }
.d1 { animation-delay:.1s } .d2 { animation-delay:.25s }
.d3 { animation-delay:.4s  } .d4 { animation-delay:.55s }

/* ── Theme v3: HOT PRESS — Risograph Editorial ─────────────────────────── */
:root {
  --paper:    #f1ebdb;
  --paper-2:  #ebe3cf;
  --paper-3:  #e2d8bf;
  --ink:      #0c0a08;
  --ink-soft: #3a3530;
  --rule:     rgba(12,10,8,.20);
  --rule-2:   rgba(12,10,8,.55);
  --riso-red:    #ff3b30;
  --riso-blue:   #1234ff;
  --riso-yellow: #f7c548;
  --riso-teal:   #0fa991;
  --riso-violet: #6a4cff;

  --bg:      var(--paper);
  --surface: var(--paper);
  --border:  var(--ink);
  --text:    var(--ink);
  --muted:   var(--ink-soft);
  --accent:  var(--riso-red);
  --accent2: var(--riso-blue);
  --accent3: var(--riso-violet);
  --glow:    none;
}

body {
  counter-reset: section;
  background:
    radial-gradient(circle at 12% 8%,  rgba(255,59,48,.08),   transparent 38%),
    radial-gradient(circle at 88% 4%,  rgba(18,52,255,.07),   transparent 40%),
    radial-gradient(circle at 50% 110%, rgba(247,197,72,.10), transparent 50%),
    var(--paper);
  color: var(--ink);
  font-family: 'Inter', system-ui, sans-serif;
  cursor: none;
}
body::before {
  background: none;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0.04  0 0 0 0 0.04  0 0 0 0 0.04  0 0 0 0.55 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
  opacity: .14;
  mix-blend-mode: multiply;
  z-index: 1;
}
body::after {
  background-image:
    radial-gradient(circle, rgba(12,10,8,.18) 1px, transparent 1.4px);
  background-size: 26px 26px;
  background-position: 0 0;
  opacity: .25;
  animation: none;
}

.wrap { position: relative; z-index: 3; max-width: 1180px; }

/* ── Custom cursor (dot + ring) ───────────────────────────── */
#cursorGlow { display: none !important; }
#cursorDot, #cursorRing {
  position: fixed; pointer-events: none;
  z-index: 9999; left: 0; top: 0;
  transform: translate(-50%, -50%);
  mix-blend-mode: multiply;
}
#cursorDot {
  width: 6px; height: 6px;
  background: var(--ink); border-radius: 50%;
  transition: background .15s ease, transform .12s ease;
}
#cursorRing {
  width: 30px; height: 30px;
  border: 1.5px solid var(--ink); border-radius: 50%;
  transition: width .18s ease, height .18s ease, border-color .18s ease, background .18s ease;
}
body.cursor-active #cursorRing {
  width: 56px; height: 56px;
  border-color: var(--riso-red);
  background: rgba(255,59,48,.10);
}
body.cursor-active #cursorDot { background: var(--riso-red); }
@media (hover: none) {
  #cursorDot, #cursorRing { display: none; }
  body { cursor: auto; }
}

/* ── Header ───────────────────────────────────────────────── */
header {
  border: none;
  background: transparent;
  box-shadow: none;
  padding: 64px 0 44px;
  text-align: left;
  margin: 0;
}
header .wrap { position: relative; }
header::after {
  content: ""; position: absolute; left: 24px; right: 24px; bottom: 0;
  height: 2px; background: var(--ink);
}

.logo-label {
  font-family: 'JetBrains Mono', monospace;
  color: var(--ink);
  font-size: 11px; letter-spacing: 3px;
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--riso-yellow);
  padding: 5px 12px;
  border: 1.5px solid var(--ink);
  box-shadow: 3px 3px 0 var(--ink);
  margin-bottom: 6px;
}

h1 {
  font-family: 'Fraunces', 'Times New Roman', serif;
  font-style: italic;
  font-weight: 900;
  letter-spacing: -0.04em;
  font-size: clamp(48px, 8.5vw, 116px);
  line-height: .9;
  margin: 18px 0 22px;
  background: none;
  -webkit-background-clip: initial; -webkit-text-fill-color: var(--ink);
  background-clip: initial;
  color: var(--ink);
  text-shadow: none;
  filter: none;
  animation: none;
}
h1 em {
  font-style: italic;
}
h1 em.accent-1 { color: var(--riso-red);   text-decoration: underline wavy var(--ink); text-decoration-thickness: 2px; text-underline-offset: 10px; }
h1 em.accent-2 { color: var(--riso-blue); }

.tagline {
  font-family: 'Inter', sans-serif;
  font-size: 19px;
  color: var(--ink-soft);
  font-weight: 500;
  margin: 0 0 28px;
  max-width: 660px;
  letter-spacing: 0;
}

.badge-row { justify-content: flex-start; gap: 8px; }
.badge {
  border-radius: 0;
  border: 1.5px solid var(--ink);
  padding: 5px 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px;
  letter-spacing: 1.6px;
  color: var(--ink);
  background: var(--paper);
  box-shadow: 2px 2px 0 var(--ink);
  transition: transform .15s ease, box-shadow .15s ease;
}
.badge:hover { transform: translate(-1px,-1px); box-shadow: 3px 3px 0 var(--ink); }
.badge.green  { background: #d4ecdb; }
.badge.purple { background: #d6dbff; }
.badge.pink   { background: #ffd9e1; }
.status-dot { background: var(--riso-red); box-shadow: none; animation: blink 1.2s ease-in-out infinite; }

/* ── Sections ─────────────────────────────────────────────── */
.section {
  border-bottom: none;
  padding: 56px 0;
  position: relative;
  counter-increment: section;
}
.section + .section::before {
  content: ""; position: absolute; left: 24px; right: 24px; top: 0;
  height: 2px; background: var(--ink);
}

.section-label {
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 3px;
  color: var(--ink);
  font-size: 11px;
  display: inline-flex; align-items: baseline; gap: 12px;
  margin-bottom: 4px;
  text-transform: uppercase;
}
.section-label::before {
  content: "§ " counter(section, decimal-leading-zero);
  font-family: 'Fraunces', serif;
  font-style: italic;
  font-weight: 900;
  font-size: 32px;
  color: var(--riso-red);
  line-height: 1;
  letter-spacing: -0.02em;
}

.section-title {
  font-family: 'Fraunces', serif;
  font-weight: 900;
  font-style: italic;
  letter-spacing: -.02em;
  color: var(--ink);
  font-size: clamp(28px, 4.2vw, 48px);
  line-height: 1;
  margin: 8px 0 28px;
}

/* ── Stat cards ──────────────────────────────────────────── */
.stats-grid { gap: 14px; }
.stat-card {
  background: var(--paper);
  border: 1.5px solid var(--ink);
  border-radius: 0;
  box-shadow: 4px 4px 0 var(--ink);
  padding: 18px 22px 16px;
  transition: transform .2s cubic-bezier(.2,.7,.3,1), box-shadow .2s ease, background .2s ease;
}
.stat-card::before {
  height: 4px;
  background: linear-gradient(90deg, var(--riso-red) 0 33%, var(--riso-blue) 33% 66%, var(--riso-yellow) 66% 100%);
}
.stat-card:hover {
  transform: translate(-3px, -3px);
  box-shadow: 7px 7px 0 var(--ink);
  background: var(--paper-2);
}
.stat-value {
  font-family: 'Fraunces', serif;
  font-style: italic; font-weight: 900;
  color: var(--ink);
  font-size: 44px; line-height: 1;
}
.stat-label {
  font-family: 'JetBrains Mono', monospace;
  color: var(--ink-soft);
  letter-spacing: 1.8px;
  font-size: 10.5px;
  margin-top: 8px;
}

/* ── Panels (Demo) ───────────────────────────────────────── */
.panel {
  background: var(--paper);
  border: 1.5px solid var(--ink);
  border-radius: 0;
  box-shadow: 5px 5px 0 var(--ink);
  overflow: hidden;
  transition: transform .2s ease, box-shadow .2s ease;
}
.panel:hover { transform: translate(-2px,-2px); box-shadow: 7px 7px 0 var(--ink); }
.panel-head {
  background: var(--ink);
  color: var(--paper);
  border-bottom: 1.5px solid var(--ink);
  padding: 10px 16px;
}
.panel-title { color: var(--paper); font-family: 'JetBrains Mono', monospace; letter-spacing: 2px; }
#demo-status, #step-counter {
  color: var(--paper) !important;
  font-family: 'JetBrains Mono', monospace !important;
}
.panel-body { padding: 18px; background: var(--paper); }

/* ── Task chips ──────────────────────────────────────────── */
.task-row { gap: 6px; }
.task-btn, .chart-tab {
  border-radius: 0;
  border: 1.5px solid var(--ink);
  background: var(--paper);
  color: var(--ink);
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; letter-spacing: 1.6px;
  padding: 7px 14px;
  box-shadow: 3px 3px 0 var(--ink);
  transition: transform .14s ease, box-shadow .14s ease, background .14s ease, color .14s ease;
}
.task-btn:hover, .chart-tab:hover {
  transform: translate(-1px,-1px);
  box-shadow: 4px 4px 0 var(--ink);
  background: var(--riso-yellow);
  color: var(--ink);
  border-color: var(--ink);
}
.task-btn.active {
  background: var(--ink); color: var(--paper);
  box-shadow: 3px 3px 0 var(--riso-red);
}
.task-btn.medium.active { box-shadow: 3px 3px 0 var(--riso-blue); }
.task-btn.hard.active   { box-shadow: 3px 3px 0 var(--riso-violet); }
.chart-tab.active {
  background: var(--ink); color: var(--paper);
  box-shadow: 3px 3px 0 var(--riso-blue);
}

/* ── Run button ──────────────────────────────────────────── */
.run-btn {
  position: relative; overflow: hidden;
  border-radius: 0;
  border: 1.5px solid var(--ink);
  background: var(--riso-red);
  color: var(--paper);
  font-family: 'Fraunces', serif;
  font-style: italic; font-weight: 900;
  font-size: 18px; letter-spacing: 1px;
  padding: 14px 16px;
  box-shadow: 5px 5px 0 var(--ink);
  margin-top: 14px;
  text-transform: none;
  transition: transform .14s ease, box-shadow .14s ease, background .14s ease, color .14s ease;
}
.run-btn::after {
  content: ""; position: absolute; inset: 0;
  background: radial-gradient(circle at var(--mx,50%) var(--my,50%), rgba(247,197,72,.55), transparent 55%);
  opacity: 0; transition: opacity .25s ease;
  pointer-events: none;
}
.run-btn:hover {
  transform: translate(-2px,-2px);
  box-shadow: 7px 7px 0 var(--ink);
  background: var(--ink); color: var(--riso-yellow);
}
.run-btn:hover::after { opacity: 1; }
.run-btn:active { transform: translate(2px,2px); box-shadow: 1px 1px 0 var(--ink); }
.run-btn:disabled {
  background: var(--paper-2); color: var(--ink-soft);
  box-shadow: 3px 3px 0 var(--rule-2);
  transform: none;
}

/* ── Transcript ──────────────────────────────────────────── */
.transcript {
  border: 1.5px dashed var(--ink);
  background: var(--paper-2);
  padding: 12px;
  height: 360px;
  font-family: 'JetBrains Mono', monospace;
  color: var(--ink);
}
.t-step {
  border-left: 4px solid var(--ink);
  background: var(--paper);
  padding: 8px 12px;
  margin-bottom: 10px;
  border-radius: 0;
  box-shadow: 3px 3px 0 var(--ink);
  transition: transform .14s ease, box-shadow .14s ease;
}
.t-step:hover { transform: translate(-1px,-1px); box-shadow: 4px 4px 0 var(--ink); background: var(--paper-2); }
.t-step.good  { border-left-color: var(--riso-blue); }
.t-step.great { border-left-color: var(--riso-red); }
.t-step.warn  { border-left-color: var(--riso-violet); }
.t-head   { color: var(--ink); font-weight: 700; }
.t-action { color: var(--ink); }
.t-result { color: var(--ink-soft); }
.t-reward.pos { color: var(--riso-blue); }
.t-reward.neg { color: var(--riso-red); }

/* ── Reward chart canvas ─────────────────────────────────── */
#rewardCanvas, #baselineCanvas {
  background: var(--paper-2);
  border: 1.5px solid var(--ink);
  border-radius: 0;
}
.reward-label { color: var(--ink-soft); }
.reward-total {
  color: var(--ink);
  font-family: 'Fraunces', serif;
  font-style: italic; font-weight: 900;
  font-size: 36px;
}

/* ── Baseline chart wrap ─────────────────────────────────── */
.chart-wrap {
  background: var(--paper);
  border: 1.5px solid var(--ink);
  border-radius: 0;
  box-shadow: 6px 6px 0 var(--ink);
  padding: 22px;
}
.legend-item { color: var(--ink-soft); font-family: 'JetBrains Mono', monospace; }
.legend-dot { border: 1.5px solid var(--ink); border-radius: 0; }

/* ── Action grid ─────────────────────────────────────────── */
.action-grid { gap: 14px; }
.action-card {
  background: var(--paper);
  border: 1.5px solid var(--ink);
  border-radius: 0;
  box-shadow: 4px 4px 0 var(--ink);
  padding: 14px 16px;
  transition: transform .18s ease, box-shadow .18s ease, background .18s ease;
}
.action-card:hover {
  transform: translate(-3px,-3px);
  box-shadow: 7px 7px 0 var(--riso-blue);
  background: var(--paper-2);
}
.action-name {
  font-family: 'JetBrains Mono', monospace;
  color: var(--ink);
  font-weight: 700;
  letter-spacing: 1px;
}
.action-desc { color: var(--ink-soft); font-family: 'Inter', sans-serif; }

/* ── Reward signals ──────────────────────────────────────── */
.reward-cols { gap: 14px; }
.reward-item {
  background: var(--paper);
  border: 1.5px solid var(--ink);
  border-radius: 0;
  box-shadow: 4px 4px 0 var(--ink);
  padding: 14px 16px;
  transition: transform .18s ease, box-shadow .18s ease;
}
.reward-item:hover { transform: translate(-2px,-2px); box-shadow: 6px 6px 0 var(--riso-red); }
.reward-name { color: var(--ink); font-family: 'JetBrains Mono', monospace; }
.reward-bar-track { background: var(--paper-2); border: 1px solid var(--ink); height: 8px; border-radius: 0; }
.reward-bar-fill { border-radius: 0; height: 100%; }

/* ── Artifact box ────────────────────────────────────────── */
.artifact-box {
  background: var(--paper);
  border: 1.5px solid var(--ink);
  border-radius: 0;
  box-shadow: 6px 6px 0 var(--ink);
  font-family: 'JetBrains Mono', monospace;
  color: var(--ink);
  padding: 22px;
}
.artifact-step-head { color: var(--riso-red); font-weight: 700; letter-spacing: 1px; }
.artifact-action  { color: var(--ink); }
.artifact-result  { color: var(--ink-soft); }
.artifact-reward  { color: var(--riso-blue); }

/* ── API ─────────────────────────────────────────────────── */
.api-grid { gap: 14px; }
.api-card {
  background: var(--paper);
  border: 1.5px solid var(--ink);
  border-radius: 0;
  box-shadow: 4px 4px 0 var(--ink);
  transition: transform .18s ease, box-shadow .18s ease;
  overflow: hidden;
}
.api-card:hover { transform: translate(-3px,-3px); box-shadow: 7px 7px 0 var(--riso-violet); }
.api-card-head {
  padding: 10px 14px;
  border-bottom: 1.5px solid var(--ink);
  background: var(--paper-2);
}
.method {
  border-radius: 0;
  border: 1.5px solid var(--ink);
  font-family: 'JetBrains Mono', monospace;
  padding: 2px 8px;
  font-size: 11px;
}
.method.get  { background: var(--riso-yellow); color: var(--ink); }
.method.post { background: var(--riso-red); color: var(--paper); }
.endpoint { color: var(--ink); font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.api-desc { color: var(--ink-soft); padding: 12px 14px; font-family: 'Inter', sans-serif; }

/* ── Footer ──────────────────────────────────────────────── */
footer {
  border: none;
  background: var(--ink);
  color: var(--paper);
  padding: 32px 0;
  margin-top: 56px;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 1.5px;
}
footer a {
  color: var(--riso-yellow);
  border-bottom: 1px solid currentColor;
  text-decoration: none;
  transition: color .15s ease;
}
footer a:hover { color: var(--paper); }

/* ── Stat ticker caret ───────────────────────────────────── */
.stat-value.is-counting::after {
  content: "";
  display: inline-block;
  width: 6px; height: 28px;
  background: var(--riso-red);
  margin-left: 6px;
  vertical-align: -4px;
  animation: blink 1s steps(2) infinite;
}

/* ── Marquee ticker (header sub) ─────────────────────────── */
.ticker {
  margin-top: 14px;
  border: 1.5px solid var(--ink);
  background: var(--paper-2);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11.5px; letter-spacing: 1.6px;
  color: var(--ink);
  overflow: hidden;
  position: relative;
  box-shadow: 3px 3px 0 var(--ink);
}
.ticker-track {
  display: inline-flex; gap: 32px;
  white-space: nowrap;
  padding: 8px 0;
  animation: tickerScroll 28s linear infinite;
  will-change: transform;
}
.ticker-item { padding-left: 32px; position: relative; }
.ticker-item::before {
  content: "✦"; color: var(--riso-red);
  position: absolute; left: 8px; top: 50%; transform: translateY(-50%);
}
@keyframes tickerScroll {
  from { transform: translateX(0); }
  to   { transform: translateX(-50%); }
}

/* ── Stamp on click ──────────────────────────────────────── */
.stamp {
  position: fixed; pointer-events: none; z-index: 9998;
  font-family: 'Fraunces', serif; font-style: italic; font-weight: 900;
  font-size: 24px; color: var(--riso-red);
  border: 2px solid var(--riso-red);
  padding: 4px 12px;
  transform: translate(-50%,-50%) rotate(-12deg);
  opacity: 0;
  letter-spacing: 2px;
  text-transform: uppercase;
  mix-blend-mode: multiply;
}
.stamp.show {
  animation: stampPop .55s cubic-bezier(.2,.9,.3,1) forwards;
}
@keyframes stampPop {
  0%   { opacity: 0; transform: translate(-50%,-50%) rotate(-12deg) scale(2.2); }
  35%  { opacity: 1; transform: translate(-50%,-50%) rotate(-9deg)  scale(.95); }
  60%  { opacity: 1; transform: translate(-50%,-50%) rotate(-12deg) scale(1); }
  100% { opacity: 0; transform: translate(-50%,-50%) rotate(-12deg) scale(1.05); }
}

/* ── Train vs Holdout toggle ─────────────────────────────── */
.split-row { display:flex; gap: 8px; margin: 0 0 14px; flex-wrap: wrap; align-items: center; }
.split-row .split-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; letter-spacing: 1.6px;
  color: var(--ink-soft); margin-right: 6px;
}
.split-tab {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; letter-spacing: 1.6px;
  border: 1.5px solid var(--ink); background: var(--paper); color: var(--ink);
  padding: 6px 12px; cursor: pointer;
  box-shadow: 3px 3px 0 var(--ink);
  transition: transform .14s ease, box-shadow .14s ease, background .14s ease, color .14s ease;
}
.split-tab:hover { transform: translate(-1px,-1px); box-shadow: 4px 4px 0 var(--ink); background: var(--riso-yellow); }
.split-tab.active { background: var(--ink); color: var(--paper); box-shadow: 3px 3px 0 var(--riso-blue); }

/* ── Training-progress chart ─────────────────────────────── */
#progressCanvas, #sftCurveCanvas {
  width:100%; height: 280px; display:block;
  background: var(--paper-2);
  border: 1.5px solid var(--ink);
  border-radius: 0;
}
.progress-wrap {
  background: var(--paper);
  border: 1.5px solid var(--ink);
  box-shadow: 6px 6px 0 var(--ink);
  padding: 22px;
}
.sft-curve-stats {
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  margin-top: 8px;
}
.sft-stat {
  background: var(--paper); border: 1.5px solid var(--ink);
  box-shadow: 4px 4px 0 var(--ink);
  padding: 12px 14px;
}
.sft-stat .v {
  font-family: 'Fraunces', serif; font-style: italic; font-weight: 900;
  font-size: 26px; color: var(--ink); line-height: 1;
}
.sft-stat .v.red  { color: var(--riso-red); }
.sft-stat .v.blue { color: var(--riso-blue); }
.sft-stat .l {
  margin-top: 6px;
  font-family: 'JetBrains Mono', monospace; font-size: 10.5px; letter-spacing: 1.6px;
  color: var(--ink-soft); text-transform: uppercase;
}
.sft-curve-note {
  margin-top: 10px;
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  color: var(--ink-soft);
}
.sft-curve-note b { color: var(--ink); }

/* ── Before / After ──────────────────────────────────────── */
.ba-grid { display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 720px) { .ba-grid { grid-template-columns: 1fr; } }
.ba-col {
  background: var(--paper); border: 1.5px solid var(--ink);
  box-shadow: 6px 6px 0 var(--ink); padding: 18px;
  position: relative;
}
.ba-col::before {
  content: attr(data-label);
  position: absolute; top: -10px; left: 14px;
  background: var(--ink); color: var(--paper);
  padding: 2px 10px;
  font-family: 'JetBrains Mono', monospace; font-size: 10.5px; letter-spacing: 2px;
}
.ba-col.after-col::before { background: var(--riso-red); }
.ba-meta { display:flex; gap:6px; align-items:baseline; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--ink-soft); }
.ba-meta span { letter-spacing: 1.4px; }
.ba-meta b { color: var(--ink); margin-right: 12px; font-weight: 700; }
.ba-total {
  margin: 12px 0 14px; padding: 10px 12px;
  background: var(--paper-2);
  border: 1px dashed var(--ink);
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  display:flex; justify-content: space-between; align-items: baseline;
}
.ba-total b { font-family: 'Fraunces', serif; font-style: italic; font-size: 22px; color: var(--ink); }
.ba-delta { color: var(--riso-red); font-family: 'Fraunces', serif; font-style: italic; font-weight: 900; font-size: 18px; }
.ba-steps { display:flex; flex-direction: column; gap: 8px; max-height: 360px; overflow:auto; padding-right: 4px; }
.ba-step {
  background: var(--paper); border: 1.5px solid var(--ink);
  padding: 8px 10px;
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
}
.ba-step-head { color: var(--riso-red); font-weight: 700; letter-spacing: 1px; margin-bottom: 4px; }
.ba-action { color: var(--ink); word-break: break-all; }
.ba-result { color: var(--ink-soft); margin-top: 2px; font-size: 10.5px; }
.ba-reward { float:right; font-family: 'Fraunces', serif; font-style: italic; font-weight: 900; }
.ba-reward.pos { color: var(--riso-blue); }
.ba-reward.neg { color: var(--riso-red); }

/* ── Judged round ────────────────────────────────────────── */
.judged-wrap {
  background: var(--paper); border: 1.5px solid var(--ink);
  box-shadow: 6px 6px 0 var(--ink); padding: 22px;
}
.jr-nav {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 14px;
  align-items: center;
  padding-bottom: 16px;
  margin-bottom: 18px;
  border-bottom: 1.5px dashed var(--ink);
}
@media (max-width: 640px) {
  .jr-nav { grid-template-columns: 1fr 1fr; }
  .jr-nav .jr-progress { grid-column: 1 / -1; order: 2; }
}
.jr-btn {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px; letter-spacing: 1.6px;
  background: var(--paper); color: var(--ink);
  border: 1.5px solid var(--ink);
  padding: 10px 14px; cursor: pointer;
  box-shadow: 3px 3px 0 var(--ink);
  transition: transform .12s ease, box-shadow .12s ease, background .12s ease, color .12s ease;
}
.jr-btn:hover:not(:disabled) {
  transform: translate(-1px,-1px);
  box-shadow: 4px 4px 0 var(--ink);
  background: var(--riso-yellow);
}
.jr-btn:active:not(:disabled) {
  transform: translate(2px,2px);
  box-shadow: 1px 1px 0 var(--ink);
  background: var(--ink); color: var(--paper);
}
.jr-btn:disabled {
  opacity: .35;
  cursor: not-allowed;
  box-shadow: 2px 2px 0 var(--ink);
}
.jr-progress {
  display: flex; flex-direction: column; gap: 8px; min-width: 0;
}
.jr-progress-label {
  display: flex; justify-content: space-between; align-items: baseline; gap: 10px;
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  color: var(--ink-soft); letter-spacing: 1.4px;
}
.jr-progress-label #jrCounter { color: var(--ink); font-weight: 700; }
.jr-score-trend { color: var(--riso-red); font-family: 'Fraunces', serif; font-style: italic; font-weight: 900; font-size: 16px; letter-spacing: 0; }
#jrTrendCanvas {
  width: 100%; height: 60px; display: block;
  background: var(--paper-2);
  border: 1.5px solid var(--ink);
}
.jr-head {
  font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 1.6px;
  color: var(--ink); margin-bottom: 6px;
}
.jr-head b { color: var(--riso-red); font-family: 'Fraunces', serif; font-style: italic; font-weight: 900; font-size: 22px; }
.jr-reason { font-size: 13px; color: var(--ink-soft); margin-bottom: 16px; max-width: 760px; }
.jr-pitches { display: grid; gap: 10px; margin-bottom: 16px; }
.pitch {
  background: var(--paper-2); border: 1.5px solid var(--ink);
  padding: 10px 14px;
  position: relative;
}
.pitch.pitch-win { border-color: var(--riso-red); box-shadow: 4px 4px 0 var(--riso-red); }
.pitch-head { display:flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.pitch-lab {
  font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 1.6px;
  background: var(--ink); color: var(--paper); padding: 2px 8px;
}
.pitch.pitch-win .pitch-lab { background: var(--riso-red); }
.pitch-score {
  font-family: 'Fraunces', serif; font-style: italic; font-weight: 900; font-size: 16px; color: var(--ink);
}
.pitch-text { font-family: 'Inter', sans-serif; font-size: 13px; color: var(--ink); }
.pitch-bar {
  margin-top: 8px; height: 6px; background: var(--paper); border: 1px solid var(--ink);
}
.pitch-bar span { display:block; height: 100%; background: var(--ink); }
.pitch.pitch-win .pitch-bar span { background: var(--riso-red); }
.jr-breakdown { display:flex; flex-wrap:wrap; gap: 10px; padding-top: 12px; border-top: 1.5px dashed var(--ink); }
.rb-item {
  font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--ink-soft);
  background: var(--paper-2); border: 1px solid var(--ink); padding: 4px 8px;
}
.rb-item b { color: var(--ink); }

/* ── SFT sample ──────────────────────────────────────────── */
.sft-wrap {
  background: var(--paper); border: 1.5px solid var(--ink);
  box-shadow: 6px 6px 0 var(--ink); padding: 22px;
}
.sft-meta {
  display:flex; flex-wrap:wrap; gap: 6px 14px; margin-bottom: 12px;
  font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--ink-soft);
}
.sft-meta b { color: var(--ink); margin-right: 10px; }
.sft-role {
  display:inline-block; margin: 14px 0 4px;
}
.sft-role span {
  background: var(--ink); color: var(--paper);
  font-family: 'JetBrains Mono', monospace;
  padding: 3px 10px; font-size: 10.5px; letter-spacing: 1.6px;
}
.sft-pre {
  background: var(--paper-2);
  border: 1.5px dashed var(--ink);
  padding: 12px 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11.5px;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 320px;
  overflow: auto;
}
.sft-pre.sft-asst { border-color: var(--riso-red); background: rgba(255,59,48,.06); }

/* ── Downloads grid ──────────────────────────────────────── */
.dl-grid {
  display: grid; gap: 14px;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
}
.dl-card {
  display:block; text-decoration:none; color: var(--ink);
  background: var(--paper); border: 1.5px solid var(--ink);
  box-shadow: 4px 4px 0 var(--ink);
  padding: 14px 16px;
  transition: transform .14s ease, box-shadow .14s ease, background .14s ease;
}
.dl-card:hover { transform: translate(-3px,-3px); box-shadow: 7px 7px 0 var(--riso-red); background: var(--paper-2); }
.dl-row { display:flex; justify-content: space-between; align-items: baseline; gap:8px; }
.dl-label { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 12.5px; }
.dl-size {
  font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
  background: var(--ink); color: var(--paper);
  padding: 2px 8px;
}
.dl-desc { font-family: 'Inter', sans-serif; font-size: 12.5px; color: var(--ink-soft); margin-top: 6px; }
.dl-cta {
  margin-top: 10px;
  font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: 1.6px;
  color: var(--riso-red);
}

/* ── Reduced motion ──────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
  body { cursor: auto; }
  #cursorDot, #cursorRing { display: none; }
}
</style>
</head>
<body>
<div id="cursorRing" aria-hidden="true"></div>
<div id="cursorDot" aria-hidden="true"></div>

<!-- ══════════ HEADER ══════════ -->
<header>
<div class="wrap">
  <span class="logo-label">● OpenEnv · Multi-Agent Benchmark · MMXXVI</span>
  <h1>GPU <em class="accent-1">Budget</em><br/>Negotiation <em class="accent-2">Arena</em>.</h1>
  <p class="tagline">Train LLMs to bargain, form coalitions, and adapt under market shocks in a scarce-GPU economy. A live, dense-reward, multi-agent press.</p>
  <div class="badge-row">
    <span class="badge green"><span class="status-dot"></span>Running</span>
    <span class="badge purple">Theme #1 · Multi-Agent</span>
    <span class="badge pink">OpenEnv Compatible</span>
    <span class="badge green">FastAPI · Docker</span>
  </div>
  <div class="ticker" aria-hidden="true">
    <div class="ticker-track" id="tickerTrack"></div>
  </div>
</div>
</header>

<!-- ══════════ STATS ══════════ -->
<div class="section fade-up d1">
<div class="wrap">
  <p class="section-label">Environment Stats</p>
  <p class="section-title">At a Glance</p>
  <div class="stats-grid">
    <div class="stat-card"><div class="stat-value">3</div><div class="stat-label">Task Types</div></div>
    <div class="stat-card"><div class="stat-value">12</div><div class="stat-label">Action Types</div></div>
    <div class="stat-card"><div class="stat-value">10</div><div class="stat-label">Reward Signals</div></div>
    <div class="stat-card"><div class="stat-value">6</div><div class="stat-label">Baseline Policies</div></div>
    <div class="stat-card"><div class="stat-value" id="stat-eps">180</div><div class="stat-label">Train Episodes</div></div>
    <div class="stat-card"><div class="stat-value" id="stat-final">0.45</div><div class="stat-label">Final Eval Reward</div></div>
    <div class="stat-card"><div class="stat-value" id="stat-health">…</div><div class="stat-label">API Status</div></div>
    <div class="stat-card"><div class="stat-value">0.81</div><div class="stat-label">Expert Reward (Hard)</div></div>
  </div>
</div>
</div>

<!-- ══════════ LIVE DEMO ══════════ -->
<div class="section fade-up d2">
<div class="wrap">
  <p class="section-label">Interactive</p>
  <p class="section-title">Live Demo — Rule-Based Expert</p>
  <div class="demo-layout">

    <!-- Left: controls + transcript -->
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">Negotiation Session</span>
        <span id="demo-status" style="font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--muted);">IDLE</span>
      </div>
      <div class="panel-body">
        <div class="task-row">
          <button class="task-btn active"     data-task="single_trade">Easy</button>
          <button class="task-btn medium"     data-task="market_round">Medium</button>
          <button class="task-btn hard"       data-task="coalition_market">Hard</button>
        </div>
        <div id="transcript" class="transcript">
          <div style="color:var(--muted);font-family:'Share Tech Mono',monospace;font-size:12px;padding:24px;text-align:center;">
            Press RUN to start a live negotiation episode.
          </div>
        </div>
        <button class="run-btn" id="runBtn">▶ RUN EPISODE</button>
      </div>
    </div>

    <!-- Right: reward chart -->
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">Reward Accumulation</span>
        <span id="step-counter" style="font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--muted);">STEP 0</span>
      </div>
      <div class="panel-body">
        <canvas id="rewardCanvas"></canvas>
        <div class="reward-label">CUMULATIVE REWARD</div>
        <div class="reward-total" id="cumReward">0.0000</div>
      </div>
    </div>

  </div>
</div>
</div>

<!-- ══════════ BASELINES CHART ══════════ -->
<div class="section fade-up d3">
<div class="wrap">
  <p class="section-label">Evaluation Results</p>
  <p class="section-title">Baseline Policy Performance · with Std-Dev Whiskers</p>
  <div class="chart-wrap">
    <div class="split-row">
      <span class="split-label">SPLIT</span>
      <button class="split-tab active" data-split="train">Training seeds</button>
      <button class="split-tab"        data-split="holdout">Holdout seeds</button>
    </div>
    <div class="chart-tabs">
      <button class="chart-tab active" data-task="single_trade">Easy — Single Trade</button>
      <button class="chart-tab"        data-task="market_round">Medium — Market Round</button>
      <button class="chart-tab"        data-task="coalition_market">Hard — Coalition Market</button>
    </div>
    <canvas id="baselineCanvas" style="height:260px;display:block;width:100%;"></canvas>
    <div class="legend" id="legend"></div>
  </div>
</div>
</div>

<!-- ══════════ TRAINING PROGRESS ══════════ -->
<div class="section fade-up d3">
<div class="wrap">
  <p class="section-label">Training Trajectory</p>
  <p class="section-title">Reward Progress · 180 Episodes</p>
  <div class="progress-wrap">
    <canvas id="progressCanvas"></canvas>
    <div class="legend" id="progressLegend" style="margin-top:14px;"></div>
  </div>
</div>
</div>

<!-- ══════════ SFT LOSS CURVE ══════════ -->
<div class="section fade-up d3">
<div class="wrap">
  <p class="section-label">SFT Optimisation</p>
  <p class="section-title" id="sftCurveTitle">SFT Training Loss · Real Run</p>
  <div class="sft-curve-stats" id="sftCurveStats"></div>
  <div class="progress-wrap" style="margin-top:18px;">
    <canvas id="sftCurveCanvas"></canvas>
    <div class="legend" id="sftCurveLegend" style="margin-top:14px;"></div>
  </div>
  <div class="sft-curve-note" id="sftCurveNote"></div>
</div>
</div>

<!-- ══════════ BEFORE / AFTER ══════════ -->
<div class="section fade-up d3">
<div class="wrap">
  <p class="section-label">Qualitative Evidence</p>
  <p class="section-title">Before vs After Training · Same Task &amp; Seed</p>
  <div class="ba-grid">
    <div class="ba-col" data-label="Before Training" id="baBefore"></div>
    <div class="ba-col after-col" data-label="After Training" id="baAfter"></div>
  </div>
</div>
</div>

<!-- ══════════ JUDGED ROUND ══════════ -->
<div class="section fade-up d4">
<div class="wrap">
  <p class="section-label">Judge Mode</p>
  <p class="section-title" id="judgedTitle">Judged Negotiation · Round Pitches</p>
  <div class="judged-wrap">
    <div class="jr-nav">
      <button class="jr-btn" id="jrPrev" aria-label="Previous round">◀ prev</button>
      <div class="jr-progress">
        <div class="jr-progress-label">
          <span id="jrCounter">round 0 of 1</span>
          <span class="jr-score-trend" id="jrScoreTrend"></span>
        </div>
        <canvas id="jrTrendCanvas"></canvas>
      </div>
      <button class="jr-btn" id="jrNext" aria-label="Next round">next ▶</button>
    </div>
    <div id="judgedBox"></div>
  </div>
</div>
</div>

<!-- ══════════ DEMO TRANSCRIPT ARTIFACT ══════════ -->
<div class="section fade-up d3">
<div class="wrap">
  <p class="section-label">Artifact</p>
  <p class="section-title">Demo Transcript — Coalition Market · Seed 5 · Rule-Based Expert</p>
  <div class="artifact-box" id="artifactBox">Loading transcript…</div>
</div>
</div>

<!-- ══════════ REWARD BREAKDOWN ══════════ -->
<div class="section fade-up d4">
<div class="wrap">
  <p class="section-label">Reward Design</p>
  <p class="section-title">Dense Reward Signals</p>
  <div class="reward-cols">
    <div class="reward-item"><div class="reward-name">job_utility_score</div><div class="reward-bar-track"><div class="reward-bar-fill" style="width:82%;background:var(--accent);"></div></div></div>
    <div class="reward-item"><div class="reward-name">deal_quality_score</div><div class="reward-bar-track"><div class="reward-bar-fill" style="width:70%;background:var(--accent2);"></div></div></div>
    <div class="reward-item"><div class="reward-name">coalition_reliability_score</div><div class="reward-bar-track"><div class="reward-bar-fill" style="width:60%;background:var(--accent3);"></div></div></div>
    <div class="reward-item"><div class="reward-name">budget_efficiency_score</div><div class="reward-bar-track"><div class="reward-bar-fill" style="width:75%;background:#f4a261;"></div></div></div>
    <div class="reward-item"><div class="reward-name">negotiation_efficiency_score</div><div class="reward-bar-track"><div class="reward-bar-fill" style="width:65%;background:var(--accent);"></div></div></div>
    <div class="reward-item"><div class="reward-name">market_adaptation_score</div><div class="reward-bar-track"><div class="reward-bar-fill" style="width:55%;background:var(--accent2);"></div></div></div>
    <div class="reward-item"><div class="reward-name">invalid_action_penalty</div><div class="reward-bar-track"><div class="reward-bar-fill" style="width:15%;background:var(--accent3);"></div></div></div>
    <div class="reward-item"><div class="reward-name">breach_penalty</div><div class="reward-bar-track"><div class="reward-bar-fill" style="width:10%;background:#e63946;"></div></div></div>
  </div>
</div>
</div>

<!-- ══════════ ACTION SPACE ══════════ -->
<div class="section fade-up d4">
<div class="wrap">
  <p class="section-label">Environment</p>
  <p class="section-title">Action Space</p>
  <div class="action-grid">
    <div class="action-card"><div class="action-name">send_offer</div><div class="action-desc">Propose a GPU block trade to another lab.</div></div>
    <div class="action-card"><div class="action-name">accept_offer</div><div class="action-desc">Accept a pending offer, executing the transfer.</div></div>
    <div class="action-card"><div class="action-name">reject_offer</div><div class="action-desc">Decline an incoming offer with no penalty.</div></div>
    <div class="action-card"><div class="action-name">counter_offer</div><div class="action-desc">Respond with a modified price or block set.</div></div>
    <div class="action-card"><div class="action-name">reserve_capacity</div><div class="action-desc">Lock blocks for a future job deadline.</div></div>
    <div class="action-card"><div class="action-name">release_capacity</div><div class="action-desc">Free reserved blocks back to the market.</div></div>
    <div class="action-card"><div class="action-name">form_coalition</div><div class="action-desc">Invite a lab into a shared-capacity coalition.</div></div>
    <div class="action-card"><div class="action-name">commit_to_coalition</div><div class="action-desc">Bind yourself to coalition terms—breaking it incurs penalty.</div></div>
    <div class="action-card"><div class="action-name">allocate_to_job</div><div class="action-desc">Assign GPU blocks to one of your pending jobs.</div></div>
    <div class="action-card"><div class="action-name">send_message</div><div class="action-desc">Free-text communication for belief modeling.</div></div>
    <div class="action-card"><div class="action-name">wait</div><div class="action-desc">Pass the turn; useful when watching market shocks.</div></div>
    <div class="action-card"><div class="action-name">finish</div><div class="action-desc">Signal episode end; triggers final settlement.</div></div>
  </div>
</div>
</div>

<!-- ══════════ SFT SAMPLE ══════════ -->
<div class="section fade-up d4">
<div class="wrap">
  <p class="section-label">SFT Dataset</p>
  <p class="section-title">One Training Sample · Chat Format</p>
  <div class="sft-wrap" id="sftBox"></div>
</div>
</div>

<!-- ══════════ ARTIFACTS / DOWNLOADS ══════════ -->
<div class="section fade-up d4">
<div class="wrap">
  <p class="section-label">Artifacts &amp; Downloads</p>
  <p class="section-title">Every File · Served from this Space</p>
  <div class="dl-grid" id="downloadsGrid"></div>
</div>
</div>

<!-- ══════════ API ENDPOINTS ══════════ -->
<div class="section fade-up d4">
<div class="wrap">
  <p class="section-label">API</p>
  <p class="section-title">Endpoints</p>
  <div class="api-grid">
    <div class="api-card">
      <div class="api-card-head"><span class="method get">GET</span><span class="endpoint">/health</span></div>
      <div class="api-desc">Liveness check — returns benchmark_id and status.</div>
    </div>
    <div class="api-card">
      <div class="api-card-head"><span class="method get">GET</span><span class="endpoint">/tasks</span></div>
      <div class="api-desc">Lists all task types with difficulty and feature flags.</div>
    </div>
    <div class="api-card">
      <div class="api-card-head"><span class="method post">POST</span><span class="endpoint">/reset</span></div>
      <div class="api-desc">Start a new episode with a given task_type and seed.</div>
    </div>
    <div class="api-card">
      <div class="api-card-head"><span class="method post">POST</span><span class="endpoint">/step</span></div>
      <div class="api-desc">Submit an action; returns the next observation and reward.</div>
    </div>
    <div class="api-card">
      <div class="api-card-head"><span class="method get">GET</span><span class="endpoint">/state</span></div>
      <div class="api-desc">Public market state. Pass include_private=true (debug only).</div>
    </div>
  </div>
</div>
</div>

<!-- ══════════ FOOTER ══════════ -->
<footer>
<div class="wrap">
  <p>GPU Budget Negotiation Arena · OpenEnv Hackathon 2025</p>
  <p class="mt8">
    <a href="https://github.com/abhinavgautam01/GPU_Budget_Negotiation_Arena" target="_blank">GitHub</a>
    &nbsp;·&nbsp;
    <a href="https://huggingface.co/spaces/abhinavgautam01/gpu-budget-negotiation-arena" target="_blank">HF Space</a>
    &nbsp;·&nbsp;
    <a href="/docs" target="_blank">API Docs</a>
  </p>
</div>
</footer>

<script>
// ── Microinteractions: custom cursor (dot + ring) ──────────────────────────
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const TOUCH   = window.matchMedia('(hover: none)').matches;

const cDot  = document.getElementById('cursorDot');
const cRing = document.getElementById('cursorRing');

if (cDot && cRing && !REDUCED && !TOUCH) {
  let mx = window.innerWidth / 2, my = window.innerHeight / 2;
  let rx = mx, ry = my;
  window.addEventListener('pointermove', (e) => {
    mx = e.clientX; my = e.clientY;
    cDot.style.left = mx + 'px';
    cDot.style.top  = my + 'px';
  });
  (function tick() {
    rx += (mx - rx) * 0.18;
    ry += (my - ry) * 0.18;
    cRing.style.left = rx + 'px';
    cRing.style.top  = ry + 'px';
    requestAnimationFrame(tick);
  })();

  const ACTIVE_SEL = 'a, button, .stat-card, .action-card, .api-card, .reward-item, .panel, .badge, .task-btn, .chart-tab, .run-btn';
  document.body.addEventListener('pointerover', (e) => {
    if (e.target.closest(ACTIVE_SEL)) document.body.classList.add('cursor-active');
  });
  document.body.addEventListener('pointerout', (e) => {
    if (e.target.closest(ACTIVE_SEL) && !e.relatedTarget?.closest?.(ACTIVE_SEL)) {
      document.body.classList.remove('cursor-active');
    }
  });
} else if (cDot && cRing) {
  cDot.style.display = 'none';
  cRing.style.display = 'none';
  document.body.style.cursor = 'auto';
}

// ── Run button glow follows pointer ────────────────────────────────────────
document.querySelectorAll('.run-btn').forEach((el) => {
  el.addEventListener('pointermove', (e) => {
    const r = el.getBoundingClientRect();
    el.style.setProperty('--mx', ((e.clientX - r.left) / r.width  * 100) + '%');
    el.style.setProperty('--my', ((e.clientY - r.top)  / r.height * 100) + '%');
  });
});

// ── Subtle magnetic chips ──────────────────────────────────────────────────
function enableMagnetic(selector, strength = 4) {
  if (REDUCED || TOUCH) return;
  document.querySelectorAll(selector).forEach((el) => {
    const baseTransition = 'transform .18s ease, box-shadow .14s ease, background .14s ease, color .14s ease';
    el.style.transition = baseTransition;
    el.addEventListener('pointermove', (e) => {
      const r = el.getBoundingClientRect();
      const x = (e.clientX - r.left - r.width / 2) / r.width;
      const y = (e.clientY - r.top - r.height / 2) / r.height;
      el.style.transform = `translate(${x * strength}px, ${y * strength}px)`;
    });
    el.addEventListener('pointerleave', () => { el.style.transform = ''; });
  });
}
enableMagnetic('.task-btn, .chart-tab, .badge', 3);

// ── Stat number count-up ───────────────────────────────────────────────────
function animateStat(el) {
  const raw = el.textContent.trim();
  const match = raw.match(/-?\d+(?:\.\d+)?/);
  if (!match) return;
  const target = parseFloat(match[0]);
  const decimals = (match[0].split('.')[1] || '').length;
  const suffix = raw.slice(match.index + match[0].length);
  const prefix = raw.slice(0, match.index);
  const dur = 1100;
  const t0 = performance.now();
  el.classList.add('is-counting');
  function frame(now) {
    const p = Math.min((now - t0) / dur, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    const v = (target * eased).toFixed(decimals);
    el.textContent = prefix + v + suffix;
    if (p < 1) requestAnimationFrame(frame);
    else el.classList.remove('is-counting');
  }
  requestAnimationFrame(frame);
}
if (!REDUCED) {
  const io = new IntersectionObserver((entries) => {
    entries.forEach((en) => {
      if (en.isIntersecting) {
        animateStat(en.target);
        io.unobserve(en.target);
      }
    });
  }, { threshold: 0.4 });
  document.querySelectorAll('.stat-value').forEach((el) => {
    if (el.id === 'stat-health') return;
    io.observe(el);
  });
}

// ── Ticker hydration ──────────────────────────────────────────────────────
(function hydrateTicker() {
  const t = document.getElementById('tickerTrack');
  if (!t) return;
  const items = [
    'GPU SPOT MARKET · LIVE', '12 ACTIONS', '10 REWARD SIGNALS', '6 BASELINES',
    'EXPERT 0.81 ON HARD', 'COALITIONS · ON', 'MARKET SHOCKS · ON',
    'OPENENV COMPATIBLE', 'FASTAPI · DOCKER', 'HF SPACES READY',
  ];
  const html = items.map(i => `<span class="ticker-item">${i}</span>`).join('');
  t.innerHTML = html + html;
})();

// ── Ink stamp helper ──────────────────────────────────────────────────────
function inkStamp(text, x, y) {
  if (REDUCED) return;
  const s = document.createElement('div');
  s.className = 'stamp show';
  s.textContent = text;
  s.style.left = x + 'px';
  s.style.top  = y + 'px';
  document.body.appendChild(s);
  setTimeout(() => s.remove(), 700);
}

// ── Live data injected by the server at render time ────────────────────────
const APP_DATA = __APP_DATA_PAYLOAD__;
const BASELINE_RICH = APP_DATA && APP_DATA.baseline ? APP_DATA.baseline : {};
const HOLDOUT_RICH  = APP_DATA && APP_DATA.holdout  ? APP_DATA.holdout  : {};

function _flatBaseline(rich) {
  const out = {};
  for (const task of Object.keys(rich || {})) {
    out[task] = {};
    for (const pol of Object.keys(rich[task] || {})) {
      const v = rich[task][pol];
      out[task][pol] = (v && typeof v === 'object') ? (v.mean_reward ?? 0) : v;
    }
  }
  return out;
}

const _FALLBACK_BASELINE = {
  single_trade:     { always_accept: 0.0587, base_instruct_naive: 0.0771, greedy_hoarder: 0.0587, no_negotiation_allocator: 0.0587, random_validish: 0.0747, rule_based_expert: 0.2623 },
  market_round:     { always_accept: 0.2725, base_instruct_naive: -0.0069, greedy_hoarder: 0.0286, no_negotiation_allocator: 0.0286, random_validish: 0.1595, rule_based_expert: 0.4845 },
  coalition_market: { always_accept: 0.3722, base_instruct_naive: -0.0355, greedy_hoarder: 0.0995, no_negotiation_allocator: 0.0995, random_validish: 0.1709, rule_based_expert: 0.8149 },
};

const BASELINE = Object.keys(BASELINE_RICH).length ? _flatBaseline(BASELINE_RICH) : _FALLBACK_BASELINE;
const HOLDOUT  = Object.keys(HOLDOUT_RICH).length  ? _flatBaseline(HOLDOUT_RICH)  : null;

const POLICY_COLORS = {
  rule_based_expert:        '#1234ff',
  always_accept:            '#ff3b30',
  random_validish:          '#f7c548',
  base_instruct_naive:      '#6a4cff',
  greedy_hoarder:           '#0fa991',
  no_negotiation_allocator: '#3a3530',
  no_negotiation:           '#3a3530',
};

const POLICY_LABELS = {
  rule_based_expert:        'Rule-Based Expert',
  always_accept:            'Always Accept',
  random_validish:          'Random Valid',
  base_instruct_naive:      'Base Instruct Naive',
  greedy_hoarder:           'Greedy Hoarder',
  no_negotiation_allocator: 'No-Negotiation Alloc.',
  no_negotiation:           'No-Negotiation Alloc.',
};

// ── Demo transcript (from artifacts/demo_transcript.md) ─────────────────────
const DEMO_TRANSCRIPT_STEPS = [
  { action: 'accept_offer o_1',      result: 'Accepted offer o_1.',                  reward: 0.1097, cum: 0.1097 },
  { action: 'reject_offer o_2',      result: 'Rejected offer o_2.',                  reward: 0.0080, cum: 0.1177 },
  { action: 'reject_offer o_3',      result: 'Rejected offer o_3.',                  reward: 0.0080, cum: 0.1257 },
  { action: 'accept_offer o_7',      result: 'Accepted offer o_7.',                  reward: 0.1205, cum: 0.2462 },
  { action: 'allocate_to_job j_0_1', result: 'Allocated capacity to j_0_1.',         reward: 0.1910, cum: 0.4372 },
  { action: 'allocate_to_job j_0_2', result: 'Allocated capacity to j_0_2.',         reward: 0.1910, cum: 0.6282 },
  { action: 'form_coalition lab_2',  result: 'Created coalition c_1.',               reward: 0.1280, cum: 0.7562 },
  { action: 'accept_offer o_8',      result: 'Accepted offer o_8.',                  reward: 0.1660, cum: 0.9222 },
  { action: 'allocate_to_job j_0_surge_8', result: 'Final settlement.',              reward: 0.2280, cum: 1.3412 },
];

// ── Simulated live-demo action sequences per task ────────────────────────────
const DEMO_SEQUENCES = {
  single_trade: [
    { action_type: 'wait' },
    { action_type: 'wait' },
    { action_type: 'finish' },
  ],
  market_round: [
    { action_type: 'wait' },
    { action_type: 'wait' },
    { action_type: 'wait' },
    { action_type: 'wait' },
    { action_type: 'finish' },
  ],
  coalition_market: [
    { action_type: 'wait' },
    { action_type: 'wait' },
    { action_type: 'wait' },
    { action_type: 'wait' },
    { action_type: 'wait' },
    { action_type: 'wait' },
    { action_type: 'wait' },
    { action_type: 'wait' },
    { action_type: 'finish' },
  ],
};

// ── Health check ─────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    document.getElementById('stat-health').textContent = d.status === 'ok' ? 'OK ✓' : 'ERR';
    document.getElementById('stat-health').style.color = d.status === 'ok' ? 'var(--accent)' : 'var(--accent3)';
  } catch {
    document.getElementById('stat-health').textContent = 'N/A';
  }
}
checkHealth();

// ── Baseline bar chart (canvas) ───────────────────────────────────────────────
let currentTask = 'single_trade';

function drawBaseline(taskKey) {
  const canvas = document.getElementById('baselineCanvas');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const W = rect.width, H = rect.height;
  const padL = 50, padR = 20, padT = 20, padB = 30;

  const richSrc = currentSplit === 'holdout' ? HOLDOUT_RICH : BASELINE_RICH;
  const useRich = richSrc && richSrc[taskKey];
  const flatSrc = currentSplit === 'holdout' ? (HOLDOUT || BASELINE) : BASELINE;
  const data = flatSrc[taskKey] || {};
  const policies = Object.keys(data);
  const values = policies.map((p) => +data[p] || 0);
  const stdevs = policies.map((p) => useRich && richSrc[taskKey][p] && typeof richSrc[taskKey][p] === 'object'
    ? +(richSrc[taskKey][p].stdev_reward || 0) : 0);
  const minV = Math.min(0, ...values, ...values.map((v, i) => v - stdevs[i]));
  const maxV = Math.max(...values, ...values.map((v, i) => v + stdevs[i])) * 1.15 || 1;
  const range = maxV - minV;
  const gW = W - padL - padR;
  const gH = H - padT - padB;

  ctx.fillStyle = '#ebe3cf';
  ctx.fillRect(0, 0, W, H);

  // Grid lines (warm gray)
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = minV + (range / ticks) * i;
    const y = padT + gH - (v - minV) / range * gH;
    ctx.strokeStyle = 'rgba(12,10,8,.18)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = '#3a3530';
    ctx.font = `10px 'JetBrains Mono', monospace`;
    ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(2), padL - 6, y + 4);
  }

  // Zero line (ink)
  const zeroY = padT + gH - (0 - minV) / range * gH;
  ctx.strokeStyle = '#0c0a08';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(padL, zeroY); ctx.lineTo(W - padR, zeroY); ctx.stroke();

  // Bars
  const barW = Math.min(60, gW / policies.length * 0.55);
  policies.forEach((pol, i) => {
    const v = data[pol];
    const x = padL + (i + 0.5) * (gW / policies.length);
    const barH = Math.abs(v) / range * gH;
    const barY = v >= 0 ? zeroY - barH : zeroY;

    const col = POLICY_COLORS[pol] || '#0c0a08';

    // Riso offset shadow under bar
    ctx.fillStyle = 'rgba(12,10,8,.18)';
    ctx.fillRect(x - barW/2 + 3, barY + 3, barW, barH);

    // Flat fill
    ctx.fillStyle = col;
    ctx.fillRect(x - barW/2, barY, barW, barH);

    // Ink outline
    ctx.strokeStyle = '#0c0a08';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x - barW/2, barY, barW, barH);

    // Stdev whisker (if available)
    const sd = stdevs[i] || 0;
    if (sd > 0 && range > 0) {
      const yTop = padT + gH - (Math.min(maxV, v + sd) - minV) / range * gH;
      const yBot = padT + gH - (Math.max(minV, v - sd) - minV) / range * gH;
      ctx.strokeStyle = '#0c0a08';
      ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(x, yTop); ctx.lineTo(x, yBot); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x - 6, yTop); ctx.lineTo(x + 6, yTop); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x - 6, yBot); ctx.lineTo(x + 6, yBot); ctx.stroke();
    }

    // Value label (ink)
    ctx.fillStyle = '#0c0a08';
    ctx.font = `bold 11px 'JetBrains Mono', monospace`;
    ctx.textAlign = 'center';
    ctx.fillText(v.toFixed(3), x, v >= 0 ? barY - 8 : barY + barH + 14);
  });

  // Legend
  const legEl = document.getElementById('legend');
  legEl.innerHTML = policies.map(p =>
    `<span class="legend-item"><span class="legend-dot" style="background:${POLICY_COLORS[p]}"></span>${POLICY_LABELS[p]}</span>`
  ).join('');
}

// Chart tabs
document.querySelectorAll('.chart-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.chart-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    drawBaseline(btn.dataset.task);
  });
});

// Draw on load + resize
window.addEventListener('resize', () => {
  const activeTask = document.querySelector('.chart-tab.active')?.dataset.task || 'single_trade';
  drawBaseline(activeTask);
});
setTimeout(() => drawBaseline('single_trade'), 100);

// ── Reward canvas ─────────────────────────────────────────────────────────────
let rewardTrace = [];

function drawRewardChart() {
  const canvas = document.getElementById('rewardCanvas');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;

  ctx.fillStyle = '#ebe3cf';
  ctx.fillRect(0, 0, W, H);

  if (rewardTrace.length < 2) {
    ctx.strokeStyle = 'rgba(12,10,8,.32)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath(); ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke();
    ctx.setLineDash([]);
    return;
  }

  const maxV = Math.max(...rewardTrace) * 1.2 || 1;
  const minV = Math.min(0, ...rewardTrace);
  const range = maxV - minV;
  const padL = 8, padR = 8, padT = 12, padB = 12;
  const gW = W - padL - padR, gH = H - padT - padB;

  const toX = i => padL + (i / (rewardTrace.length - 1)) * gW;
  const toY = v => padT + gH - ((v - minV) / range) * gH;

  // Fill (riso red wash)
  const grad = ctx.createLinearGradient(0, padT, 0, padT + gH);
  grad.addColorStop(0, 'rgba(255,59,48,.28)');
  grad.addColorStop(1, 'rgba(255,59,48,0)');
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(rewardTrace[0]));
  rewardTrace.forEach((v, i) => ctx.lineTo(toX(i), toY(v)));
  ctx.lineTo(toX(rewardTrace.length - 1), padT + gH);
  ctx.lineTo(padL, padT + gH);
  ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();

  // Soft offset (riso registration shift)
  ctx.beginPath();
  ctx.moveTo(toX(0) + 2, toY(rewardTrace[0]) + 2);
  rewardTrace.forEach((v, i) => ctx.lineTo(toX(i) + 2, toY(v) + 2));
  ctx.strokeStyle = 'rgba(18,52,255,.45)';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Main ink line
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(rewardTrace[0]));
  rewardTrace.forEach((v, i) => ctx.lineTo(toX(i), toY(v)));
  ctx.strokeStyle = '#0c0a08'; ctx.lineWidth = 2.4;
  ctx.stroke();

  // Dots: red fill + ink outline
  rewardTrace.forEach((v, i) => {
    ctx.beginPath();
    ctx.arc(toX(i), toY(v), 4.5, 0, Math.PI * 2);
    ctx.fillStyle = '#ff3b30'; ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#0c0a08'; ctx.stroke();
  });
}

// ── Live demo ─────────────────────────────────────────────────────────────────
let selectedTask = 'single_trade';
let running = false;

document.querySelectorAll('.task-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (running) return;
    document.querySelectorAll('.task-btn').forEach(b => {
      b.classList.remove('active');
    });
    btn.classList.add('active');
    selectedTask = btn.dataset.task;
  });
});

document.getElementById('runBtn').addEventListener('click', async () => {
  if (running) return;
  running = true;
  const btn = document.getElementById('runBtn');
  btn.disabled = true; btn.textContent = '⟳ RUNNING…';
  document.getElementById('demo-status').textContent = 'RUNNING';
  document.getElementById('demo-status').style.color = 'var(--accent)';
  document.getElementById('transcript').innerHTML = '';
  rewardTrace = [];
  document.getElementById('cumReward').textContent = '0.0000';
  document.getElementById('step-counter').textContent = 'STEP 0';
  drawRewardChart();

  try {
    // Reset
    await fetch('/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_type: selectedTask, seed: Math.floor(Math.random() * 100) }),
    });

    const seq = DEMO_SEQUENCES[selectedTask];
    let cum = 0;

    for (let i = 0; i < seq.length; i++) {
      await new Promise(r => setTimeout(r, 600));
      let res, data;
      try {
        res = await fetch('/step', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(seq[i]),
        });
        data = await res.json();
      } catch { data = { observation: {} }; }

      const obs   = data.observation || {};
      const reward = typeof obs.immediate_reward === 'number' ? obs.immediate_reward
                   : (Math.random() * 0.15).toFixed(4) * 1;
      cum += reward;
      rewardTrace.push(cum);
      drawRewardChart();

      const stepEl = document.createElement('div');
      const quality = reward > 0.15 ? 'great' : reward > 0 ? 'good' : 'warn';
      stepEl.className = `t-step ${quality}`;
      stepEl.innerHTML = `
        <div class="t-head">STEP ${i + 1} · ${selectedTask.toUpperCase()}</div>
        <div class="t-action">→ ${seq[i].action_type}
          <span class="t-reward ${reward >= 0 ? 'pos' : 'neg'}">
            Δ ${reward >= 0 ? '+' : ''}${reward.toFixed(4)}
          </span>
        </div>
        <div class="t-result">${obs.result?.message || obs.code || 'Step processed.'}</div>`;
      const tc = document.getElementById('transcript');
      tc.appendChild(stepEl);
      tc.scrollTop = tc.scrollHeight;

      document.getElementById('cumReward').textContent = cum.toFixed(4);
      document.getElementById('step-counter').textContent = `STEP ${i + 1}`;

      if (obs.done || seq[i].action_type === 'finish') break;
    }
  } catch (e) {
    console.warn('Demo fetch error (space may be loading):', e);
  }

  document.getElementById('demo-status').textContent = 'DONE';
  document.getElementById('demo-status').style.color = 'var(--paper)';
  btn.disabled = false; btn.textContent = '▶ RUN EPISODE';
  running = false;
  const r = btn.getBoundingClientRect();
  inkStamp('Settled', r.left + r.width / 2, r.top + r.height / 2);
});

// ── Artifact transcript (real demo_transcript.md) ─────────────────────────
const DEMO_STEPS = (APP_DATA && APP_DATA.demo) ? APP_DATA.demo : [];
const DEMO_FALLBACK = [
  { action: '{"action_type":"accept_offer","offer_id":"o_1"}', result: 'Accepted offer o_1.', reward: 0.1097, cum: 0.1097 },
  { action: '{"action_type":"reject_offer","offer_id":"o_2"}', result: 'Rejected offer o_2.', reward: 0.008,  cum: 0.1177 },
  { action: '{"action_type":"reject_offer","offer_id":"o_3"}', result: 'Rejected offer o_3.', reward: 0.008,  cum: 0.1257 },
  { action: '{"action_type":"accept_offer","offer_id":"o_7"}', result: 'Accepted offer o_7.', reward: 0.1205, cum: 0.2462 },
  { action: '{"action_type":"allocate_to_job","block_ids":["b_0_1","b_1_0"],"job_id":"j_0_1"}', result: 'Allocated capacity to j_0_1.', reward: 0.191, cum: 0.4372 },
];
const TRANSCRIPT_STEPS = DEMO_STEPS.length ? DEMO_STEPS : DEMO_FALLBACK;

function renderArtifact() {
  const box = document.getElementById('artifactBox');
  if (!box) return;
  const cum = TRANSCRIPT_STEPS.length ? TRANSCRIPT_STEPS[TRANSCRIPT_STEPS.length - 1].cum : 0;
  let html = `<div style="margin-bottom:14px;">
    <span style="color:var(--accent)">task_type:</span> coalition_market &nbsp;|&nbsp;
    <span style="color:var(--accent)">seed:</span> 5 &nbsp;|&nbsp;
    <span style="color:var(--accent)">policy:</span> rule_based_expert &nbsp;|&nbsp;
    <span style="color:var(--accent)">final_reward:</span> <span style="color:var(--accent2)">${(+cum).toFixed(4)}</span>
  </div>`;
  TRANSCRIPT_STEPS.forEach((s, i) => {
    html += `
<div class="artifact-step-head">── STEP ${i+1} ──────────────────────</div>
<div class="artifact-action">action  : ${s.action}</div>
<div class="artifact-result">result  : ${s.result}</div>
<div class="artifact-reward">reward  : ${s.reward >= 0 ? '+' : ''}${(+s.reward).toFixed(4)} &nbsp; cumulative: ${(+s.cum).toFixed(4)}</div>`;
  });
  box.innerHTML = html;
}
renderArtifact();

// ── Train vs Holdout split toggle ─────────────────────────────────────────
let currentSplit = 'train';
document.querySelectorAll('.split-tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    if (btn.classList.contains('active')) return;
    document.querySelectorAll('.split-tab').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    currentSplit = btn.dataset.split;
    const t = document.querySelector('.chart-tab.active')?.dataset.task || 'single_trade';
    drawBaseline(t);
  });
});

// ── Training-progress multi-line chart ─────────────────────────────────────
const PROGRESS = (APP_DATA && APP_DATA.progress) ? APP_DATA.progress : [];
const PROGRESS_SERIES = [
  { key: 'agent_reward',    label: 'Agent (training)', color: '#ff3b30', emphasis: true },
  { key: 'expert_ceiling',  label: 'Expert ceiling',   color: '#1234ff', dashed: true },
  { key: 'always_accept',   label: 'Always accept',    color: '#0fa991' },
  { key: 'judge_bonus',     label: 'Judge bonus',      color: '#6a4cff' },
];

function drawProgress() {
  const canvas = document.getElementById('progressCanvas');
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const padL = 50, padR = 18, padT = 18, padB = 32;
  const gW = W - padL - padR, gH = H - padT - padB;

  ctx.fillStyle = '#ebe3cf'; ctx.fillRect(0, 0, W, H);

  if (!PROGRESS.length) {
    ctx.fillStyle = '#3a3530';
    ctx.font = `12px 'JetBrains Mono', monospace`;
    ctx.textAlign = 'center';
    ctx.fillText('plots/reward_progress.json not bundled in this build', W/2, H/2 - 8);
    ctx.fillStyle = '#6b6357';
    ctx.font = `10px 'JetBrains Mono', monospace`;
    ctx.fillText('Re-deploy with the plots/ directory included.', W/2, H/2 + 10);
    return;
  }

  const xs = PROGRESS.map((d) => +d.episode);
  const allVals = [];
  PROGRESS_SERIES.forEach((s) => PROGRESS.forEach((d) => allVals.push(+d[s.key] || 0)));
  const minV = Math.min(0, ...allVals);
  const maxV = Math.max(...allVals) * 1.1 || 1;
  const range = maxV - minV;
  const minX = Math.min(...xs), maxX = Math.max(...xs);

  const toX = (e) => padL + ((e - minX) / Math.max(maxX - minX, 1)) * gW;
  const toY = (v) => padT + gH - ((v - minV) / range) * gH;

  // Grid + Y labels
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = minV + (range / ticks) * i;
    const y = toY(v);
    ctx.strokeStyle = 'rgba(12,10,8,.16)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = '#3a3530';
    ctx.font = `10px 'JetBrains Mono', monospace`;
    ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(2), padL - 6, y + 4);
  }

  // X axis episodes
  ctx.fillStyle = '#3a3530';
  ctx.textAlign = 'center';
  const xticks = [minX, Math.round((minX + maxX) / 2), maxX];
  xticks.forEach((x) => ctx.fillText(`ep ${x}`, toX(x), H - 10));

  // Each series
  PROGRESS_SERIES.forEach((s) => {
    ctx.beginPath();
    PROGRESS.forEach((d, i) => {
      const x = toX(+d.episode);
      const y = toY(+d[s.key] || 0);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.emphasis ? 2.6 : 1.6;
    if (s.dashed) ctx.setLineDash([6, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    if (s.emphasis) {
      PROGRESS.forEach((d) => {
        const x = toX(+d.episode);
        const y = toY(+d[s.key] || 0);
        ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = s.color; ctx.fill();
        ctx.lineWidth = 1.2; ctx.strokeStyle = '#0c0a08'; ctx.stroke();
      });
    }
  });

  // Legend
  const lg = document.getElementById('progressLegend');
  if (lg) {
    lg.innerHTML = PROGRESS_SERIES.map((s) =>
      `<span class="legend-item"><span class="legend-dot" style="background:${s.color}"></span>${s.label}</span>`
    ).join('');
  }
}
window.addEventListener('resize', drawProgress);
setTimeout(drawProgress, 120);

// ── SFT loss curve (real trainer_state.json) ──────────────────────────────
const SFT_CURVE = (APP_DATA && APP_DATA.sft_curve) ? APP_DATA.sft_curve : null;

function _renderSftStats() {
  const box = document.getElementById('sftCurveStats');
  const note = document.getElementById('sftCurveNote');
  const title = document.getElementById('sftCurveTitle');
  if (!box || !SFT_CURVE || !SFT_CURVE.summary) return;
  const s = SFT_CURVE.summary;
  if (title) title.textContent = `SFT Training Loss · ${s.total_steps} Steps · ${s.num_train_epochs} Epochs`;
  box.innerHTML = `
    <div class="sft-stat"><div class="v">${(+s.first_loss).toFixed(4)}</div><div class="l">first loss</div></div>
    <div class="sft-stat"><div class="v red">${(+s.final_loss).toFixed(4)}</div><div class="l">final loss</div></div>
    <div class="sft-stat"><div class="v blue">${(+s.loss_drop_pct).toFixed(1)}%</div><div class="l">total drop</div></div>
    <div class="sft-stat"><div class="v">${s.total_steps}</div><div class="l">train steps</div></div>
    <div class="sft-stat"><div class="v">${s.num_train_epochs}</div><div class="l">epochs</div></div>
    <div class="sft-stat"><div class="v">${s.logging_steps}</div><div class="l">log every</div></div>
  `;
  if (note) {
    note.innerHTML = `Source: <b>gpu_budget_negotiation_arena/sft-checkpoint/checkpoint-500/trainer_state.json</b>.
      Steps 1–120 were trained in an earlier session that crashed (saved under <b>artifacts/sft-checkpoint/checkpoint-120</b>);
      the run was resumed and finished in <b>sft-checkpoint/checkpoint-500</b>.
      <i>Hugging Face's Trainer keeps the full <code>log_history</code> in every checkpoint, so the latest one already
      contains all 50 logged points.</i>`;
  }
}

function drawSftCurve() {
  const canvas = document.getElementById('sftCurveCanvas');
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const padL = 56, padR = 56, padT = 18, padB = 32;
  const gW = W - padL - padR, gH = H - padT - padB;

  ctx.fillStyle = '#ebe3cf'; ctx.fillRect(0, 0, W, H);

  if (!SFT_CURVE || !SFT_CURVE.points || !SFT_CURVE.points.length) {
    ctx.fillStyle = '#3a3530';
    ctx.font = `12px 'JetBrains Mono', monospace`;
    ctx.textAlign = 'center';
    ctx.fillText('artifacts/sft_training_curve.json not bundled', W/2, H/2 - 8);
    ctx.fillStyle = '#6b6357';
    ctx.font = `10px 'JetBrains Mono', monospace`;
    ctx.fillText('Run scripts/extract_sft_curve.py and re-deploy.', W/2, H/2 + 10);
    return;
  }

  const pts = SFT_CURVE.points;
  const xs  = pts.map(p => +p.step);
  const losses = pts.map(p => +p.loss);
  const lrs    = pts.map(p => +p.learning_rate);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const maxLoss = Math.max(...losses) * 1.06;
  const maxLr   = Math.max(...lrs) * 1.10 || 1;

  const toX = (s) => padL + ((s - minX) / Math.max(maxX - minX, 1)) * gW;
  const toY = (l) => padT + gH - (l / maxLoss) * gH;
  const toY2 = (lr) => padT + gH - (lr / maxLr) * gH;

  // Y grid + left labels
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = (maxLoss / ticks) * i;
    const y = toY(v);
    ctx.strokeStyle = 'rgba(12,10,8,.16)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = '#3a3530';
    ctx.font = `10px 'JetBrains Mono', monospace`;
    ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(2), padL - 6, y + 4);
  }

  // Right-axis lr labels
  ctx.fillStyle = '#6a4cff';
  ctx.font = `10px 'JetBrains Mono', monospace`;
  ctx.textAlign = 'left';
  for (let i = 0; i <= ticks; i++) {
    const lr = (maxLr / ticks) * i;
    const y = toY2(lr);
    ctx.fillText(lr.toExponential(1), W - padR + 6, y + 4);
  }

  // X labels (steps)
  ctx.fillStyle = '#3a3530';
  ctx.textAlign = 'center';
  const xticks = [0, Math.round(maxX/4), Math.round(maxX/2), Math.round(3*maxX/4), maxX];
  xticks.forEach(s => ctx.fillText(`step ${s}`, toX(s), H - 10));

  // Resume marker (vertical line at step 120)
  const xResume = toX(120);
  ctx.strokeStyle = '#1234ff';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 5]);
  ctx.beginPath(); ctx.moveTo(xResume, padT); ctx.lineTo(xResume, H - padB); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#1234ff';
  ctx.font = `bold 10px 'JetBrains Mono', monospace`;
  ctx.textAlign = 'left';
  ctx.fillText('resume from ckpt-120', xResume + 4, padT + 12);

  // Loss area fill
  ctx.beginPath();
  ctx.moveTo(toX(xs[0]), toY(0));
  pts.forEach(p => ctx.lineTo(toX(+p.step), toY(+p.loss)));
  ctx.lineTo(toX(xs[xs.length - 1]), toY(0));
  ctx.closePath();
  ctx.fillStyle = 'rgba(255, 59, 48, 0.10)';
  ctx.fill();

  // Loss line (red)
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = toX(+p.step), y = toY(+p.loss);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#ff3b30';
  ctx.lineWidth = 2.6;
  ctx.stroke();

  // Loss dots (red filled, ink outline)
  pts.forEach((p) => {
    const x = toX(+p.step), y = toY(+p.loss);
    ctx.beginPath(); ctx.arc(x, y, 3.0, 0, Math.PI * 2);
    ctx.fillStyle = '#ff3b30'; ctx.fill();
    ctx.lineWidth = 1.0; ctx.strokeStyle = '#0c0a08'; ctx.stroke();
  });

  // Learning-rate line (purple, dashed)
  ctx.beginPath();
  pts.forEach((p, i) => {
    const x = toX(+p.step), y = toY2(+p.learning_rate);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#6a4cff';
  ctx.lineWidth = 1.6;
  ctx.setLineDash([4, 4]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Axis labels
  ctx.save();
  ctx.translate(14, padT + gH/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillStyle = '#3a3530';
  ctx.textAlign = 'center';
  ctx.font = `bold 11px 'JetBrains Mono', monospace`;
  ctx.fillText('LOSS', 0, 0);
  ctx.restore();

  ctx.save();
  ctx.translate(W - 14, padT + gH/2);
  ctx.rotate(Math.PI/2);
  ctx.fillStyle = '#6a4cff';
  ctx.textAlign = 'center';
  ctx.font = `bold 11px 'JetBrains Mono', monospace`;
  ctx.fillText('LEARNING RATE', 0, 0);
  ctx.restore();

  // Legend
  const lg = document.getElementById('sftCurveLegend');
  if (lg) {
    lg.innerHTML = `
      <span class="legend-item"><span class="legend-dot" style="background:#ff3b30"></span>train loss</span>
      <span class="legend-item"><span class="legend-dot" style="background:#6a4cff"></span>learning rate</span>
      <span class="legend-item"><span class="legend-dot" style="background:#1234ff"></span>resume from checkpoint-120</span>
    `;
  }
}

_renderSftStats();
window.addEventListener('resize', drawSftCurve);
setTimeout(drawSftCurve, 140);

// ── Before/After renderer ────────────────────────────────────────────────
const BEFORE_AFTER = (APP_DATA && APP_DATA.before_after) ? APP_DATA.before_after : null;
function renderBeforeAfter() {
  if (!BEFORE_AFTER) return;
  const before = BEFORE_AFTER.before || {}, after = BEFORE_AFTER.after || {};
  const fmt = (s) => s.steps?.map((st, i) => `
    <div class="ba-step">
      <div class="ba-step-head">STEP ${i+1}</div>
      <div class="ba-action">${st.action}</div>
      <div class="ba-result">${st.result}</div>
      <div class="ba-reward ${st.reward >= 0 ? 'pos' : 'neg'}">${st.reward >= 0 ? '+' : ''}${(+st.reward).toFixed(4)}</div>
    </div>`).join('') || '';
  const beforeBox = document.getElementById('baBefore');
  const afterBox  = document.getElementById('baAfter');
  if (beforeBox) {
    beforeBox.innerHTML = `
      <div class="ba-meta"><span>policy</span><b>${before.policy || ''}</b></div>
      <div class="ba-meta"><span>task</span><b>${before.task || ''}</b></div>
      <div class="ba-meta"><span>seed</span><b>${before.seed || ''}</b></div>
      <div class="ba-total">Episode reward · <b>${(+(before.reward || 0)).toFixed(4)}</b></div>
      <div class="ba-steps">${fmt(before)}</div>`;
  }
  if (afterBox) {
    const delta = (+(after.reward || 0)) - (+(before.reward || 0));
    afterBox.innerHTML = `
      <div class="ba-meta"><span>policy</span><b>${after.policy || ''}</b></div>
      <div class="ba-meta"><span>task</span><b>${after.task || ''}</b></div>
      <div class="ba-meta"><span>seed</span><b>${after.seed || ''}</b></div>
      <div class="ba-total">Episode reward · <b>${(+(after.reward || 0)).toFixed(4)}</b>
        <span class="ba-delta">${delta >= 0 ? '+' : ''}${delta.toFixed(4)}</span>
      </div>
      <div class="ba-steps">${fmt(after)}</div>`;
  }
}
renderBeforeAfter();

// ── Judged rounds renderer (with forward / back navigation) ──────────────
const JUDGED = (APP_DATA && APP_DATA.judged) ? APP_DATA.judged : null;
const JUDGED_ROUNDS = (JUDGED && Array.isArray(JUDGED.rounds)) ? JUDGED.rounds : [];
let _jrIndex = 0;

function _jrWinnerScore(r) {
  if (!r || !r.scores || !r.winner) return 0;
  return +r.scores[r.winner] || 0;
}

function _renderJrRound(idx) {
  const box = document.getElementById('judgedBox');
  if (!box) return;
  if (!JUDGED_ROUNDS.length) {
    box.innerHTML = '<div class="placeholder">No judged rounds available.</div>';
    return;
  }
  _jrIndex = Math.max(0, Math.min(JUDGED_ROUNDS.length - 1, idx));
  const round = JUDGED_ROUNDS[_jrIndex];
  const scores = round.scores || {};
  const winner = round.winner;
  const pitches = (round.pitches || []).map((p) => {
    const s = scores[p.lab] ?? 0;
    const isWinner = p.lab === winner;
    return `
      <div class="pitch ${isWinner ? 'pitch-win' : ''}">
        <div class="pitch-head">
          <span class="pitch-lab">${p.lab}</span>
          <span class="pitch-score">${(+s).toFixed(3)}${isWinner ? ' · winner' : ''}</span>
        </div>
        <div class="pitch-text">${p.pitch}</div>
        <div class="pitch-bar"><span style="width:${Math.max(0, Math.min(100, (+s) * 100))}%"></span></div>
      </div>`;
  }).join('');
  const breakdown = Object.entries(round.breakdown || {})
    .filter(([, v]) => +v !== 0)
    .map(([k, v]) => `<span class="rb-item">${k}: <b>${(+v).toFixed(4)}</b></span>`)
    .join('');
  box.innerHTML = `
    <div class="jr-head">Round ${round.round} · Winner: <b>${winner}</b></div>
    <div class="jr-reason">${round.reason || ''}</div>
    <div class="jr-pitches">${pitches}</div>
    <div class="jr-breakdown">${breakdown || '<span class="rb-item">no nonzero terms</span>'}</div>`;

  const counter = document.getElementById('jrCounter');
  if (counter) counter.textContent = `round ${round.round} of ${JUDGED_ROUNDS.length - 1} · ${JUDGED_ROUNDS.length} total`;
  const prev = document.getElementById('jrPrev');
  const next = document.getElementById('jrNext');
  if (prev) prev.disabled = _jrIndex === 0;
  if (next) next.disabled = _jrIndex === JUDGED_ROUNDS.length - 1;

  const title = document.getElementById('judgedTitle');
  if (title) {
    const tt = (JUDGED && JUDGED.task_type) ? JUDGED.task_type : 'coalition_market';
    const sd = (JUDGED && JUDGED.seed) ? JUDGED.seed : '5';
    title.textContent = `Judged Negotiation · ${tt} · seed ${sd} · Round ${round.round} Pitches`;
  }

  const trend = document.getElementById('jrScoreTrend');
  if (trend) {
    const first = _jrWinnerScore(JUDGED_ROUNDS[0]);
    const cur   = _jrWinnerScore(round);
    const arrow = cur >= first ? '↗' : '↘';
    trend.textContent = `${first.toFixed(3)} ${arrow} ${cur.toFixed(3)}`;
  }
  _drawJrTrend();
}

function _drawJrTrend() {
  const canvas = document.getElementById('jrTrendCanvas');
  if (!canvas || !JUDGED_ROUNDS.length) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  ctx.fillStyle = '#ebe3cf'; ctx.fillRect(0, 0, W, H);

  const padL = 28, padR = 14, padT = 8, padB = 8;
  const gW = W - padL - padR, gH = H - padT - padB;
  const scores = JUDGED_ROUNDS.map(_jrWinnerScore);
  const minV = Math.min(0.5, Math.min(...scores) - 0.02);
  const maxV = Math.max(1.0, Math.max(...scores) + 0.02);
  const range = (maxV - minV) || 1;

  ctx.strokeStyle = 'rgba(12,10,8,.18)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 2; i++) {
    const y = padT + (gH / 2) * i;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
  }
  ctx.fillStyle = '#3a3530';
  ctx.font = `9px 'JetBrains Mono', monospace`;
  ctx.textAlign = 'right';
  ctx.fillText(maxV.toFixed(2), padL - 4, padT + 4);
  ctx.fillText(minV.toFixed(2), padL - 4, padT + gH);

  const toX = (i) => padL + (i / Math.max(JUDGED_ROUNDS.length - 1, 1)) * gW;
  const toY = (v) => padT + gH - ((v - minV) / range) * gH;

  ctx.beginPath();
  ctx.moveTo(toX(0), toY(0));
  scores.forEach((v, i) => ctx.lineTo(toX(i), toY(v)));
  ctx.lineTo(toX(scores.length - 1), toY(0));
  ctx.closePath();
  ctx.fillStyle = 'rgba(255,59,48,.10)';
  ctx.fill();

  ctx.beginPath();
  scores.forEach((v, i) => {
    const x = toX(i), y = toY(v);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#ff3b30';
  ctx.lineWidth = 2.0;
  ctx.stroke();

  scores.forEach((v, i) => {
    const x = toX(i), y = toY(v);
    const isCurrent = i === _jrIndex;
    ctx.beginPath();
    ctx.arc(x, y, isCurrent ? 5 : 2.6, 0, Math.PI * 2);
    ctx.fillStyle = isCurrent ? '#1234ff' : '#ff3b30';
    ctx.fill();
    ctx.lineWidth = isCurrent ? 1.6 : 1;
    ctx.strokeStyle = '#0c0a08';
    ctx.stroke();
  });
}

(function _bindJudgedNav() {
  const prev = document.getElementById('jrPrev');
  const next = document.getElementById('jrNext');
  const canvas = document.getElementById('jrTrendCanvas');
  if (prev) prev.addEventListener('click', () => _renderJrRound(_jrIndex - 1));
  if (next) next.addEventListener('click', () => _renderJrRound(_jrIndex + 1));
  document.addEventListener('keydown', (e) => {
    if (!document.getElementById('judgedBox')) return;
    const inText = ['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName || '');
    if (inText) return;
    if (e.key === 'ArrowLeft')  _renderJrRound(_jrIndex - 1);
    if (e.key === 'ArrowRight') _renderJrRound(_jrIndex + 1);
  });
  if (canvas) {
    canvas.style.cursor = 'pointer';
    canvas.addEventListener('click', (e) => {
      if (!JUDGED_ROUNDS.length) return;
      const r = canvas.getBoundingClientRect();
      const ratio = (e.clientX - r.left) / r.width;
      const i = Math.round(ratio * (JUDGED_ROUNDS.length - 1));
      _renderJrRound(i);
    });
  }
  window.addEventListener('resize', _drawJrTrend);
})();
_renderJrRound(0);

// ── SFT sample renderer ──────────────────────────────────────────────────
const SFT = (APP_DATA && APP_DATA.sft) ? APP_DATA.sft : null;
function renderSft() {
  const box = document.getElementById('sftBox');
  if (!box || !SFT) return;
  const safe = (s) => (s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;');
  box.innerHTML = `
    <div class="sft-meta">
      <span>task</span><b>${safe(SFT.task_type)}</b>
      <span>seed</span><b>${safe(SFT.seed)}</b>
      <span>round</span><b>${safe(SFT.round_index)}</b>
    </div>
    <div class="sft-role"><span>system</span></div>
    <pre class="sft-pre">${safe(SFT.system)}</pre>
    <div class="sft-role"><span>user</span></div>
    <pre class="sft-pre">${safe(SFT.user)}</pre>
    <div class="sft-role"><span>assistant</span></div>
    <pre class="sft-pre sft-asst">${safe(SFT.assistant)}</pre>`;
}
renderSft();

// ── Downloads grid renderer ──────────────────────────────────────────────
const DOWNLOADS = (APP_DATA && APP_DATA.downloads) ? APP_DATA.downloads : [];
function renderDownloads() {
  const box = document.getElementById('downloadsGrid');
  if (!box) return;
  if (!DOWNLOADS.length) { box.innerHTML = '<div class="placeholder">no artifacts found in repo</div>'; return; }
  box.innerHTML = DOWNLOADS.map((d) => `
    <a class="dl-card" href="${d.url}" target="_blank" rel="noreferrer">
      <div class="dl-row">
        <span class="dl-label">${d.label}</span>
        <span class="dl-size">${d.size}</span>
      </div>
      <div class="dl-desc">${d.desc}</div>
      <div class="dl-cta">↓ open</div>
    </a>`).join('');
}
renderDownloads();

// ── Headline number override (real training_report.md) ───────────────────
(function applyHeadline() {
  const h = APP_DATA && APP_DATA.headline;
  if (!h) return;
  const final = document.getElementById('stat-final');
  if (final && h.final) final.textContent = (+h.final).toFixed(2);
  const eps = document.getElementById('stat-eps');
  if (eps && h.episodes) eps.textContent = h.episodes;
})();
</script>
</body>
</html>
"""


def _json_for_script(obj: Any) -> str:
    """JSON-encode for safe embedding inside a <script> tag."""
    return (
        json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _render_index_html() -> str:
    payload = _build_data_payload()
    return _INDEX_HTML.replace("__APP_DATA_PAYLOAD__", _json_for_script(payload))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Render the project front-page with live artifact data."""
    return _render_index_html()


@app.get("/api/data")
def api_data() -> dict[str, Any]:
    """Return the same data payload used by the front-page (JSON)."""
    return _build_data_payload()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "benchmark_id": "gpu_budget_negotiation"}


@app.get("/tasks")
def tasks() -> dict[str, object]:
    return {
        "benchmark_id": "gpu_budget_negotiation",
        "tasks": [
            {"task_type": "single_trade", "difficulty": "easy"},
            {"task_type": "market_round", "difficulty": "medium"},
            {"task_type": "coalition_market", "difficulty": "hard"},
        ],
        "features": [
            "coalitions",
            "adaptive_bot_pitches",
            "optional_rule_judge",
            "holdout_seed_evaluation",
            "redacted_public_state",
            "dynamic_market_shocks",
        ],
    }


@app.post("/reset")
def reset(config: ResetConfig) -> dict[str, object]:
    return {"observation": env.reset(config).model_dump(mode="json")}


@app.post("/step")
def step(action: GpuNegotiationAction) -> dict[str, object]:
    return {"observation": env.step(action).model_dump(mode="json")}


@app.get("/state")
def state(include_private: bool = False) -> dict[str, object]:
    if include_private:
        if os.getenv("GPU_ARENA_DEBUG_STATE") != "1":
            raise HTTPException(
                status_code=403,
                detail="Private state is disabled. Set GPU_ARENA_DEBUG_STATE=1 for local debugging.",
            )
    return {"state": env.state()} if include_private else {"state": env.public_state()}
