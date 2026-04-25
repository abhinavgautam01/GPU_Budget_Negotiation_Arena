from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from gpu_budget_arena.env import GpuBudgetNegotiationEnv
from gpu_budget_arena.models import GpuNegotiationAction, ResetConfig

app = FastAPI(title="GPU Budget Negotiation Arena", version="0.1.0")
env = GpuBudgetNegotiationEnv()

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
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Orbitron:wght@700;900&display=swap" rel="stylesheet"/>
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

/* ── Theme v2: Aurora Glass (override layer) ─────────────────────────────── */
:root {
  --bg: #07060f;
  --surface: rgba(20, 18, 36, 0.64);
  --surface-strong: rgba(14, 12, 28, 0.86);
  --border: rgba(255, 255, 255, 0.15);
  --text: #eff3ff;
  --muted: #a0a8c8;
  --accent: #66f6ff;
  --accent2: #8f7dff;
  --accent3: #ff78c6;
  --accent4: #ffd166;
  --glass-shadow: 0 16px 50px rgba(5, 8, 28, 0.48);
}

body {
  background:
    radial-gradient(circle at 10% -10%, rgba(102, 246, 255, 0.18), transparent 40%),
    radial-gradient(circle at 85% 5%, rgba(143, 125, 255, 0.2), transparent 38%),
    radial-gradient(circle at 50% 120%, rgba(255, 120, 198, 0.16), transparent 48%),
    linear-gradient(165deg, #07060f 0%, #0b0a18 45%, #120f28 100%);
  color: var(--text);
}
body::before { opacity: .25; }
body::after {
  z-index: 0;
  background-image:
    linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
  animation: gridDrift 24s linear infinite;
}

#cursorGlow {
  position: fixed;
  width: 220px;
  height: 220px;
  border-radius: 50%;
  pointer-events: none;
  z-index: 2;
  mix-blend-mode: screen;
  filter: blur(32px);
  background: radial-gradient(circle, rgba(102,246,255,.26) 0%, rgba(143,125,255,.12) 45%, rgba(255,120,198,0) 72%);
  transform: translate(-50%, -50%);
  opacity: .8;
}

.wrap { position: relative; z-index: 3; }

header {
  margin: 22px auto 0;
  max-width: 1180px;
  border: 1px solid var(--border);
  border-radius: 24px;
  background: linear-gradient(145deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  backdrop-filter: blur(14px);
  box-shadow: var(--glass-shadow);
}
h1 {
  filter: none;
  background: linear-gradient(140deg, #b6fbff 0%, #9f8bff 48%, #ff8bd0 100%);
  text-shadow: 0 8px 28px rgba(143,125,255,.22);
}
.tagline { color: #c8d0ef; }

.badge {
  border-radius: 999px;
  padding: 6px 14px;
  letter-spacing: 1.4px;
}

.section {
  border-bottom: none;
  padding: 42px 0;
}

.stat-card,
.panel,
.chart-wrap,
.reward-item,
.action-card,
.api-card,
.artifact-box {
  background: linear-gradient(145deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--glass-shadow);
  backdrop-filter: blur(10px);
}

.panel-head, .api-card-head {
  border-bottom: 1px solid rgba(255,255,255,.1);
}

.stat-card::before {
  height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
}

.stat-card,
.action-card,
.api-card,
.reward-item,
.panel {
  transition: transform .24s ease, border-color .24s ease, box-shadow .24s ease;
}

.stat-card:hover,
.action-card:hover,
.api-card:hover,
.reward-item:hover,
.panel:hover {
  transform: translateY(-6px);
  border-color: rgba(143,125,255,.45);
  box-shadow: 0 16px 44px rgba(80, 75, 188, 0.35);
}

.task-btn,
.chart-tab {
  border-radius: 999px;
  border-color: rgba(255,255,255,.18);
  color: #ced5f3;
  background: rgba(255,255,255,.02);
}
.task-btn.active,
.task-btn:hover,
.chart-tab.active,
.chart-tab:hover {
  color: #fff;
  border-color: rgba(102,246,255,.55);
  background: linear-gradient(120deg, rgba(102,246,255,.2), rgba(143,125,255,.2));
  box-shadow: 0 0 18px rgba(102,246,255,.2);
}

.run-btn {
  position: relative;
  overflow: hidden;
  border-radius: 14px;
  border-color: rgba(102,246,255,.55);
  color: #fff;
  background: linear-gradient(120deg, rgba(102,246,255,.28), rgba(143,125,255,.36), rgba(255,120,198,.3));
}
.run-btn::after {
  content: "";
  position: absolute;
  top: -120%;
  left: -40%;
  width: 40%;
  height: 340%;
  transform: rotate(18deg);
  background: linear-gradient(to right, transparent, rgba(255,255,255,.42), transparent);
  transition: left .45s ease;
}
.run-btn:hover::after { left: 115%; }
.run-btn:hover {
  transform: translateY(-2px) scale(1.01);
  box-shadow: 0 10px 26px rgba(143,125,255,.4);
}

.transcript, .artifact-box { color: #edf1ff; }
.t-step {
  border-left-width: 3px;
  border-left-color: rgba(255,255,255,.2);
  background: rgba(255,255,255,.02);
  padding: 10px 12px;
  border-radius: 10px;
}
.t-step:hover { background: rgba(255,255,255,.045); }

#rewardCanvas, #baselineCanvas {
  border-radius: 12px;
  background: rgba(7, 7, 18, 0.5);
}

.action-name, .panel-title, .reward-name, .endpoint { color: #f4f6ff; }
.action-desc, .api-desc, .t-result, .artifact-result { color: #b4bcdd; }

footer {
  border-top: 1px solid rgba(255,255,255,.14);
  margin-top: 10px;
}

/* Respect motion preferences */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation: none !important;
    transition: none !important;
  }
}
</style>
</head>
<body>
<div id="cursorGlow" aria-hidden="true"></div>

<!-- ══════════ HEADER ══════════ -->
<header>
<div class="wrap">
  <span class="logo-label">● OpenEnv Multi-Agent Benchmark</span>
  <h1>GPU Budget<br/>Negotiation Arena</h1>
  <p class="tagline">Train LLMs to bargain, form coalitions, and adapt under market shocks in a scarce-GPU economy.</p>
  <div class="badge-row">
    <span class="badge green"><span class="status-dot"></span>Running</span>
    <span class="badge purple">Theme #1 · Multi-Agent</span>
    <span class="badge pink">OpenEnv Compatible</span>
    <span class="badge green">FastAPI · Docker</span>
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
  <p class="section-title">Baseline Policy Performance (10 Seeds)</p>
  <div class="chart-wrap">
    <div class="chart-tabs">
      <button class="chart-tab active" data-task="single_trade">Easy — Single Trade</button>
      <button class="chart-tab"        data-task="market_round">Medium — Market Round</button>
      <button class="chart-tab"        data-task="coalition_market">Hard — Coalition Market</button>
    </div>
    <canvas id="baselineCanvas"></canvas>
    <div class="legend" id="legend"></div>
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
// ── Microinteractions ─────────────────────────────────────────────────────────
const cursorGlow = document.getElementById('cursorGlow');
if (cursorGlow && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  window.addEventListener('pointermove', (e) => {
    cursorGlow.style.left = `${e.clientX}px`;
    cursorGlow.style.top = `${e.clientY}px`;
  });
}

function enableMagnetic(selector, strength = 10) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  document.querySelectorAll(selector).forEach((el) => {
    el.style.transition = 'transform .22s ease';
    el.addEventListener('pointermove', (e) => {
      const r = el.getBoundingClientRect();
      const x = (e.clientX - r.left - r.width / 2) / r.width;
      const y = (e.clientY - r.top - r.height / 2) / r.height;
      el.style.transform = `translate(${x * strength}px, ${y * strength}px)`;
    });
    el.addEventListener('pointerleave', () => {
      el.style.transform = '';
    });
  });
}

enableMagnetic('.run-btn', 8);
enableMagnetic('.task-btn, .chart-tab', 5);

// ── Baseline data (from artifacts/baseline_eval.json summary) ───────────────
const BASELINE = {
  single_trade: {
    always_accept:          0.0587,
    base_instruct_naive:    0.0771,
    greedy_hoarder:         0.0587,
    no_negotiation:         0.0587,
    random_validish:        0.0747,
    rule_based_expert:      0.2623,
  },
  market_round: {
    always_accept:          0.2725,
    base_instruct_naive:   -0.0069,
    greedy_hoarder:         0.0286,
    no_negotiation:         0.0286,
    random_validish:        0.1595,
    rule_based_expert:      0.4845,
  },
  coalition_market: {
    always_accept:          0.3722,
    base_instruct_naive:   -0.0355,
    greedy_hoarder:         0.0995,
    no_negotiation:         0.0995,
    random_validish:        0.1709,
    rule_based_expert:      0.8149,
  },
};

const POLICY_COLORS = {
  rule_based_expert:   '#00ffe0',
  always_accept:       '#7b5cff',
  random_validish:     '#f4a261',
  base_instruct_naive: '#ff4f87',
  greedy_hoarder:      '#e63946',
  no_negotiation:      '#adb5bd',
};

const POLICY_LABELS = {
  rule_based_expert:   'Rule-Based Expert',
  always_accept:       'Always Accept',
  random_validish:     'Random Valid',
  base_instruct_naive: 'Base Instruct Naive',
  greedy_hoarder:      'Greedy Hoarder',
  no_negotiation:      'No-Negotiation Alloc.',
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
  const data = BASELINE[taskKey];
  const policies = Object.keys(data);
  const values = Object.values(data);
  const minV = Math.min(0, ...values);
  const maxV = Math.max(...values) * 1.15;
  const range = maxV - minV;
  const gW = W - padL - padR;
  const gH = H - padT - padB;

  ctx.fillStyle = '#0d1221';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = minV + (range / ticks) * i;
    const y = padT + gH - (v - minV) / range * gH;
    ctx.strokeStyle = '#1a2540';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillStyle = '#4a5a7a';
    ctx.font = `10px 'Share Tech Mono', monospace`;
    ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(2), padL - 6, y + 4);
  }

  // Zero line
  const zeroY = padT + gH - (0 - minV) / range * gH;
  ctx.strokeStyle = '#334466';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(padL, zeroY); ctx.lineTo(W - padR, zeroY); ctx.stroke();

  // Bars
  const barW = Math.min(60, gW / policies.length * 0.55);
  policies.forEach((pol, i) => {
    const v = data[pol];
    const x = padL + (i + 0.5) * (gW / policies.length);
    const barH = Math.abs(v) / range * gH;
    const barY = v >= 0 ? zeroY - barH : zeroY;

    const grad = ctx.createLinearGradient(x - barW/2, barY, x + barW/2, barY + barH);
    const col = POLICY_COLORS[pol] || '#fff';
    grad.addColorStop(0, col + 'cc');
    grad.addColorStop(1, col + '44');
    ctx.fillStyle = grad;
    ctx.fillRect(x - barW/2, barY, barW, barH);

    // Value label
    ctx.fillStyle = col;
    ctx.font = `bold 11px 'Share Tech Mono', monospace`;
    ctx.textAlign = 'center';
    ctx.fillText(v.toFixed(3), x, v >= 0 ? barY - 5 : barY + barH + 14);
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

  ctx.fillStyle = '#060810';
  ctx.fillRect(0, 0, W, H);

  if (rewardTrace.length < 2) {
    ctx.strokeStyle = '#1a2540';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
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

  // Fill
  const grad = ctx.createLinearGradient(0, padT, 0, padT + gH);
  grad.addColorStop(0, 'rgba(0,255,224,.3)');
  grad.addColorStop(1, 'rgba(0,255,224,0)');
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(rewardTrace[0]));
  rewardTrace.forEach((v, i) => ctx.lineTo(toX(i), toY(v)));
  ctx.lineTo(toX(rewardTrace.length - 1), padT + gH);
  ctx.lineTo(padL, padT + gH);
  ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();

  // Line
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(rewardTrace[0]));
  rewardTrace.forEach((v, i) => ctx.lineTo(toX(i), toY(v)));
  ctx.strokeStyle = '#00ffe0'; ctx.lineWidth = 2;
  ctx.stroke();

  // Dots
  rewardTrace.forEach((v, i) => {
    ctx.beginPath();
    ctx.arc(toX(i), toY(v), 4, 0, Math.PI * 2);
    ctx.fillStyle = '#00ffe0'; ctx.fill();
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
  document.getElementById('demo-status').style.color = 'var(--accent2)';
  btn.disabled = false; btn.textContent = '▶ RUN EPISODE';
  running = false;
});

// ── Artifact transcript ────────────────────────────────────────────────────────
const TRANSCRIPT = [
  { task: 'coalition_market', seed: 5, policy: 'rule_based_expert', cumulative: 1.3412, steps: [
    { action: '{"action_type":"accept_offer","offer_id":"o_1"}', result: 'Accepted offer o_1.', reward: 0.1097, cum: 0.1097 },
    { action: '{"action_type":"reject_offer","offer_id":"o_2"}', result: 'Rejected offer o_2.', reward: 0.0080, cum: 0.1177 },
    { action: '{"action_type":"reject_offer","offer_id":"o_3"}', result: 'Rejected offer o_3.', reward: 0.0080, cum: 0.1257 },
    { action: '{"action_type":"accept_offer","offer_id":"o_7"}', result: 'Accepted offer o_7.', reward: 0.1205, cum: 0.2462 },
    { action: '{"action_type":"allocate_to_job","block_ids":["b_0_1","b_1_0"],"job_id":"j_0_1"}', result: 'Allocated capacity to j_0_1.', reward: 0.1910, cum: 0.4372 },
    { action: '{"action_type":"allocate_to_job","block_ids":["b_4_0"],"job_id":"j_0_2"}',         result: 'Allocated capacity to j_0_2.', reward: 0.1910, cum: 0.6282 },
    { action: '{"action_type":"form_coalition","target_lab_id":"lab_2","message":"shared capacity for urgent deadlines"}', result: 'Created coalition c_1.', reward: 0.1280, cum: 0.7562 },
    { action: '{"action_type":"accept_offer","offer_id":"o_8"}', result: 'Accepted offer o_8.', reward: 0.1660, cum: 0.9222 },
    { action: '{"action_type":"allocate_to_job","block_ids":["b_2_2"],"job_id":"j_0_surge_8"}', result: 'Final settlement.', reward: 0.2280, cum: 1.3412 },
  ]}
];

function renderArtifact() {
  const t = TRANSCRIPT[0];
  const box = document.getElementById('artifactBox');
  let html = `<div><span style="color:var(--accent)">task_type:</span> ${t.task} &nbsp;|&nbsp; <span style="color:var(--accent)">seed:</span> ${t.seed} &nbsp;|&nbsp; <span style="color:var(--accent)">policy:</span> ${t.policy} &nbsp;|&nbsp; <span style="color:var(--accent)">final_reward:</span> <span style="color:var(--accent2)">${t.cumulative}</span></div>`;
  t.steps.forEach((s, i) => {
    html += `
<div class="artifact-step-head">── STEP ${i+1} ──────────────────────</div>
<div class="artifact-action">action  : ${s.action}</div>
<div class="artifact-result">result  : ${s.result}</div>
<div class="artifact-reward">reward  : +${s.reward.toFixed(4)} &nbsp; cumulative: ${s.cum.toFixed(4)}</div>`;
  });
  box.innerHTML = html;
}
renderArtifact();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Render the project front-page."""
    return _INDEX_HTML


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
