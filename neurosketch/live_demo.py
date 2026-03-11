from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from typing import Callable
import webbrowser


def normalize_demo_formats(formats_raw: str) -> str:
    wanted = [f.strip().lower() for f in formats_raw.split(",") if f.strip()]
    seen = set(wanted)
    if "svg" not in seen:
        wanted.append("svg")
        seen.add("svg")
    if "json" not in seen:
        wanted.append("json")
    return ",".join(wanted)


@dataclass
class LiveDemoStatus:
    source: str
    output_dir: str
    running: bool = True
    initialized: bool = False
    last_build_ok: bool = False
    last_error: str | None = None
    last_build_utc: str | None = None
    build_count: int = 0
    model: str | None = None
    nodes: int = 0
    edges: int = 0
    mode: str | None = None
    renderer: str | None = None
    theme: str | None = None
    layout: str | None = None

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "output_dir": self.output_dir,
            "running": self.running,
            "initialized": self.initialized,
            "last_build_ok": self.last_build_ok,
            "last_error": self.last_error,
            "last_build_utc": self.last_build_utc,
            "build_count": self.build_count,
            "model": self.model,
            "nodes": self.nodes,
            "edges": self.edges,
            "mode": self.mode,
            "renderer": self.renderer,
            "theme": self.theme,
            "layout": self.layout,
        }


@dataclass
class LiveDemoRunner:
    host: str
    port: int
    source_path: Path
    output_dir: Path
    interval: float
    analyze_once: Callable[[], int]
    renderer: str
    theme: str
    layout: str
    open_browser: bool = False
    _server: ThreadingHTTPServer | None = field(default=None, init=False)
    _watcher: threading.Thread | None = field(default=None, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    status: LiveDemoStatus = field(init=False)

    def __post_init__(self) -> None:
        self.status = LiveDemoStatus(
            source=str(self.source_path),
            output_dir=str(self.output_dir),
            renderer=self.renderer,
            theme=self.theme,
            layout=self.layout,
        )

    def _set_error(self, message: str) -> None:
        with self._lock:
            self.status.last_build_ok = False
            self.status.last_error = message
            self.status.initialized = True

    def _set_success(self, graph_json_path: Path) -> None:
        payload = json.loads(graph_json_path.read_text(encoding="utf-8"))
        with self._lock:
            self.status.last_build_ok = True
            self.status.last_error = None
            self.status.initialized = True
            self.status.last_build_utc = datetime.now(tz=timezone.utc).isoformat()
            self.status.build_count += 1
            self.status.model = payload.get("model", {}).get("name")
            self.status.nodes = len(payload.get("nodes", []))
            self.status.edges = len(payload.get("edges", []))
            self.status.mode = payload.get("meta", {}).get("mode")

    def _watch_loop(self) -> None:
        last_hash: bytes | None = None
        while not self._stop_event.is_set():
            try:
                current_hash = self.source_path.read_bytes()
            except Exception as exc:
                self._set_error(f"Could not read source file: {exc}")
                time.sleep(self.interval)
                continue

            changed = last_hash != current_hash
            if changed:
                last_hash = current_hash
                code = self.analyze_once()
                graph_json = self.output_dir / "graph.json"
                if code != 0:
                    self._set_error("Analysis failed. Check terminal logs for details.")
                elif not graph_json.exists():
                    self._set_error("Analysis succeeded but graph.json was not generated.")
                else:
                    try:
                        self._set_success(graph_json)
                    except Exception as exc:
                        self._set_error(f"Failed to parse graph.json: {exc}")
            time.sleep(self.interval)

    def _build_handler(self):
        runner = self

        class DemoHandler(BaseHTTPRequestHandler):
            def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _bytes(self, data: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)

            def _serve_file(self, filename: str, content_type: str) -> None:
                path = runner.output_dir / filename
                if not path.exists():
                    self._bytes(b"Not Found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                    return
                self._bytes(path.read_bytes(), content_type)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/" or self.path.startswith("/?"):
                    self._bytes(_demo_html().encode("utf-8"), "text/html; charset=utf-8")
                    return
                if self.path.startswith("/api/status"):
                    with runner._lock:
                        self._json(runner.status.as_dict())
                    return
                if self.path.startswith("/graph.svg"):
                    self._serve_file("graph.svg", "image/svg+xml")
                    return
                if self.path.startswith("/graph.png"):
                    self._serve_file("graph.png", "image/png")
                    return
                if self.path.startswith("/graph.pdf"):
                    self._serve_file("graph.pdf", "application/pdf")
                    return
                if self.path.startswith("/graph.json"):
                    self._serve_file("graph.json", "application/json")
                    return
                if self.path.startswith("/graph.d2"):
                    self._serve_file("graph.d2", "text/plain; charset=utf-8")
                    return
                self._bytes(b"Not Found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

            def log_message(self, _format: str, *_args) -> None:
                # Keep terminal clean; build logs already show important info.
                return

        return DemoHandler

    def serve(self) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._watcher = threading.Thread(target=self._watch_loop, daemon=True)
        self._watcher.start()

        handler = self._build_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        url = f"http://{self.host}:{self.port}/"
        print(f"Live demo running at {url}")
        print("Open the URL, edit your model file, and save to see updates.")
        print("Press Ctrl+C to stop.")
        if self.open_browser:
            webbrowser.open(url)

        try:
            self._server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
        return 0

    def shutdown(self) -> None:
        self._stop_event.set()
        with self._lock:
            self.status.running = False
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._watcher is not None and self._watcher.is_alive():
            self._watcher.join(timeout=2.0)


def _demo_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>NeuroSketch Live Demo</title>
  <style>
    :root {
      --bg: #f8fafc;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --accent: #0f766e;
      --warn: #b91c1c;
      --line: #e2e8f0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 400px at 20% 0%, #dcfce7 0%, transparent 60%),
        radial-gradient(900px 320px at 85% 0%, #dbeafe 0%, transparent 55%),
        var(--bg);
    }
    .shell {
      max-width: 1280px;
      margin: 24px auto;
      padding: 0 18px;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 32px rgba(2, 6, 23, 0.08);
    }
    .meta {
      padding: 16px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 20px;
      letter-spacing: 0.02em;
    }
    .subtitle {
      color: var(--muted);
      font-size: 13px;
      margin: 0 0 14px;
    }
    .k {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .v {
      font-size: 14px;
      margin-bottom: 10px;
      word-break: break-all;
    }
    .chip {
      display: inline-block;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 600;
      border: 1px solid var(--line);
      margin-right: 6px;
      margin-bottom: 6px;
      background: #f1f5f9;
    }
    .ok { color: #166534; background: #dcfce7; border-color: #bbf7d0; }
    .bad { color: #991b1b; background: #fee2e2; border-color: #fecaca; }
    .canvas {
      min-height: 80vh;
      overflow: auto;
      padding: 12px;
    }
    #diagram {
      width: 100%;
      min-height: 75vh;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
    }
    .error {
      margin-top: 10px;
      color: var(--warn);
      font-size: 13px;
      white-space: pre-wrap;
    }
    @media (max-width: 960px) {
      .shell { grid-template-columns: 1fr; }
      .canvas { min-height: 65vh; }
      #diagram { min-height: 60vh; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="panel meta">
      <h1>NeuroSketch</h1>
      <p class="subtitle">Live architecture diagram dashboard</p>
      <div id="chips"></div>
      <div class="k">Source</div><div id="source" class="v">-</div>
      <div class="k">Output Dir</div><div id="output" class="v">-</div>
      <div class="k">Model</div><div id="model" class="v">-</div>
      <div class="k">Graph</div><div id="graph" class="v">-</div>
      <div class="k">Mode</div><div id="mode" class="v">-</div>
      <div class="k">Renderer</div><div id="renderer" class="v">-</div>
      <div class="k">Last Build (UTC)</div><div id="time" class="v">-</div>
      <div id="error" class="error"></div>
    </section>
    <section class="panel canvas">
      <img id="diagram" alt="NeuroSketch diagram preview" />
    </section>
  </div>
  <script>
    let lastBuildCount = -1;
    async function refreshStatus() {
      try {
        const res = await fetch('/api/status', { cache: 'no-store' });
        if (!res.ok) return;
        const s = await res.json();
        document.getElementById('source').textContent = s.source || '-';
        document.getElementById('output').textContent = s.output_dir || '-';
        document.getElementById('model').textContent = s.model || '-';
        document.getElementById('graph').textContent = `${s.nodes ?? 0} nodes, ${s.edges ?? 0} edges`;
        document.getElementById('mode').textContent = s.mode || '-';
        document.getElementById('renderer').textContent = `${s.renderer || '-'} | ${s.theme || '-'} | ${s.layout || '-'}`;
        document.getElementById('time').textContent = s.last_build_utc || '-';
        document.getElementById('error').textContent = s.last_error || '';

        const chips = [];
        chips.push(`<span class="chip ${s.running ? 'ok' : 'bad'}">${s.running ? 'running' : 'stopped'}</span>`);
        chips.push(`<span class="chip ${s.last_build_ok ? 'ok' : 'bad'}">${s.last_build_ok ? 'build ok' : 'build failed'}</span>`);
        chips.push(`<span class="chip">build #${s.build_count || 0}</span>`);
        document.getElementById('chips').innerHTML = chips.join('');

        if (s.build_count !== lastBuildCount) {
          lastBuildCount = s.build_count;
          const img = document.getElementById('diagram');
          img.src = '/graph.svg?t=' + Date.now();
          img.onerror = () => { img.src = '/graph.png?t=' + Date.now(); };
        }
      } catch (_) {
      }
    }
    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""
