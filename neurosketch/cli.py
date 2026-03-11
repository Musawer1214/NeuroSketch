from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from . import TOOL_NAME
from .exporters import export_graph
from .live_demo import LiveDemoRunner, normalize_demo_formats
from .merge import merge_graphs
from .runtime_verifier import RuntimeVerificationError, verify_runtime
from .static_parser import ParseError, parse_pytorch_source


def _parse_shape(shape: str) -> list[int]:
    parts = [p.strip() for p in shape.split(",") if p.strip()]
    if not parts:
        raise ValueError("Input shape must be comma-separated integers.")
    return [int(p) for p in parts]


def _parse_json_value(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    return json.loads(raw)


def run_analyze(args: argparse.Namespace) -> int:
    source = Path(args.source)
    quiet = bool(getattr(args, "quiet", False))
    if not source.exists():
        if not quiet:
            print(f"Error: source file not found: {source}")
        return 2

    try:
        draft_graph = parse_pytorch_source(source_path=source, class_name=args.class_name)
    except ParseError as exc:
        if not quiet:
            print(f"Parse error: {exc}")
        return 2

    final_graph = draft_graph
    verify_error: str | None = None

    if args.verify:
        try:
            init_args = _parse_json_value(args.init_args, [])
            init_kwargs = _parse_json_value(args.init_kwargs, {})
            if not isinstance(init_args, list):
                raise ValueError("--init-args must be a JSON array")
            if not isinstance(init_kwargs, dict):
                raise ValueError("--init-kwargs must be a JSON object")
            runtime_graph = verify_runtime(
                source_path=source,
                class_name=args.class_name or draft_graph.model_name,
                input_shape=_parse_shape(args.input_shape),
                init_args=init_args,
                init_kwargs=init_kwargs,
            )
            final_graph = merge_graphs(draft_graph, runtime_graph)
        except (RuntimeVerificationError, ValueError, json.JSONDecodeError) as exc:
            verify_error = str(exc)

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    outputs, effective_renderer = export_graph(
        final_graph,
        out_dir=args.output_dir,
        formats=formats,
        renderer=args.renderer,
        theme=args.theme,
        layout=args.layout,
        pad=args.pad,
    )
    requested_render = [f for f in formats if f in {"svg", "png", "pdf"}]
    missing_render = [f for f in requested_render if f not in outputs]

    if not quiet:
        print(f"{TOOL_NAME}")
        print(f"Model: {final_graph.model_name}")
        print(f"Nodes: {len(final_graph.nodes)} | Edges: {len(final_graph.edges)}")
        print(f"Mode: {final_graph.meta.get('mode', 'draft')}")
        print(f"Renderer: {effective_renderer} | Theme: {args.theme} | Layout: {args.layout}")
        if verify_error:
            print(f"Verification warning: {verify_error}")
    if missing_render and not quiet:
        joined = ", ".join(missing_render)
        if effective_renderer == "d2":
            print(
                "Export warning: could not render "
                f"{joined}. Install D2 and ensure `d2` is on PATH."
            )
        else:
            print(
                "Export warning: could not render "
                f"{joined}. Ensure Graphviz is installed and `dot` is on PATH."
            )
    if not quiet:
        print("Outputs:")
        for key, path in sorted(outputs.items()):
            print(f"  - {key}: {path}")

    return 0


def run_watch(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.exists():
        print(f"Error: source file not found: {source}")
        return 2

    print(f"Watching {source} every {args.interval:.2f}s (Ctrl+C to stop).")
    last_hash = None
    while True:
        try:
            current_hash = source.read_bytes()
            if current_hash != last_hash:
                last_hash = current_hash
                print("- change detected: rebuilding graph")
                analyze_args = argparse.Namespace(
                    source=args.source,
                    class_name=args.class_name,
                    output_dir=args.output_dir,
                    verify=args.verify,
                    input_shape=args.input_shape,
                    init_args=args.init_args,
                    init_kwargs=args.init_kwargs,
                    formats=args.formats,
                    renderer=args.renderer,
                    theme=args.theme,
                    layout=args.layout,
                    pad=args.pad,
                    quiet=False,
                )
                run_analyze(analyze_args)
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("Stopped watcher.")
            return 0


def run_demo(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if not source.exists():
        print(f"Error: source file not found: {source}")
        return 2

    demo_formats = normalize_demo_formats(args.formats)
    analyze_args = argparse.Namespace(
        source=args.source,
        class_name=args.class_name,
        output_dir=args.output_dir,
        verify=args.verify,
        input_shape=args.input_shape,
        init_args=args.init_args,
        init_kwargs=args.init_kwargs,
        formats=demo_formats,
        renderer=args.renderer,
        theme=args.theme,
        layout=args.layout,
        pad=args.pad,
        quiet=True,
    )

    def _analyze_once() -> int:
        return run_analyze(analyze_args)

    runner = LiveDemoRunner(
        host=args.host,
        port=args.port,
        source_path=source,
        output_dir=Path(args.output_dir),
        interval=args.interval,
        analyze_once=_analyze_once,
        renderer=args.renderer,
        theme=args.theme,
        layout=args.layout,
        open_browser=args.open_browser,
    )
    return runner.serve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurosketch", description=TOOL_NAME)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", required=True, help="Path to model source .py file")
    common.add_argument("--class-name", help="nn.Module class name (defaults to first found)")
    common.add_argument("--output-dir", default="out", help="Output directory")
    common.add_argument("--formats", default="json,dot", help="Comma list: json,dot,d2,svg,png,pdf")
    common.add_argument(
        "--renderer",
        default="auto",
        choices=["auto", "d2", "graphviz"],
        help="Renderer backend for image export",
    )
    common.add_argument(
        "--theme",
        default="journal-light",
        choices=["journal-light", "journal-gray", "journal-minimal"],
        help="Publication theme preset",
    )
    common.add_argument(
        "--layout",
        default="elk",
        help="Layout engine (for D2: elk|dagre; ignored by Graphviz)",
    )
    common.add_argument(
        "--pad",
        type=int,
        default=40,
        help="Canvas padding in pixels",
    )
    common.add_argument("--verify", action="store_true", help="Enable runtime verification")
    common.add_argument("--input-shape", default="1,3,224,224", help="Input tensor shape")
    common.add_argument("--init-args", default="[]", help="JSON array for model constructor args")
    common.add_argument(
        "--init-kwargs",
        default="{}",
        help="JSON object for model constructor kwargs",
    )

    analyze = sub.add_parser("analyze", parents=[common], help="Analyze one model file")
    analyze.set_defaults(func=run_analyze)

    watch = sub.add_parser("watch", parents=[common], help="Watch file and rebuild on changes")
    watch.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    watch.set_defaults(func=run_watch)

    demo = sub.add_parser("demo", parents=[common], help="Run live local web demo")
    demo.add_argument("--interval", type=float, default=0.8, help="Polling interval in seconds")
    demo.add_argument("--host", default="127.0.0.1", help="Web server host")
    demo.add_argument("--port", type=int, default=8765, help="Web server port")
    demo.add_argument("--open-browser", action="store_true", help="Open browser on start")
    demo.set_defaults(func=run_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
