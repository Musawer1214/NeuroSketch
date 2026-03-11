# NeuroSketch

🧠 NeuroSketch is a live architecture companion for deep-learning model development.

Write model code, keep coding, and see the architecture update beside your editor in real time. The long-term vision is broader than any one framework, but the current MVP is PyTorch-first and already includes a working parser, runtime verification flow, diagram export pipeline, and VS Code side panel.

## ✨ What NeuroSketch Tries To Do

NeuroSketch is built around one simple workflow:

1. write a model
2. see the architecture update while you code
3. inspect the structure, layer details, and flow without leaving the editor
4. verify the graph with runtime execution when you want more accuracy
5. export diagrams for debugging, documentation, or research papers

## 🚀 What Works Today

This repository already contains a real MVP, not just an idea:

- PyTorch `nn.Module` parsing from source code
- draft graph generation from AST analysis
- optional runtime verification with hooks and `torch.fx`
- graph merge logic for draft + verified views
- JSON, DOT, and D2 graph export
- SVG, PNG, and PDF rendering support when external tools are available
- local browser demo for live preview
- VS Code side panel / beside-editor extension
- automated tests that currently pass

## 🧩 Why It’s Useful

NeuroSketch is meant to help with:

- understanding how a model is actually structured
- spotting design issues while still coding
- communicating architecture ideas to teammates
- generating cleaner visuals for notes, reports, and papers
- reducing the need to manually redraw model diagrams

## 🛠 Current Scope

Right now, the implementation is focused on PyTorch.

That means:

- the product vision is broader
- the current engine is PyTorch-specific
- the next major evolution would be making the graph layer less tied to PyTorch assumptions

## 🚧 What Still Needs Work

This is still a prototype/MVP, so several parts are incomplete:

- richer interactive diagram UX
- broader support for real-world model patterns
- stronger alignment between static and runtime graphs
- more advanced abstractions like repeated block collapsing
- multi-framework support beyond PyTorch
- better packaging, onboarding, and release polish

## ⚡ Quick Start

Install the project in editable mode:

```powershell
python -m pip install -e .[dev]
```

Analyze a sample model:

```powershell
python -m neurosketch analyze --source examples/cnn_example.py --class-name TinyCNN --output-dir out
```

Run a verified analysis:

```powershell
python -m pip install -e .[runtime]
python -m neurosketch analyze --source examples/cnn_example.py --class-name TinyCNN --verify --input-shape 1,3,32,32 --output-dir out_verified
```

Run the live local demo:

```powershell
python -m neurosketch demo --source train.py --class-name TrainNet --verify --input-shape 1,3,32,32 --output-dir out_live --open-browser
```

## 🧪 Test Status

Run the test suite with:

```powershell
python -m pytest
```

Current local validation during this repo update:

- `10 passed`

## 📁 Project Structure

- `neurosketch/` - core analyzer, runtime verifier, graph IR, exporters, CLI, and live demo server
- `vscode_neurosketch_sidepanel/` - VS Code extension for live architecture previews
- `tests/` - parser, exporter, runtime, merge, and demo tests
- `examples/` - small example model for quick experiments
- `train.py` - editable sample model for live iteration
- `neurosketch_design_documentation.pdf` - original design/specification document

## 📚 Documentation

- Project analysis: [`docs/PROJECT_ANALYSIS.md`](./docs/PROJECT_ANALYSIS.md)
- Contributing guide: [`CONTRIBUTING.md`](./CONTRIBUTING.md)

## 📝 Note On The Design PDF

The bundled PDF is the original concept document for the project. It preserves the project’s earlier working title inside the document, but it still captures the main design direction behind NeuroSketch.

## 🤝 Contributions

Contributions are welcome.

Useful contribution areas include:

- parser coverage for more model patterns
- better verified graph reconstruction
- better VS Code UX
- framework-agnostic graph design
- export quality and presentation
- packaging and developer experience

## 📌 Important Repo Note

The repository still does not include an explicit open-source license file.

That should be added by the maintainer if the project is going to be opened up for broader reuse and outside contributions.
