# Hussain LiveTorch Architect (HLA)

Hussain LiveTorch Architect is an MVP for a simple but useful idea: turn deep-learning model code into a live architecture diagram while the model is being written.

The project combines:

- static parsing for fast draft diagrams while editing
- optional runtime verification for shape-aware graphs
- a stable JSON graph format for downstream rendering
- export paths for documentation, demos, and paper-style figures
- a VS Code side panel for live feedback while coding

This repository is public as an open prototype. The core idea is here, the implementation is partially complete, and contributions are welcome.

## Project Idea

The original concept is described in [`live_pytorch_architecture_tool_documentation.pdf`](./live_pytorch_architecture_tool_documentation.pdf): a developer tool that keeps a deep-learning architecture diagram synchronized with model source code while the user is coding.

The broader vision is:

- write model code and see the architecture update live beside the editor
- inspect the structure while designing, debugging, and iterating
- include useful details on the diagram such as layer names, parameters, tensor shapes, and source locations
- eventually support more than one modeling stack

Today, the implementation is clearly PyTorch-first. That is the current working scope of the repository. The larger product idea can later expand to other deep-learning libraries or even a framework-agnostic intermediate representation.

The intended workflow is:

1. edit a PyTorch model
2. see a draft architecture graph update immediately
3. run a verification pass with sample inputs
4. upgrade the graph from draft to verified with tensor shapes and execution order
5. export clean diagrams for debugging, documentation, or research papers

## Current Status

This is not just a blank concept repo. It already contains a working MVP for the PyTorch version of the idea:

- Python package with CLI commands for `analyze`, `watch`, and `demo`
- AST-based parser for `nn.Module` classes and common layer patterns
- runtime verification with forward hooks and `torch.fx` fallback logic
- graph merge pipeline that combines draft and verified views
- DOT and D2 exporters with SVG, PNG, and PDF rendering support
- local web demo for live preview
- VS Code extension for side-panel and beside-editor diagrams
- automated tests that currently pass

What is still incomplete or only partially implemented:

- the spec describes a richer editor UX centered on React Flow and ELK-driven interaction, but the current UI is lighter-weight
- advanced graph abstractions like collapsible repeated blocks are not implemented
- `torch.export` integration is referenced in the design doc but not implemented in the current Python runtime path
- the long-term idea of supporting frameworks beyond PyTorch is not implemented yet
- error surfacing, packaging, onboarding, and release polish are still prototype-level
- the overall repo still reflects experimentation, not a finished product

## Repository Layout

- `hussain_livetorch_architect/` - core Python analyzer, runtime verifier, graph IR, exporters, CLI, and live demo server
- `vscode_hla_sidepanel/` - VS Code extension that renders live diagrams while editing Python files
- `tests/` - unit tests for parser, merge logic, exporters, runtime verification, and demo helpers
- `examples/` - small sample model used for quick local checks
- `train.py` - editable PyTorch sample for live iteration
- `live_pytorch_architecture_tool_documentation.pdf` - original design/specification document

## Quick Start

```powershell
python -m pip install -e .[dev]
python -m hussain_livetorch_architect analyze --source examples/cnn_example.py --class-name TinyCNN --output-dir out
```

Run with runtime verification:

```powershell
python -m pip install -e .[runtime]
python -m hussain_livetorch_architect analyze --source examples/cnn_example.py --class-name TinyCNN --verify --input-shape 1,3,32,32 --output-dir out_verified
```

Run the local demo:

```powershell
python -m hussain_livetorch_architect demo --source train.py --class-name TrainNet --verify --input-shape 1,3,32,32 --output-dir out_live --open-browser
```

## Documentation

- Project analysis: [`docs/PROJECT_ANALYSIS.md`](./docs/PROJECT_ANALYSIS.md)
- Contribution guide: [`CONTRIBUTING.md`](./CONTRIBUTING.md)

## Why Contribute

This repo is a good base for anyone interested in:

- PyTorch developer tooling
- diagram generation from source code
- ML debugging and observability UX
- VS Code extension development
- paper-quality architecture figure export

High-value contribution areas include:

- better parsing coverage for more model patterns
- stronger runtime graph alignment between AST and execution
- richer diagram UX and interaction
- design work toward a framework-agnostic graph layer beyond PyTorch
- packaging and install experience
- better error reporting and example projects

## Development Notes

Local test status during repository analysis:

```powershell
python -m pytest
```

Current result: `10 passed`

## Contribution Policy

Issues and pull requests are welcome.

One practical note: the repository does not currently include an explicit open-source license file. That should be added by the maintainer before broader reuse expectations are set. Until then, contributions are still welcome for discussion, review, and improvement of the project itself.
