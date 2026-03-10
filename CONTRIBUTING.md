# Contributing

Contributions are welcome.

This repository is currently best treated as an open MVP: there is a working foundation, but plenty of room to improve parser coverage, runtime verification, export quality, extension UX, and project polish.

## Good First Areas

- improve README and examples
- add parser coverage for more PyTorch patterns
- strengthen runtime verification edge cases
- improve VS Code extension behavior and messaging
- add more tests around graph merging and exports
- explore paths toward a less PyTorch-specific graph core
- improve packaging and developer setup

## Local Setup

```powershell
python -m pip install -e .[dev]
```

If you want runtime verification:

```powershell
python -m pip install -e .[runtime]
```

If you want Graphviz-backed exports:

```powershell
python -m pip install -e .[export]
```

## Running Tests

```powershell
python -m pytest
```

## Useful Local Commands

Analyze an example model:

```powershell
python -m hussain_livetorch_architect analyze --source examples/cnn_example.py --class-name TinyCNN --output-dir out
```

Run a verified analysis:

```powershell
python -m hussain_livetorch_architect analyze --source examples/cnn_example.py --class-name TinyCNN --verify --input-shape 1,3,32,32 --output-dir out_verified
```

Run the live demo:

```powershell
python -m hussain_livetorch_architect demo --source train.py --class-name TrainNet --verify --input-shape 1,3,32,32 --output-dir out_live
```

## Contribution Expectations

- keep changes focused
- add or update tests when behavior changes
- preserve the split between draft parsing, runtime verification, merge logic, and export logic
- document new commands, flags, or workflows in the README when relevant

## Notes

The repository does not currently include an explicit license file. That should be addressed by the maintainer for long-term open-source reuse. In the meantime, pull requests and issue discussions are still welcome.
