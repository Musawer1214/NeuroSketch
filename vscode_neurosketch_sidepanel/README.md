# NeuroSketch Side Panel (VS Code)

This extension adds a live architecture diagram side panel in VS Code for NeuroSketch.

## What it does

- updates the diagram on save
- supports optional debounced realtime updates while typing
- renders live architecture diagrams beside your code
- shows model stats like node count, layer count, estimated params, and update reason

## Run in Extension Development Host

Fastest way from the project root:

```powershell
.\open_neurosketch_sidepanel.ps1
```

Then in the opened VS Code window run:

- `NeuroSketch: Open Side Panel`

Install it into normal VS Code:

```powershell
.\install_neurosketch_extension.ps1
```

Then run:

- `NeuroSketch: Open Diagram Beside`

Manual setup:

1. Open this folder in VS Code:
   - `C:\Users\musaw\OneDrive\Desktop\DL_Architecture\vscode_neurosketch_sidepanel`
2. Install dependencies and compile:
   - `npm install`
   - `npm run compile`
3. Launch Extension Development Host:
   - `npm run devhost`
4. In the new window, open the main project folder:
   - `C:\Users\musaw\OneDrive\Desktop\DL_Architecture`
5. Open the `NeuroSketch` activity bar icon and start editing a `.py` model file.

## Commands

- `NeuroSketch: Open Side Panel`
- `NeuroSketch: Open Diagram Beside`
- `NeuroSketch: Refresh Diagram`
- `NeuroSketch: Toggle Realtime Updates`

## Settings

- `neurosketch.pythonPath`
- `neurosketch.analyzerCommand`
- `neurosketch.autoUpdateOnType`
- `neurosketch.updateDebounceMs`
- `neurosketch.verifyOnSave`
- `neurosketch.renderer`
- `neurosketch.theme`
- `neurosketch.layout`
- `neurosketch.inputShape`
- `neurosketch.pad`
- `neurosketch.className`
