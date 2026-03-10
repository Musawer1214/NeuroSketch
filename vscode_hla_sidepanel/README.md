# Hussain LiveTorch Architect Side Panel (VS Code)

This extension adds a live architecture diagram side panel in VS Code.

## What it does

- Updates diagram on `Ctrl+S` (save).
- Optional debounced updates while typing.
- Renders modern journal-style diagrams via HLA (`d2` by default in auto mode).
- Shows status, model name, node/edge counts, and update reason.

## Run in Extension Development Host

Fastest way (from project root):

```powershell
.\open_hla_sidepanel.ps1
```

Then in the opened window run command:
- `HLA: Open Side Panel`

No extra development window (install once, then use normal VS Code):

```powershell
cd C:\Users\musaw\OneDrive\Desktop\DL_Architecture
.\install_hla_extension.ps1
```

After install, in your normal VS Code window run:
- `HLA: Open Diagram Beside`

Manual way:

1. Open this folder in VS Code:
   - `C:\Users\musaw\OneDrive\Desktop\DL_Architecture\vscode_hla_sidepanel`
2. Install deps and compile:
   - `npm install`
   - `npm run compile`
3. Launch Extension Development Host (no `F5` needed):
   - `npm run devhost`
4. In the new window, open your project folder:
   - `C:\Users\musaw\OneDrive\Desktop\DL_Architecture`
5. Open the `HLA` activity bar icon, then open a `.py` model file.
6. Start coding; save to update immediately, and type for realtime updates.

## Commands

- `HLA: Open Side Panel`
- `HLA: Open Diagram Beside`
- `HLA: Refresh Diagram`
- `HLA: Toggle Realtime Updates`

Default shortcuts (inside Python editor):

- Open side panel: `Ctrl+Alt+H`
- Open diagram beside editor: `Ctrl+Alt+D`
- Refresh diagram: `Ctrl+Alt+R`
- Toggle realtime updates: `Ctrl+Alt+T`

## Settings

- `hla.pythonPath` (default: `python`)
- `hla.autoUpdateOnType` (default: `true`)
- `hla.updateDebounceMs` (default: `600`)
- `hla.verifyOnSave` (default: `false`)
- `hla.renderer` (`auto`, `d2`, `graphviz`)
- `hla.theme` (`journal-light`, `journal-gray`, `journal-minimal`)
- `hla.layout` (default: `elk`)
- `hla.inputShape` (used when verify-on-save is enabled)
- `hla.pad`
- `hla.className` (optional fixed class target)
