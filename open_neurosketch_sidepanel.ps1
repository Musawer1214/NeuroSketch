$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ext = Join-Path $root "vscode_neurosketch_sidepanel"

if (-not (Test-Path $ext)) {
    throw "Extension folder not found: $ext"
}

Push-Location $ext
try {
    if (-not (Test-Path "node_modules")) {
        npm install
    }
    npm run compile
}
finally {
    Pop-Location
}

code --extensionDevelopmentPath "$ext" --new-window "$root"
