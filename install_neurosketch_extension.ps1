$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ext = Join-Path $root "vscode_neurosketch_sidepanel"

if (-not (Test-Path $ext)) {
    throw "Extension folder not found: $ext"
}

Push-Location $ext
try {
    npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed" }

    npm run compile
    if ($LASTEXITCODE -ne 0) { throw "npm run compile failed" }

    npm run package:vsix
    if ($LASTEXITCODE -ne 0) { throw "npm run package:vsix failed" }

    $vsix = Get-ChildItem -Filter "*.vsix" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $vsix) {
        throw "VSIX package not generated."
    }
    code --install-extension $vsix.FullName --force
    if ($LASTEXITCODE -ne 0) { throw "code --install-extension failed" }
}
finally {
    Pop-Location
}

Write-Host "NeuroSketch VS Code extension installed."
Write-Host "Open your normal VS Code window and run: NeuroSketch: Open Diagram Beside"
