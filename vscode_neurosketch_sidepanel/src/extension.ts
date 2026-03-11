import * as vscode from "vscode";
import { ChildProcess, execFile } from "child_process";
import * as fs from "fs/promises";
import * as path from "path";
import * as os from "os";
import * as crypto from "crypto";

type TriggerReason = "init" | "manual" | "save" | "typing" | "switch";

interface StatusPayload {
  state: "idle" | "running" | "ok" | "error";
  message: string;
  model?: string;
  nodes?: number;
  edges?: number;
  layers?: number;
  paramsTotal?: number;
  activations?: string;
  layerDetails?: string[];
  mode?: string;
  updatedAt?: string;
  reason?: TriggerReason;
}

interface GraphNodePayload {
  kind?: string;
  label?: string;
  params?: Record<string, unknown>;
}

interface GraphPayload {
  model?: { name?: string };
  nodes?: GraphNodePayload[];
  edges?: unknown[];
  meta?: { mode?: string };
}

const VIEW_ID = "neurosketch.diagramView";
const DEFAULT_STATUS: StatusPayload = {
  state: "idle",
  message: "Open a Python model file to start live diagram updates.",
};

export function activate(context: vscode.ExtensionContext): void {
  const provider = new NeuroSketchSidePanelProvider(context);

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(VIEW_ID, provider, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("neurosketch.openPanel", async () => {
      await provider.openPanel();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("neurosketch.openDiagramBeside", async () => {
      await provider.openDiagramBeside();
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("neurosketch.refreshDiagram", async () => {
      await provider.refresh("manual");
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("neurosketch.toggleRealtime", () => {
      provider.toggleRealtime();
    }),
  );

  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument(async (doc) => {
      await provider.handleSave(doc);
    }),
  );

  context.subscriptions.push(
    vscode.workspace.onDidChangeTextDocument(async (event) => {
      await provider.handleTyping(event.document);
    }),
  );

  context.subscriptions.push(
    vscode.window.onDidChangeActiveTextEditor(async (editor) => {
      await provider.handleEditorSwitch(editor);
    }),
  );
}

export function deactivate(): void {}

class NeuroSketchSidePanelProvider implements vscode.WebviewViewProvider {
  private readonly context: vscode.ExtensionContext;
  private view: vscode.WebviewView | undefined;
  private editorPanel: vscode.WebviewPanel | undefined;
  private debounceTimer: NodeJS.Timeout | undefined;
  private activeProcess: ChildProcess | undefined;
  private requestToken = 0;
  private realtimeEnabled = true;
  private status: StatusPayload = DEFAULT_STATUS;
  private lastCandidateDoc: vscode.TextDocument | undefined;

  constructor(context: vscode.ExtensionContext) {
    this.context = context;
    this.realtimeEnabled = this.getConfig<boolean>("autoUpdateOnType", true);
  }

  public resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = { enableScripts: true };
    view.webview.html = this.getHtml(view.webview);
    view.webview.onDidReceiveMessage(async (message: { command?: string }) => {
      if (message.command === "refresh") {
        await this.refresh("manual");
      } else if (message.command === "toggleRealtime") {
        this.toggleRealtime();
      }
    });
    this.postStatus();
    void this.refresh("init");
  }

  public async openPanel(): Promise<void> {
    await vscode.commands.executeCommand("workbench.view.extension.neurosketch");
    try {
      await vscode.commands.executeCommand(`${VIEW_ID}.focus`);
    } catch {
      // ignore if focus command is unavailable; container open is enough
    }
    await this.refresh("manual");
  }

  public async openDiagramBeside(): Promise<void> {
    if (!this.editorPanel) {
      this.editorPanel = vscode.window.createWebviewPanel(
        "neurosketch.diagramBeside",
        "NeuroSketch Diagram",
        { viewColumn: vscode.ViewColumn.Beside, preserveFocus: false },
        { enableScripts: true, retainContextWhenHidden: true },
      );
      this.editorPanel.webview.html = this.getHtml(this.editorPanel.webview);
      this.editorPanel.webview.onDidReceiveMessage(async (message: { command?: string }) => {
        if (message.command === "refresh") {
          await this.refresh("manual");
        } else if (message.command === "toggleRealtime") {
          this.toggleRealtime();
        }
      });
      this.editorPanel.onDidDispose(() => {
        this.editorPanel = undefined;
      });
    } else {
      this.editorPanel.reveal(vscode.ViewColumn.Beside, false);
    }
    this.postStatus();
    await this.refresh("manual");
  }

  public async handleSave(doc: vscode.TextDocument): Promise<void> {
    if (!this.isCandidateDoc(doc)) {
      return;
    }
    this.lastCandidateDoc = doc;
    await this.runUpdate(doc, "save");
  }

  public async handleTyping(doc: vscode.TextDocument): Promise<void> {
    if (!this.isCandidateDoc(doc)) {
      return;
    }
    this.lastCandidateDoc = doc;
    if (!this.realtimeEnabled) {
      return;
    }
    const debounceMs = this.getConfig<number>("updateDebounceMs", 600);
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    this.debounceTimer = setTimeout(() => {
      void this.runUpdate(doc, "typing");
    }, debounceMs);
  }

  public async handleEditorSwitch(editor: vscode.TextEditor | undefined): Promise<void> {
    if (!editor) {
      return;
    }
    if (!this.isCandidateDoc(editor.document)) {
      return;
    }
    this.lastCandidateDoc = editor.document;
    await this.runUpdate(editor.document, "switch");
  }

  public async refresh(reason: TriggerReason): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    let targetDoc: vscode.TextDocument | undefined;
    if (editor && this.isCandidateDoc(editor.document)) {
      targetDoc = editor.document;
      this.lastCandidateDoc = editor.document;
    } else if (this.lastCandidateDoc && !this.lastCandidateDoc.isClosed) {
      targetDoc = this.lastCandidateDoc;
    }

    if (!targetDoc) {
      if (reason === "init") {
        this.setStatus({
          state: "idle",
          message: "Open a Python file to preview the architecture diagram.",
        });
        this.postDiagram("");
      }
      return;
    }
    await this.runUpdate(targetDoc, reason);
  }

  public toggleRealtime(): void {
    this.realtimeEnabled = !this.realtimeEnabled;
    this.setStatus({
      ...this.status,
      state: "ok",
      message: this.realtimeEnabled
        ? "Realtime typing updates enabled."
        : "Realtime typing updates disabled. Save (Ctrl+S) still updates.",
      reason: "manual",
      updatedAt: new Date().toISOString(),
    });
    void vscode.window.showInformationMessage(
      this.realtimeEnabled
        ? "NeuroSketch realtime typing updates enabled."
        : "NeuroSketch realtime typing updates disabled (save still updates).",
    );
    this.postStatus();
  }

  private async runUpdate(doc: vscode.TextDocument, reason: TriggerReason): Promise<void> {
    this.lastCandidateDoc = doc;
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(doc.uri);
    if (!workspaceFolder) {
      this.setStatus({
        state: "error",
        message: "File is not inside an open workspace folder.",
        reason,
      });
      return;
    }

    const token = ++this.requestToken;
    this.killRunningProcess();
    this.setStatus({
      ...this.status,
      state: "running",
      message:
        reason === "typing"
          ? "Updating diagram from current edits..."
          : reason === "save"
            ? "Updating diagram from saved file..."
            : "Building diagram...",
      reason,
      updatedAt: new Date().toISOString(),
    });

    const outputDir = path.join(workspaceFolder.uri.fsPath, ".neurosketch_vscode");
    await fs.mkdir(outputDir, { recursive: true });

    const useTempFromBuffer = reason === "typing" && doc.isDirty;
    let sourcePath = doc.uri.fsPath;
    let tempPath: string | undefined;

    try {
      if (useTempFromBuffer) {
        tempPath = path.join(
          os.tmpdir(),
          `neurosketch-live-${Date.now()}-${Math.floor(Math.random() * 100000)}.py`,
        );
        await fs.writeFile(tempPath, doc.getText(), "utf-8");
        sourcePath = tempPath;
      }

      const pythonPath = this.getConfig<string>("pythonPath", "python");
      const analyzerCommand = this.getConfig<string>("analyzerCommand", "neurosketch").trim();
      const renderer = this.getConfig<string>("renderer", "auto");
      const theme = this.getConfig<string>("theme", "journal-light");
      const layout = this.getConfig<string>("layout", "elk");
      const inputShape = this.getConfig<string>("inputShape", "1,3,224,224");
      const pad = String(this.getConfig<number>("pad", 40));
      const className = this.getConfig<string>("className", "").trim();
      const verifyOnSave = this.getConfig<boolean>("verifyOnSave", false);

      const analyzeArgs = [
        "analyze",
        "--source",
        sourcePath,
        "--output-dir",
        outputDir,
        "--formats",
        "json,d2,svg",
        "--renderer",
        renderer,
        "--theme",
        theme,
        "--layout",
        layout,
        "--pad",
        pad,
      ];
      if (className) {
        analyzeArgs.push("--class-name", className);
      }
      if (verifyOnSave && reason === "save") {
        analyzeArgs.push("--verify", "--input-shape", inputShape);
      }

      await this.execAnalyzer(
        analyzerCommand,
        pythonPath,
        analyzeArgs,
        workspaceFolder.uri.fsPath,
        token,
      );
      if (token !== this.requestToken) {
        return;
      }

      const graphJsonPath = path.join(outputDir, "graph.json");
      const graphSvgPath = path.join(outputDir, "graph.svg");

      const [jsonText, svgText] = await Promise.all([
        fs.readFile(graphJsonPath, "utf-8"),
        fs.readFile(graphSvgPath, "utf-8"),
      ]);
      const graph = JSON.parse(jsonText) as GraphPayload;
      const graphNodes = Array.isArray(graph.nodes) ? graph.nodes : [];
      const graphEdges = Array.isArray(graph.edges) ? graph.edges : [];
      const layersCount = graphNodes.filter((n) => {
        const kind = String(n?.kind ?? "").toLowerCase();
        return kind !== "input" && kind !== "output";
      }).length;
      const totalParams = graphNodes.reduce((acc, n) => {
        const kind = String(n?.kind ?? "");
        const rawParams = (n?.params ?? {}) as Record<string, unknown>;
        return acc + this.estimateParams(kind, rawParams);
      }, 0);
      const activationKinds = new Set(
        graphNodes
          .map((n) => String(n?.kind ?? ""))
          .filter((kind) => {
            const k = kind.toLowerCase();
            return (
              k === "relu" ||
              k === "gelu" ||
              k === "silu" ||
              k === "sigmoid" ||
              k === "tanh" ||
              k === "softmax" ||
              k === "leakyrelu"
            );
          }),
      );
      const activationLabel =
        activationKinds.size > 0 ? Array.from(activationKinds).join(", ") : "none";
      const layerDetails = graphNodes
        .filter((n) => {
          const kind = String(n?.kind ?? "").toLowerCase();
          return kind !== "input" && kind !== "output";
        })
        .map((n, idx) => {
          const rawLabel = String(n.label ?? `layer_${idx + 1}`);
          const label = rawLabel.includes(".")
            ? (() => {
                const [stage, tail] = rawLabel.split(".", 2);
                return `${stage}[${tail}]`;
              })()
            : rawLabel;
          const kind = String(n.kind ?? "Layer");
          const rawParams = (n.params ?? {}) as Record<string, unknown>;
          const paramText = this.summarizeParams(kind, rawParams);
          const paramCount = this.estimateParams(kind, rawParams);
          const countText = paramCount > 0 ? ` | params~${this.formatCompactCount(paramCount)}` : "";
          return paramText
            ? `${label}: ${kind} | ${paramText}${countText}`
            : `${label}: ${kind}${countText}`;
        });

      this.setStatus({
        state: "ok",
        message: reason === "save" ? "Updated on save." : "Diagram updated.",
        model: graph.model?.name ?? "Unknown",
        nodes: graphNodes.length,
        edges: graphEdges.length,
        layers: layersCount,
        paramsTotal: totalParams,
        activations: activationLabel,
        layerDetails,
        mode: graph.meta?.mode ?? "draft",
        updatedAt: new Date().toISOString(),
        reason,
      });
      this.postDiagram(svgText);
    } catch (error) {
      if (token !== this.requestToken) {
        return;
      }
      this.setStatus({
        state: "error",
        message: this.formatError(error),
        updatedAt: new Date().toISOString(),
        reason,
      });
    } finally {
      if (tempPath) {
        try {
          await fs.unlink(tempPath);
        } catch {
          // ignore temp cleanup errors
        }
      }
      if (token === this.requestToken) {
        this.activeProcess = undefined;
      }
    }
  }

  private killRunningProcess(): void {
    if (this.activeProcess && !this.activeProcess.killed) {
      this.activeProcess.kill();
    }
  }

  private execAnalyzer(
    analyzerCommand: string,
    pythonPath: string,
    analyzeArgs: string[],
    cwd: string,
    token: number,
  ): Promise<void> {
    const primaryCmd = analyzerCommand || "neurosketch";
    return this.execCommand(primaryCmd, analyzeArgs, cwd, token).catch(async (primaryErr) => {
      const fallbackArgs = ["-m", "neurosketch", ...analyzeArgs];
      try {
        await this.execCommand(pythonPath, fallbackArgs, cwd, token);
      } catch (fallbackErr) {
        const first = primaryErr instanceof Error ? primaryErr.message : String(primaryErr);
        const second = fallbackErr instanceof Error ? fallbackErr.message : String(fallbackErr);
        throw new Error(
          `Analyzer failed with "${primaryCmd}" and Python fallback.\nPrimary: ${first}\nFallback: ${second}`,
        );
      }
    });
  }

  private execCommand(
    command: string,
    args: string[],
    cwd: string,
    token: number,
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const proc = execFile(command, args, { cwd }, (error, _stdout, stderr) => {
        if (token !== this.requestToken) {
          resolve();
          return;
        }
        if (error) {
          if ((error as NodeJS.ErrnoException).code === "ENOENT") {
            reject(new Error(`Command not found: ${command}`));
            return;
          }
          const detail = stderr?.trim() ? `\n${stderr.trim()}` : "";
          reject(new Error(`Analyzer failed (${command}).${detail}`));
          return;
        }
        resolve();
      });
      this.activeProcess = proc;
    });
  }

  private isCandidateDoc(doc: vscode.TextDocument): boolean {
    if (doc.languageId === "python") {
      return true;
    }
    const file = doc.uri.fsPath.toLowerCase();
    return file.endsWith(".py");
  }

  private getConfig<T>(key: string, fallback: T): T {
    const value = vscode.workspace.getConfiguration("neurosketch").get<T>(key);
    return value === undefined ? fallback : value;
  }

  private setStatus(status: StatusPayload): void {
    this.status = status;
    this.postStatus();
  }

  private postStatus(): void {
    const payload = {
      type: "status",
      payload: {
        ...this.status,
        realtimeEnabled: this.realtimeEnabled,
      },
    };
    this.view?.webview.postMessage(payload);
    this.editorPanel?.webview.postMessage(payload);
  }

  private postDiagram(svg: string): void {
    const payload = {
      type: "diagram",
      payload: { svg },
    };
    this.view?.webview.postMessage(payload);
    this.editorPanel?.webview.postMessage(payload);
  }

  private formatError(error: unknown): string {
    if (error instanceof Error) {
      return error.message;
    }
    return String(error);
  }

  private summarizeParams(kind: string, params: Record<string, unknown>): string {
    if (!params || Object.keys(params).length === 0) {
      return "";
    }
    const kindLower = kind.toLowerCase();
    const first = (...keys: string[]): unknown => {
      for (const key of keys) {
        if (key in params) {
          return params[key];
        }
      }
      return undefined;
    };

    if (kindLower.startsWith("conv")) {
      const inCh = first("in_channels", "arg0");
      const outCh = first("out_channels", "arg1");
      const k = first("kernel_size", "arg2");
      const s = first("stride");
      const p = first("padding");
      const chunks: string[] = [];
      if (inCh !== undefined && outCh !== undefined) chunks.push(`${inCh}->${outCh}`);
      if (k !== undefined) chunks.push(`k=${k}`);
      if (s !== undefined) chunks.push(`s=${s}`);
      if (p !== undefined) chunks.push(`p=${p}`);
      return chunks.join(" ");
    }

    if (kindLower === "linear") {
      const inF = first("in_features", "arg0");
      const outF = first("out_features", "arg1");
      if (inF !== undefined && outF !== undefined) return `${inF}->${outF}`;
      return "";
    }

    if (kindLower.includes("dropout")) {
      const p = first("p", "arg0");
      return p !== undefined ? `p=${p}` : "";
    }

    if (kindLower.includes("pool")) {
      const k = first("kernel_size", "arg0");
      const s = first("stride");
      const chunks: string[] = [];
      if (k !== undefined) chunks.push(`k=${k}`);
      if (s !== undefined) chunks.push(`s=${s}`);
      return chunks.join(" ");
    }

    const entries = Object.entries(params).slice(0, 2);
    return entries.map(([k, v]) => `${k}=${v}`).join(", ");
  }

  private asInt(value: unknown): number | undefined {
    if (typeof value === "number" && Number.isFinite(value) && Number.isInteger(value)) {
      return value;
    }
    if (typeof value === "string") {
      const raw = value.trim();
      if (!raw) {
        return undefined;
      }
      if (/^-?\d+$/.test(raw)) {
        const parsed = Number(raw);
        return Number.isInteger(parsed) ? parsed : undefined;
      }
      try {
        const parsed = JSON.parse(raw) as unknown;
        return this.asInt(parsed);
      } catch {
        return undefined;
      }
    }
    return undefined;
  }

  private product(value: unknown): number | undefined {
    const direct = this.asInt(value);
    if (direct !== undefined) {
      return direct;
    }
    if (Array.isArray(value)) {
      let prod = 1;
      for (const item of value) {
        const iv = this.asInt(item);
        if (iv === undefined) {
          return undefined;
        }
        prod *= iv;
      }
      return prod;
    }
    if (typeof value === "string") {
      const raw = value.trim();
      if (!raw) {
        return undefined;
      }
      const parts = raw.match(/\d+/g);
      if (!parts || parts.length === 0) {
        return undefined;
      }
      return parts.reduce((acc, part) => acc * Number(part), 1);
    }
    return undefined;
  }

  private boolParam(value: unknown, defaultValue: boolean): boolean {
    if (typeof value === "boolean") {
      return value;
    }
    if (typeof value === "string") {
      const raw = value.trim().toLowerCase();
      if (["false", "0", "no"].includes(raw)) {
        return false;
      }
      if (["true", "1", "yes"].includes(raw)) {
        return true;
      }
    }
    return defaultValue;
  }

  private formatCompactCount(value: number): string {
    if (value >= 1_000_000) {
      return `${(value / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
    }
    if (value >= 1_000) {
      return `${(value / 1_000).toFixed(1).replace(/\.0$/, "")}K`;
    }
    return String(value);
  }

  private estimateParams(kind: string, params: Record<string, unknown>): number {
    if (!params || Object.keys(params).length === 0) {
      return 0;
    }
    const first = (...keys: string[]): unknown => {
      for (const key of keys) {
        if (key in params) {
          return params[key];
        }
      }
      return undefined;
    };

    const kindLower = kind.toLowerCase();
    if (kindLower.startsWith("conv")) {
      const inCh = this.asInt(first("in_channels", "arg0"));
      const outCh = this.asInt(first("out_channels", "arg1"));
      const kernelProd = this.product(first("kernel_size", "arg2"));
      const groups = this.asInt(first("groups")) ?? 1;
      const bias = this.boolParam(first("bias"), true);
      if (
        inCh === undefined ||
        outCh === undefined ||
        kernelProd === undefined ||
        groups <= 0
      ) {
        return 0;
      }
      const inPerGroup = Math.floor(inCh / groups);
      const weight = outCh * inPerGroup * kernelProd;
      return weight + (bias ? outCh : 0);
    }

    if (kindLower === "linear") {
      const inF = this.product(first("in_features", "arg0"));
      const outF = this.product(first("out_features", "arg1"));
      const bias = this.boolParam(first("bias"), true);
      if (inF === undefined || outF === undefined) {
        return 0;
      }
      return inF * outF + (bias ? outF : 0);
    }

    if (kindLower.includes("batchnorm")) {
      const n = this.asInt(first("num_features", "arg0"));
      return n !== undefined ? n * 2 : 0;
    }

    if (kindLower === "embedding") {
      const n = this.asInt(first("num_embeddings", "arg0"));
      const d = this.asInt(first("embedding_dim", "arg1"));
      return n !== undefined && d !== undefined ? n * d : 0;
    }

    return 0;
  }

  private getHtml(webview: vscode.Webview): string {
    const nonce = crypto.randomBytes(16).toString("hex");
    const csp = [
      "default-src 'none'",
      `img-src ${webview.cspSource} data:`,
      `style-src ${webview.cspSource} 'unsafe-inline'`,
      `script-src 'nonce-${nonce}'`,
    ].join("; ");

    return `<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="${csp}" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <style>
    :root {
      --bg: #f8fafc;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --line: #dbe3ef;
      --ok: #16a34a;
      --run: #2563eb;
      --err: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "Inter", sans-serif;
      color: var(--ink);
      background: radial-gradient(700px 180px at 20% -10%, #dbeafe 0%, transparent 70%), var(--bg);
    }
    .bar {
      display: flex;
      gap: 8px;
      align-items: center;
      padding: 10px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 2;
    }
    button {
      border: 1px solid #c9d8ea;
      background: white;
      border-radius: 8px;
      padding: 4px 10px;
      font-size: 12px;
      cursor: pointer;
    }
    button.zoom-btn {
      min-width: 34px;
      padding: 4px 8px;
    }
    button:hover { background: #f1f5f9; }
    .status {
      margin-left: auto;
      font-size: 12px;
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #94a3b8;
    }
    .meta {
      padding: 10px;
      font-size: 12px;
      color: var(--muted);
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px 10px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    .meta strong { color: var(--ink); font-weight: 600; }
    #error {
      color: var(--err);
      white-space: pre-wrap;
      padding: 8px 10px 0;
      font-size: 12px;
      display: none;
    }
    #canvas {
      padding: 10px;
      overflow: auto;
      min-height: calc(100vh - 120px);
    }
    .frame {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
      min-height: 240px;
      padding: 8px;
      box-shadow: 0 10px 26px rgba(2, 6, 23, 0.08);
    }
    #diagram {
      min-height: 220px;
      position: relative;
      overflow: hidden;
      cursor: grab;
      user-select: none;
      touch-action: none;
    }
    #diagram.dragging {
      cursor: grabbing;
    }
    #zoomPct {
      font-size: 11px;
      color: var(--muted);
      min-width: 44px;
      text-align: center;
    }
    #diagram svg {
      width: auto;
      height: auto;
      max-width: none;
      display: block;
      transform-origin: 0 0;
      will-change: transform;
    }
    .empty {
      color: var(--muted);
      padding: 18px;
      text-align: center;
      font-size: 12px;
    }
    .layers {
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
      box-shadow: 0 10px 26px rgba(2, 6, 23, 0.08);
      padding: 8px 10px;
    }
    .layers h3 {
      margin: 0 0 6px;
      font-size: 12px;
      color: var(--ink);
    }
    #layerList {
      margin: 0;
      padding-left: 16px;
      max-height: 180px;
      overflow: auto;
      font-size: 11px;
      color: var(--muted);
    }
    #layerList li {
      margin: 4px 0;
      line-height: 1.3;
    }
  </style>
</head>
<body>
  <div class="bar">
    <button id="refreshBtn">Refresh</button>
    <button id="toggleBtn">Toggle Realtime</button>
    <button id="zoomOutBtn" class="zoom-btn">-</button>
    <button id="zoomInBtn" class="zoom-btn">+</button>
    <button id="zoomFitBtn" class="zoom-btn">Fit</button>
    <button id="zoomResetBtn" class="zoom-btn">100%</button>
    <span id="zoomPct">100%</span>
    <div class="status">
      <span id="stateDot" class="dot"></span>
      <span id="stateText">idle</span>
    </div>
  </div>
  <div class="meta">
    <div><strong>Model:</strong> <span id="model">-</span></div>
    <div><strong>Mode:</strong> <span id="mode">-</span></div>
    <div><strong>Layers:</strong> <span id="layers">-</span></div>
    <div><strong>Params:</strong> <span id="paramsTotal">-</span></div>
    <div><strong>Activations:</strong> <span id="activations">-</span></div>
    <div><strong>Nodes:</strong> <span id="nodes">-</span></div>
    <div><strong>Edges:</strong> <span id="edges">-</span></div>
    <div><strong>Reason:</strong> <span id="reason">-</span></div>
    <div><strong>Realtime:</strong> <span id="realtime">-</span></div>
  </div>
  <div id="error"></div>
  <div id="canvas">
    <div class="frame">
      <div id="diagram" class="empty">No diagram yet.</div>
    </div>
    <div class="layers">
      <h3>Layer Details</h3>
      <ol id="layerList"><li>No layers yet.</li></ol>
    </div>
  </div>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const stateDot = document.getElementById("stateDot");
    const stateText = document.getElementById("stateText");
    const model = document.getElementById("model");
    const mode = document.getElementById("mode");
    const layers = document.getElementById("layers");
    const paramsTotal = document.getElementById("paramsTotal");
    const activations = document.getElementById("activations");
    const nodes = document.getElementById("nodes");
    const edges = document.getElementById("edges");
    const reason = document.getElementById("reason");
    const realtime = document.getElementById("realtime");
    const errorEl = document.getElementById("error");
    const diagram = document.getElementById("diagram");
    const layerList = document.getElementById("layerList");
    const zoomPct = document.getElementById("zoomPct");
    const esc = (s) =>
      String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    const formatCompact = (n) => {
      const x = Number(n);
      if (!Number.isFinite(x) || x <= 0) return "-";
      if (x >= 1000000) return (x / 1000000).toFixed(1).replace(/\\.0$/, "") + "M";
      if (x >= 1000) return (x / 1000).toFixed(1).replace(/\\.0$/, "") + "K";
      return String(Math.round(x));
    };

    let currentSvg = null;
    let zoomScale = 1;
    let zoomTx = 0;
    let zoomTy = 0;
    let dragging = false;
    let dragLastX = 0;
    let dragLastY = 0;

    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const updateZoomLabel = () => {
      zoomPct.textContent = String(Math.round(zoomScale * 100)) + "%";
    };
    const applyTransform = () => {
      if (!currentSvg) return;
      currentSvg.style.transform =
        "translate(" + zoomTx + "px, " + zoomTy + "px) scale(" + zoomScale + ")";
      updateZoomLabel();
    };
    const zoomAt = (factor, clientX, clientY) => {
      if (!currentSvg) return;
      const rect = diagram.getBoundingClientRect();
      const mx = clientX - rect.left;
      const my = clientY - rect.top;
      const next = clamp(zoomScale * factor, 0.2, 8);
      if (next === zoomScale) return;
      const ratio = next / zoomScale;
      zoomTx = mx - (mx - zoomTx) * ratio;
      zoomTy = my - (my - zoomTy) * ratio;
      zoomScale = next;
      applyTransform();
    };
    const fitDiagram = () => {
      if (!currentSvg) return;
      const viewport = diagram.getBoundingClientRect();
      const vb = currentSvg.viewBox && currentSvg.viewBox.baseVal;
      let baseW = vb && vb.width ? vb.width : 0;
      let baseH = vb && vb.height ? vb.height : 0;
      if (!baseW || !baseH) {
        const wAttr = Number(currentSvg.getAttribute("width"));
        const hAttr = Number(currentSvg.getAttribute("height"));
        baseW = Number.isFinite(wAttr) && wAttr > 0 ? wAttr : 1200;
        baseH = Number.isFinite(hAttr) && hAttr > 0 ? hAttr : 600;
      }
      const pad = 16;
      const sx = (viewport.width - pad) / baseW;
      const sy = (viewport.height - pad) / baseH;
      zoomScale = clamp(Math.min(sx, sy), 0.2, 3);
      zoomTx = (viewport.width - baseW * zoomScale) / 2;
      zoomTy = (viewport.height - baseH * zoomScale) / 2;
      applyTransform();
    };
    const resetZoom = () => {
      zoomScale = 1;
      zoomTx = 0;
      zoomTy = 0;
      applyTransform();
    };

    const colors = {
      idle: "#94a3b8",
      running: "#2563eb",
      ok: "#16a34a",
      error: "#b91c1c",
    };

    document.getElementById("refreshBtn").addEventListener("click", () => {
      vscode.postMessage({ command: "refresh" });
    });
    document.getElementById("toggleBtn").addEventListener("click", () => {
      vscode.postMessage({ command: "toggleRealtime" });
    });
    document.getElementById("zoomInBtn").addEventListener("click", () => {
      const rect = diagram.getBoundingClientRect();
      zoomAt(1.15, rect.left + rect.width / 2, rect.top + rect.height / 2);
    });
    document.getElementById("zoomOutBtn").addEventListener("click", () => {
      const rect = diagram.getBoundingClientRect();
      zoomAt(1 / 1.15, rect.left + rect.width / 2, rect.top + rect.height / 2);
    });
    document.getElementById("zoomFitBtn").addEventListener("click", () => {
      fitDiagram();
    });
    document.getElementById("zoomResetBtn").addEventListener("click", () => {
      resetZoom();
    });
    diagram.addEventListener("wheel", (e) => {
      if (!currentSvg) return;
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      zoomAt(factor, e.clientX, e.clientY);
    }, { passive: false });
    diagram.addEventListener("pointerdown", (e) => {
      if (!currentSvg || e.button !== 0) return;
      dragging = true;
      dragLastX = e.clientX;
      dragLastY = e.clientY;
      diagram.classList.add("dragging");
    });
    window.addEventListener("pointermove", (e) => {
      if (!dragging || !currentSvg) return;
      const dx = e.clientX - dragLastX;
      const dy = e.clientY - dragLastY;
      dragLastX = e.clientX;
      dragLastY = e.clientY;
      zoomTx += dx;
      zoomTy += dy;
      applyTransform();
    });
    window.addEventListener("pointerup", () => {
      dragging = false;
      diagram.classList.remove("dragging");
    });

    window.addEventListener("message", (event) => {
      const message = event.data;
      if (message.type === "status") {
        const p = message.payload || {};
        stateText.textContent = p.message || p.state || "idle";
        stateDot.style.background = colors[p.state] || colors.idle;
        model.textContent = p.model || "-";
        mode.textContent = p.mode || "-";
        layers.textContent = Number.isFinite(p.layers) ? String(p.layers) : "-";
        paramsTotal.textContent = formatCompact(p.paramsTotal);
        activations.textContent = p.activations || "-";
        nodes.textContent = Number.isFinite(p.nodes) ? String(p.nodes) : "-";
        edges.textContent = Number.isFinite(p.edges) ? String(p.edges) : "-";
        reason.textContent = p.reason || "-";
        realtime.textContent = p.realtimeEnabled ? "on" : "off";
        const rows = Array.isArray(p.layerDetails) ? p.layerDetails : [];
        if (rows.length > 0) {
          layerList.innerHTML = rows.map((row) => "<li>" + esc(row) + "</li>").join("");
        } else {
          layerList.innerHTML = "<li>No layers yet.</li>";
        }
        if (p.state === "error" && p.message) {
          errorEl.style.display = "block";
          errorEl.textContent = p.message;
        } else {
          errorEl.style.display = "none";
          errorEl.textContent = "";
        }
      } else if (message.type === "diagram") {
        const svg = message.payload?.svg || "";
        if (!svg) {
          currentSvg = null;
          diagram.className = "empty";
          diagram.textContent = "No diagram yet.";
          resetZoom();
          return;
        }
        diagram.className = "";
        diagram.innerHTML = svg;
        currentSvg = diagram.querySelector("svg");
        if (currentSvg) {
          currentSvg.setAttribute("preserveAspectRatio", "xMidYMid meet");
          fitDiagram();
        } else {
          resetZoom();
        }
      }
    });
  </script>
</body>
</html>`;
  }
}
