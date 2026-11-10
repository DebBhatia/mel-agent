"""
MEL AI AGENT - Authorized File Workspace
=========================================
Sandboxed file access for runtime tasks. Only explicitly configured roots
are visible:

    MEL_FILE_ROOTS   os.pathsep-separated read roots (default: ~/Desktop, ~/Documents)
    MEL_WORKSPACE    MEL's own output/working-copy directory (default: ~/MEL-Workspace)

Rules:
- Every path is resolved (symlinks included) and must stay inside a root;
  traversal and symlink escapes are rejected.
- Originals are never modified: writes only ever create *new* files
  (a numeric suffix is added instead of overwriting), and edits only touch
  working copies inside the workspace.
- Documents are parsed as data with the standard library (zip/XML for
  .docx/.xlsx); macros, formulas and embedded code are never executed.
  Macro-enabled formats are refused outright.
- Content stays local: nothing here talks to a network.
"""

import csv
import json
import os
import re
import shutil
import statistics
import zipfile
from datetime import datetime
from xml.etree import ElementTree as ET

TEXT_EXTS = {".txt", ".md", ".csv", ".tsv", ".json", ".log"}
DOC_EXTS = {".docx", ".xlsx", ".pdf"}
SUPPORTED_EXTS = TEXT_EXTS | DOC_EXTS
BLOCKED_EXTS = {".xlsm", ".xltm", ".docm", ".dotm", ".pptm", ".exe", ".app", ".sh",
                ".command", ".py", ".js", ".scpt", ".bat", ".ps1"}
WRITE_EXTS = {".md", ".txt", ".csv", ".json"}

MAX_READ_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARS = 20000
MAX_SCAN_FILES = 2000
MAX_SCAN_DEPTH = 4

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


class FileAccessError(PermissionError):
    pass


def _default_roots() -> list[str]:
    env = os.getenv("MEL_FILE_ROOTS", "")
    if env:
        return [p for p in env.split(os.pathsep) if p.strip()]
    return [os.path.expanduser("~/Desktop"), os.path.expanduser("~/Documents")]


class FileWorkspace:
    def __init__(self, roots: list[str] | None = None, workspace: str | None = None):
        self.workspace = os.path.realpath(
            workspace or os.getenv("MEL_WORKSPACE", os.path.expanduser("~/MEL-Workspace")))
        named = roots if roots is not None else _default_roots()
        # Named roots ("desktop" -> path) let requests like "save to my Desktop" resolve
        self.roots: dict[str, str] = {}
        for r in named:
            real = os.path.realpath(os.path.expanduser(r))
            self.roots[os.path.basename(real).lower() or real] = real
        self.roots["workspace"] = self.workspace

    # ── boundary checks ──
    def _inside(self, path: str, root: str) -> bool:
        try:
            return os.path.commonpath([path, root]) == root
        except ValueError:
            return False

    def resolve(self, path: str) -> str:
        """Resolve a user/tool-supplied path and enforce the root boundary.
        Relative paths are taken relative to the workspace."""
        if not isinstance(path, str) or not path or "\x00" in path:
            raise FileAccessError("Invalid path")
        raw = os.path.expanduser(path)
        if not os.path.isabs(raw):
            raw = os.path.join(self.workspace, raw)
        real = os.path.realpath(raw)
        if not any(self._inside(real, root) for root in self.roots.values()):
            raise FileAccessError("That location is outside the folders I'm allowed to use")
        return real

    def root_for(self, name: str) -> str:
        key = (name or "workspace").lower().strip()
        if key not in self.roots:
            raise FileAccessError(f"'{name}' isn't an approved folder")
        return self.roots[key]

    # ── read-only ──
    def find(self, query: str = "", exts: set[str] | None = None, root: str | None = None,
             limit: int = 10) -> list[dict]:
        """Newest-first listing of supported files under approved roots
        matching all query words in the filename."""
        roots = [self.root_for(root)] if root else list(dict.fromkeys(self.roots.values()))
        words = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(w) > 1]
        allowed = exts or SUPPORTED_EXTS
        found, scanned = [], 0
        for base in roots:
            if not os.path.isdir(base):
                continue
            base_depth = base.rstrip(os.sep).count(os.sep)
            for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                if dirpath.count(os.sep) - base_depth >= MAX_SCAN_DEPTH:
                    dirnames[:] = []
                for fn in filenames:
                    scanned += 1
                    if scanned > MAX_SCAN_FILES:
                        break
                    ext = os.path.splitext(fn)[1].lower()
                    if fn.startswith(".") or ext not in allowed or ext in BLOCKED_EXTS:
                        continue
                    if words and not all(w in fn.lower() for w in words):
                        continue
                    full = os.path.join(dirpath, fn)
                    if os.path.islink(full):
                        try:
                            self.resolve(full)
                        except FileAccessError:
                            continue
                    try:
                        st = os.stat(full)
                    except OSError:
                        continue
                    found.append({"path": full, "name": fn, "ext": ext,
                                  "size": st.st_size, "modified": st.st_mtime})
        found.sort(key=lambda f: f["modified"], reverse=True)
        return found[:limit]

    def read(self, path: str) -> dict:
        """Returns {"path","kind","text", optional "table"}. Never executes content."""
        real = self.resolve(path)
        ext = os.path.splitext(real)[1].lower()
        if ext in BLOCKED_EXTS:
            raise FileAccessError("I don't open macro-enabled or executable files")
        if ext not in SUPPORTED_EXTS:
            raise FileAccessError(f"Unsupported file type: {ext or 'none'}")
        if not os.path.isfile(real):
            raise FileNotFoundError("File not found")
        if os.path.getsize(real) > MAX_READ_BYTES:
            raise FileAccessError("File is too large to read safely")

        if ext in (".csv", ".tsv"):
            with open(real, newline="", encoding="utf-8", errors="replace") as f:
                rows = list(csv.reader(f, delimiter="\t" if ext == ".tsv" else ","))[:5000]
            return {"path": real, "kind": "table", "table": rows,
                    "text": "\n".join(", ".join(r) for r in rows[:50])[:MAX_TEXT_CHARS]}
        if ext in TEXT_EXTS:
            with open(real, encoding="utf-8", errors="replace") as f:
                return {"path": real, "kind": "text", "text": f.read(MAX_TEXT_CHARS)}
        if ext == ".xlsx":
            rows = _read_xlsx(real)
            return {"path": real, "kind": "table", "table": rows,
                    "text": "\n".join(", ".join(r) for r in rows[:50])[:MAX_TEXT_CHARS]}
        if ext == ".docx":
            return {"path": real, "kind": "text", "text": _read_docx(real)[:MAX_TEXT_CHARS]}
        if ext == ".pdf":
            return {"path": real, "kind": "text", "text": _read_pdf(real)[:MAX_TEXT_CHARS]}
        raise FileAccessError("Unsupported file type")

    # ── writes (new files / workspace copies only) ──
    def _unique(self, path: str) -> str:
        if not os.path.exists(path):
            return path
        stem, ext = os.path.splitext(path)
        for i in range(2, 1000):
            candidate = f"{stem} ({i}){ext}"
            if not os.path.exists(candidate):
                return candidate
        raise FileAccessError("Too many files with that name")

    def write_new(self, filename: str, content: str, root: str = "workspace") -> str:
        """Create a new file; never overwrites (adds a suffix instead)."""
        base = self.root_for(root)
        name = os.path.basename(filename or "")
        name = re.sub(r"[^\w\-. ()]", "_", name).strip(" .")
        if not name:
            raise FileAccessError("Invalid filename")
        if os.path.splitext(name)[1].lower() not in WRITE_EXTS:
            name += ".md"
        os.makedirs(base, exist_ok=True)
        target = self._unique(os.path.join(base, name))
        self.resolve(target)  # boundary re-check after joining
        with open(target, "x", encoding="utf-8") as f:
            f.write(content)
        return target

    def copy_to_workspace(self, path: str) -> str:
        src = self.resolve(path)
        if not os.path.isfile(src):
            raise FileNotFoundError("File not found")
        os.makedirs(self.workspace, exist_ok=True)
        target = self._unique(os.path.join(self.workspace, os.path.basename(src)))
        shutil.copy2(src, target, follow_symlinks=True)
        return target

    def edit_working_copy(self, path: str, content: str) -> str:
        """Edits are only permitted on files inside the MEL workspace."""
        real = self.resolve(path)
        if not self._inside(real, self.workspace):
            raise FileAccessError("I only edit working copies in the MEL workspace -- originals stay untouched")
        if os.path.splitext(real)[1].lower() not in WRITE_EXTS:
            raise FileAccessError("I can only edit text-based working copies")
        with open(real, "w", encoding="utf-8") as f:
            f.write(content)
        return real


# ─────────────────────────────────────────────
# Stdlib document parsers (data only)
# ─────────────────────────────────────────────
def _safe_zip(path: str) -> zipfile.ZipFile:
    zf = zipfile.ZipFile(path)
    if sum(i.file_size for i in zf.infolist()) > 50 * MAX_READ_BYTES:
        zf.close()
        raise FileAccessError("Document expands too large to read safely")
    return zf


def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref or "A")
    n = 0
    for ch in (letters.group(0) if letters else "A"):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _read_xlsx(path: str, max_rows: int = 5000) -> list[list[str]]:
    """First worksheet as rows of strings. Uses cached cell values only --
    formulas are never evaluated."""
    with _safe_zip(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        sheets = sorted(n for n in zf.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
        if not sheets:
            return []
        root = ET.fromstring(zf.read(sheets[0]))
        rows = []
        for row in root.iter(f"{{{_NS['m']}}}row"):
            cells: dict[int, str] = {}
            for c in row.findall("m:c", _NS):
                v = c.find("m:v", _NS)
                t = c.get("t")
                if t == "s" and v is not None:
                    val = shared[int(v.text)] if v.text and int(v.text) < len(shared) else ""
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(f"{{{_NS['m']}}}t"))
                else:
                    val = v.text if v is not None and v.text else ""
                cells[_col_index(c.get("r", ""))] = val
            if cells:
                width = max(cells) + 1
                rows.append([cells.get(i, "") for i in range(width)])
            if len(rows) >= max_rows:
                break
        return rows


def _read_docx(path: str) -> str:
    with _safe_zip(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    paras = []
    for p in root.iter(f"{{{_NS['w']}}}p"):
        text = "".join(t.text or "" for t in p.iter(f"{{{_NS['w']}}}t"))
        if text.strip():
            paras.append(text)
    return "\n".join(paras)


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise FileAccessError("PDF reading needs the 'pypdf' package (pip install pypdf)")
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages[:50])


# ─────────────────────────────────────────────
# Deterministic analysis (no model call)
# ─────────────────────────────────────────────
def _to_number(s: str):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def analyze(doc: dict) -> dict:
    """Compact, model-free facts about a document."""
    name = os.path.basename(doc.get("path", ""))
    if doc.get("kind") == "table":
        rows = [r for r in doc.get("table", []) if any(str(c).strip() for c in r)]
        if not rows:
            return {"name": name, "kind": "table", "rows": 0, "columns": [], "numeric": {}}
        header, body = rows[0], rows[1:]
        numeric = {}
        for i, col in enumerate(header):
            vals = [n for r in body if i < len(r) and (n := _to_number(r[i])) is not None]
            if vals and len(vals) >= max(1, len(body) // 2):
                numeric[col or f"col{i + 1}"] = {
                    "sum": round(sum(vals), 2), "mean": round(statistics.fmean(vals), 2),
                    "min": round(min(vals), 2), "max": round(max(vals), 2), "count": len(vals)}
        return {"name": name, "kind": "table", "rows": len(body), "columns": header, "numeric": numeric}
    text = doc.get("text", "")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    headings = [l.lstrip("# ") for l in lines if l.startswith("#")][:10]
    return {"name": name, "kind": "text", "words": len(text.split()), "lines": len(lines),
            "headings": headings, "preview": " ".join(lines[:5])[:400]}


def render_analysis_markdown(facts: dict, narrative: str = "") -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = [f"# Summary: {facts.get('name', 'document')}", f"_Generated by Mel on {stamp}. Original file was not modified._", ""]
    if narrative:
        out += ["## Overview", narrative.strip(), ""]
    if facts.get("kind") == "table":
        out += ["## Data", f"- Rows: {facts['rows']}", f"- Columns: {', '.join(map(str, facts['columns']))}", ""]
        if facts["numeric"]:
            out += ["| Column | Sum | Mean | Min | Max |", "|---|---|---|---|---|"]
            for col, s in facts["numeric"].items():
                out.append(f"| {col} | {s['sum']} | {s['mean']} | {s['min']} | {s['max']} |")
    else:
        out += ["## Document", f"- Words: {facts.get('words', 0)}", f"- Lines: {facts.get('lines', 0)}"]
        if facts.get("headings"):
            out += ["- Sections: " + "; ".join(facts["headings"])]
        if facts.get("preview"):
            out += ["", "## Opening", facts["preview"]]
    return "\n".join(out) + "\n"


def analysis_as_json(facts: dict) -> str:
    return json.dumps(facts, default=str)[:2000]
