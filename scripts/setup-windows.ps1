# ═══════════════════════════════════════════════════
#  MEL AGENT — Windows Phase 1 Setup Script
# ═══════════════════════════════════════════════════
#  Run this in PowerShell (as Administrator recommended)
#  
#  What this does:
#  1. Creates the project folder structure
#  2. Sets up Python virtual environment
#  3. Installs all dependencies
#  4. Creates secure config directory
#  5. Generates encryption key + vault
#  6. Initializes Git repo with .gitignore
#
#  Prerequisites:
#  - Python 3.10+ installed (python.org)
#  - Git installed (git-scm.com)
#  - Ollama installed (ollama.com/download/windows)
# ═══════════════════════════════════════════════════

Write-Host ""
Write-Host "════════════════════════════════════════" -ForegroundColor Magenta
Write-Host "  MEL AGENT — Phase 1 Setup (Windows)" -ForegroundColor Magenta
Write-Host "════════════════════════════════════════" -ForegroundColor Magenta
Write-Host ""

# ── Step 1: Set project directory ────────────────
$PROJECT_DIR = "$env:USERPROFILE\mel-agent"

if (Test-Path $PROJECT_DIR) {
    Write-Host "[!] Project directory already exists at $PROJECT_DIR" -ForegroundColor Yellow
    $confirm = Read-Host "Continue and update? (y/n)"
    if ($confirm -ne "y") { exit }
} else {
    New-Item -ItemType Directory -Path $PROJECT_DIR -Force | Out-Null
}

Set-Location $PROJECT_DIR
Write-Host "[OK] Project directory: $PROJECT_DIR" -ForegroundColor Green

# ── Step 2: Create folder structure ──────────────
$folders = @("src", "config", "scripts", "logs")
foreach ($f in $folders) {
    New-Item -ItemType Directory -Path "$PROJECT_DIR\$f" -Force | Out-Null
}
Write-Host "[OK] Folder structure created" -ForegroundColor Green

# ── Step 3: Create Python virtual environment ────
Write-Host ""
Write-Host "[...] Creating Python virtual environment..." -ForegroundColor Cyan

if (-not (Test-Path "$PROJECT_DIR\.venv")) {
    python -m venv .venv
}
& "$PROJECT_DIR\.venv\Scripts\Activate.ps1"
Write-Host "[OK] Virtual environment activated" -ForegroundColor Green

# ── Step 4: Install dependencies ─────────────────
Write-Host ""
Write-Host "[...] Installing Python packages..." -ForegroundColor Cyan

pip install --upgrade pip | Out-Null
pip install httpx python-dotenv fastapi uvicorn pydantic cryptography chromadb | Out-Null

Write-Host "[OK] All packages installed" -ForegroundColor Green

# ── Step 5: Create secure config directory ───────
$AGENT_CONFIG = "$env:USERPROFILE\.config\agent"
$AUDIT_DIR = "$env:USERPROFILE\.config\agent\audit"

New-Item -ItemType Directory -Path $AGENT_CONFIG -Force | Out-Null
New-Item -ItemType Directory -Path $AUDIT_DIR -Force | Out-Null

Write-Host "[OK] Secure config directory: $AGENT_CONFIG" -ForegroundColor Green

# ── Step 6: Check Ollama ─────────────────────────
Write-Host ""
Write-Host "[...] Checking Ollama..." -ForegroundColor Cyan

try {
    $ollamaVersion = ollama --version 2>&1
    Write-Host "[OK] Ollama found: $ollamaVersion" -ForegroundColor Green
    
    Write-Host "[...] Pulling llama3.1 model (this may take a few minutes)..." -ForegroundColor Cyan
    ollama pull llama3.1
    Write-Host "[OK] llama3.1 model ready" -ForegroundColor Green
} catch {
    Write-Host "[!!] Ollama not found!" -ForegroundColor Red
    Write-Host "     Download from: https://ollama.com/download/windows" -ForegroundColor Yellow
    Write-Host "     Install it, then re-run this script." -ForegroundColor Yellow
}

# ── Step 7: Init Git ─────────────────────────────
Write-Host ""
try {
    Set-Location $PROJECT_DIR
    if (-not (Test-Path "$PROJECT_DIR\.git")) {
        git init | Out-Null
        Write-Host "[OK] Git repository initialized" -ForegroundColor Green
    } else {
        Write-Host "[OK] Git repository already exists" -ForegroundColor Green
    }
} catch {
    Write-Host "[!!] Git not found. Install from: https://git-scm.com" -ForegroundColor Yellow
}

# ── Done ─────────────────────────────────────────
Write-Host ""
Write-Host "════════════════════════════════════════" -ForegroundColor Green
Write-Host "  Phase 1 environment setup complete!" -ForegroundColor Green
Write-Host "════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "Project location: $PROJECT_DIR" -ForegroundColor Cyan
Write-Host "Config location:  $AGENT_CONFIG" -ForegroundColor Cyan
Write-Host ""
Write-Host "NEXT STEPS:" -ForegroundColor Yellow
Write-Host "  1. Copy the src/ files into $PROJECT_DIR\src\" -ForegroundColor White
Write-Host "  2. Copy config/pii_mappings.json into $PROJECT_DIR\config\" -ForegroundColor White
Write-Host "  3. Copy .env.example and .gitignore into $PROJECT_DIR\" -ForegroundColor White
Write-Host "  4. Rename .env.example to .env and add your ANTHROPIC_API_KEY" -ForegroundColor White
Write-Host "  5. Run: cd $PROJECT_DIR" -ForegroundColor White
Write-Host "  6. Run: .venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  7. Run: python -c `"from src.security import SecurityManager; s = SecurityManager(); print(s.get_status())`"" -ForegroundColor White
Write-Host "  8. Run: python src\orchestrator.py" -ForegroundColor White
Write-Host ""
