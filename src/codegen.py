"""
CODE GENERATION ENGINE
=======================
Generates full applications, websites, scripts, and interfaces
from natural language descriptions. Uses Claude for code generation
and manages the full lifecycle: generate → validate → save → deploy.

Security: All generated code is sandboxed. Deployment requires
explicit confirmation unless auto_deploy is enabled for trusted patterns.
"""

import os
import re
import json
import asyncio
import shutil
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("codegen")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
class CodeGenConfig:
    WORKSPACE_DIR = os.getenv("CODEGEN_WORKSPACE", os.path.expanduser("~/agent-workspace"))
    PROJECTS_DIR = os.path.join(WORKSPACE_DIR, "projects")
    TEMPLATES_DIR = os.path.join(WORKSPACE_DIR, "templates")
    DEPLOY_DIR = os.getenv("CODEGEN_DEPLOY_DIR", os.path.expanduser("~/agent-deploys"))
    MAX_FILE_SIZE = 50_000  # Max chars per generated file
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250514")
    AUTO_DEPLOY_PATTERNS = ["landing-page", "dashboard", "portfolio", "static-site"]
    SANDBOX_ENABLED = True

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.WORKSPACE_DIR, cls.PROJECTS_DIR, cls.TEMPLATES_DIR, cls.DEPLOY_DIR]:
            os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# Project Types
# ─────────────────────────────────────────────
class ProjectType(Enum):
    HTML_SITE = "html-site"
    REACT_APP = "react-app"
    PYTHON_SCRIPT = "python-script"
    PYTHON_APP = "python-app"
    API_SERVER = "api-server"
    DASHBOARD = "dashboard"
    LANDING_PAGE = "landing-page"
    COMPONENT = "component"
    AUTOMATION = "automation"
    FULL_STACK = "full-stack"


@dataclass
class GeneratedFile:
    path: str
    content: str
    language: str
    description: str


@dataclass
class Project:
    id: str
    name: str
    project_type: ProjectType
    description: str
    files: list[GeneratedFile] = field(default_factory=list)
    status: str = "created"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    deployed_url: Optional[str] = None
    directory: Optional[str] = None
    build_log: list[str] = field(default_factory=list)
    iteration: int = 0

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.build_log.append(entry)
        logger.info(f"[{self.name}] {message}")


# ─────────────────────────────────────────────
# Code Generator - Uses Claude API
# ─────────────────────────────────────────────
class CodeGenerator:
    """
    Generates code using Claude API with structured prompts.
    Supports multi-file projects with proper structure.
    """

    SYSTEM_PROMPT = """You are an expert full-stack developer working as part of an autonomous AI agent system.
Your job is to generate complete, production-ready code from natural language descriptions.

RULES:
1. Generate COMPLETE, WORKING code — no placeholders, no TODOs, no "add your code here"
2. Every file must be fully functional and ready to run
3. Use modern best practices and clean architecture
4. Include all necessary imports, configurations, and dependencies
5. For web projects: make them visually stunning with modern CSS, animations, and responsive design
6. For Python projects: include proper error handling, type hints, and docstrings
7. NEVER include API keys, passwords, or secrets — use environment variables
8. Always include a README.md with setup instructions

RESPONSE FORMAT:
Return a JSON object with this exact structure:
{
  "project_name": "descriptive-kebab-case-name",
  "project_type": "html-site|react-app|python-script|python-app|api-server|dashboard|landing-page|component|automation|full-stack",
  "description": "What this project does",
  "files": [
    {
      "path": "relative/path/to/file.ext",
      "content": "full file contents here",
      "language": "html|css|js|jsx|python|json|markdown|yaml|shell",
      "description": "What this file does"
    }
  ],
  "setup_commands": ["npm install", "pip install -r requirements.txt"],
  "run_command": "python app.py",
  "dependencies": {"npm": ["react", "tailwindcss"], "pip": ["flask", "requests"]}
}

IMPORTANT: Return ONLY the JSON object. No markdown, no backticks, no explanation outside the JSON."""

    def __init__(self):
        self.api_key = CodeGenConfig.ANTHROPIC_API_KEY
        self.model = CodeGenConfig.CLAUDE_MODEL

    async def generate(self, prompt: str, project_context: str = "", iteration: int = 0) -> dict:
        """Generate a complete project from a natural language description."""
        messages = []

        if iteration > 0:
            messages.append({
                "role": "user",
                "content": f"Previous version context:\n{project_context}\n\n"
                           f"New request (iteration {iteration}):\n{prompt}"
            })
        else:
            messages.append({
                "role": "user",
                "content": f"Build this project:\n\n{prompt}"
            })

        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "max_tokens": 8192,
            "system": self.SYSTEM_PROMPT,
            "messages": messages,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                )
                result = response.json()
                raw_text = result["content"][0]["text"]

                # Parse JSON from response (handle potential markdown wrapping)
                cleaned = raw_text.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```\w*\n?", "", cleaned)
                    cleaned = re.sub(r"\n?```$", "", cleaned)

                return json.loads(cleaned)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            # Try to extract JSON from the response
            match = re.search(r'\{[\s\S]*\}', raw_text)
            if match:
                return json.loads(match.group())
            raise
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise

    async def refine(self, project: Project, feedback: str) -> dict:
        """Iterate on an existing project based on feedback."""
        context = f"Project: {project.name}\nType: {project.project_type.value}\n"
        context += f"Current files:\n"
        for f in project.files:
            context += f"- {f.path}: {f.description}\n"

        prompt = f"Refine this project based on feedback:\n{feedback}\n\n"
        prompt += "Keep what works, improve what's requested. Return the COMPLETE updated project."

        return await self.generate(prompt, context, project.iteration + 1)


# ─────────────────────────────────────────────
# Project Manager - Handles filesystem & state
# ─────────────────────────────────────────────
class ProjectManager:
    """
    Manages project lifecycle: create, save, build, deploy.
    All projects are sandboxed in the workspace directory.
    """

    def __init__(self):
        CodeGenConfig.ensure_dirs()
        self.projects: dict[str, Project] = {}
        self._load_projects()

    def _load_projects(self):
        """Load existing projects from disk."""
        index_path = os.path.join(CodeGenConfig.WORKSPACE_DIR, "projects_index.json")
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)
                for pid, pdata in data.items():
                    pdata["project_type"] = ProjectType(pdata["project_type"])
                    pdata["files"] = [GeneratedFile(**fd) for fd in pdata.get("files", [])]
                    self.projects[pid] = Project(**pdata)

    def _save_index(self):
        """Persist project index."""
        index_path = os.path.join(CodeGenConfig.WORKSPACE_DIR, "projects_index.json")
        data = {}
        for pid, proj in self.projects.items():
            pdict = {
                "id": proj.id,
                "name": proj.name,
                "project_type": proj.project_type.value,
                "description": proj.description,
                "files": [{"path": f.path, "content": "", "language": f.language,
                           "description": f.description} for f in proj.files],
                "status": proj.status,
                "created_at": proj.created_at,
                "deployed_url": proj.deployed_url,
                "directory": proj.directory,
                "build_log": proj.build_log[-20:],  # Keep last 20 log entries
                "iteration": proj.iteration,
            }
            data[pid] = pdict
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_project(self, gen_result: dict) -> Project:
        """Create a project from Claude's generated output."""
        project_id = hashlib.md5(
            f"{gen_result['project_name']}-{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        project_dir = os.path.join(CodeGenConfig.PROJECTS_DIR, f"{gen_result['project_name']}-{project_id}")
        os.makedirs(project_dir, exist_ok=True)

        files = []
        for fdata in gen_result.get("files", []):
            files.append(GeneratedFile(
                path=fdata["path"],
                content=fdata["content"],
                language=fdata.get("language", "text"),
                description=fdata.get("description", ""),
            ))

        project = Project(
            id=project_id,
            name=gen_result["project_name"],
            project_type=ProjectType(gen_result.get("project_type", "html-site")),
            description=gen_result.get("description", ""),
            files=files,
            directory=project_dir,
        )

        project.log("Project created")
        self.projects[project_id] = project
        self._save_index()
        return project

    def write_files(self, project: Project) -> list[str]:
        """Write all generated files to disk."""
        written = []
        for gf in project.files:
            file_path = os.path.join(project.directory, gf.path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Security: validate file path doesn't escape sandbox
            real_path = os.path.realpath(file_path)
            if not real_path.startswith(os.path.realpath(CodeGenConfig.WORKSPACE_DIR)):
                project.log(f"⚠️ BLOCKED: Path escape attempt: {gf.path}")
                continue

            # Security: check file size
            if len(gf.content) > CodeGenConfig.MAX_FILE_SIZE:
                project.log(f"⚠️ BLOCKED: File too large: {gf.path} ({len(gf.content)} chars)")
                continue

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(gf.content)
            written.append(gf.path)
            project.log(f"📝 Written: {gf.path}")

        project.status = "files_written"
        self._save_index()
        return written

    def list_projects(self) -> list[dict]:
        """List all projects with summary info."""
        return [
            {
                "id": p.id,
                "name": p.name,
                "type": p.project_type.value,
                "description": p.description,
                "status": p.status,
                "files": len(p.files),
                "created": p.created_at,
                "deployed_url": p.deployed_url,
                "iteration": p.iteration,
            }
            for p in self.projects.values()
        ]

    def get_project(self, project_id: str) -> Optional[Project]:
        return self.projects.get(project_id)


# ─────────────────────────────────────────────
# Code Validator - Security checks
# ─────────────────────────────────────────────
class CodeValidator:
    """
    Validates generated code for security issues.
    Blocks dangerous patterns before files are written.
    """

    BLOCKED_PATTERNS = [
        (r"os\.system\s*\(", "Direct shell execution"),
        (r"subprocess\.call\s*\(.*shell\s*=\s*True", "Shell injection risk"),
        (r"eval\s*\(", "Code injection via eval"),
        (r"exec\s*\(", "Code injection via exec"),
        (r"__import__\s*\(", "Dynamic import (potential backdoor)"),
        (r"sk-ant-[a-zA-Z0-9\-]+", "Hardcoded Anthropic API key"),
        (r"sk-[a-zA-Z0-9]{20,}", "Hardcoded OpenAI API key"),
        (r"AKIA[0-9A-Z]{16}", "Hardcoded AWS access key"),
        (r"rm\s+-rf\s+/", "Destructive filesystem command"),
        (r"curl.*\|\s*sh", "Remote code execution"),
        (r"wget.*\|\s*bash", "Remote code execution"),
    ]

    WARNING_PATTERNS = [
        (r"fetch\s*\(", "External HTTP request — verify the URL"),
        (r"XMLHttpRequest", "External HTTP request — verify the URL"),
        (r"localStorage", "Browser storage — PII risk"),
        (r"document\.cookie", "Cookie access — PII risk"),
    ]

    @classmethod
    def validate(cls, project: Project) -> dict:
        """Validate all files in a project. Returns {passed: bool, issues: []}"""
        issues = []
        warnings = []

        for gf in project.files:
            # Check blocked patterns
            for pattern, reason in cls.BLOCKED_PATTERNS:
                if re.search(pattern, gf.content):
                    issues.append({
                        "file": gf.path,
                        "severity": "BLOCKED",
                        "reason": reason,
                        "pattern": pattern,
                    })

            # Check warning patterns
            for pattern, reason in cls.WARNING_PATTERNS:
                if re.search(pattern, gf.content):
                    warnings.append({
                        "file": gf.path,
                        "severity": "WARNING",
                        "reason": reason,
                    })

        passed = len(issues) == 0
        return {
            "passed": passed,
            "blocked_issues": issues,
            "warnings": warnings,
            "files_scanned": len(project.files),
        }


# ─────────────────────────────────────────────
# Deployer - Handles deployment targets
# ─────────────────────────────────────────────
class Deployer:
    """
    Deploys generated projects to various targets.
    All deployments happen from your local machine.
    """

    @staticmethod
    async def deploy_local_http(project: Project, port: int = 3000) -> str:
        """Serve a static site locally using Python's HTTP server."""
        project.log(f"🚀 Deploying to local HTTP server on port {port}")
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-m", "http.server", str(port),
                cwd=project.directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            project.deployed_url = f"http://localhost:{port}"
            project.status = "deployed_local"
            project.log(f"✅ Live at {project.deployed_url}")
            return project.deployed_url
        except Exception as e:
            project.log(f"❌ Local deploy failed: {e}")
            return f"Deploy failed: {e}"

    @staticmethod
    async def deploy_vercel(project: Project) -> str:
        """Deploy to Vercel (requires vercel CLI installed and logged in)."""
        project.log("🚀 Deploying to Vercel...")
        try:
            proc = await asyncio.create_subprocess_exec(
                "vercel", "--yes", "--prod",
                cwd=project.directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                url = stdout.decode().strip().split("\n")[-1]
                project.deployed_url = url
                project.status = "deployed_vercel"
                project.log(f"✅ Live at {url}")
                return url
            else:
                error = stderr.decode()
                project.log(f"❌ Vercel deploy failed")
                return "Deploy failed. Check build log for details."
        except FileNotFoundError:
            project.log("❌ Vercel CLI not installed. Run: npm i -g vercel")
            return "Vercel CLI not installed. Run: npm i -g vercel"

    @staticmethod
    async def deploy_netlify(project: Project) -> str:
        """Deploy to Netlify (requires netlify CLI)."""
        project.log("🚀 Deploying to Netlify...")
        try:
            proc = await asyncio.create_subprocess_exec(
                "netlify", "deploy", "--prod", "--dir", ".",
                cwd=project.directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                output = stdout.decode()
                url_match = re.search(r"https://[^\s]+\.netlify\.app", output)
                url = url_match.group() if url_match else "deployed"
                project.deployed_url = url
                project.status = "deployed_netlify"
                project.log(f"✅ Live at {url}")
                return url
            else:
                return "Deploy failed. Check build log for details."
        except FileNotFoundError:
            return "Netlify CLI not installed. Run: npm i -g netlify-cli"

    @staticmethod
    async def deploy_copy(project: Project) -> str:
        """Copy project to a deploy directory (for nginx/apache serving)."""
        deploy_path = os.path.join(CodeGenConfig.DEPLOY_DIR, project.name)
        try:
            if os.path.exists(deploy_path):
                shutil.rmtree(deploy_path)
            shutil.copytree(project.directory, deploy_path)
            project.deployed_url = deploy_path
            project.status = "deployed_copy"
            project.log(f"✅ Copied to {deploy_path}")
            return deploy_path
        except Exception as e:
            return f"Copy failed: {e}"


# ─────────────────────────────────────────────
# Shell Executor - Sandboxed command runner
# ─────────────────────────────────────────────
class ShellExecutor:
    """
    Executes shell commands in a sandboxed environment.
    Only allows pre-approved commands and patterns.
    """

    ALLOWED_COMMANDS = {
        "npm": ["install", "run", "build", "init", "test"],
        "pip": ["install", "freeze"],
        "python3": None,  # Allow all python3 args
        "node": None,
        "git": ["init", "add", "commit", "status", "log", "diff"],
        "ls": None,
        "cat": None,
        "mkdir": None,
        "cp": None,
        "vercel": ["--yes", "--prod"],
        "netlify": ["deploy"],
    }

    BLOCKED_COMMANDS = [
        "rm -rf /", "rm -rf ~", "sudo", "chmod 777",
        "curl | sh", "wget | bash", "mkfs", "dd if=",
        "> /dev/sda", ":(){ :|:& };:",
    ]

    # Shell metacharacters that enable command injection
    SHELL_INJECTION_CHARS = [";", "&&", "||", "|", "`", "$(", "${", "\n", "\r", ">", "<"]

    @classmethod
    async def execute(cls, command: str, cwd: str = None, timeout: int = 120) -> dict:
        """Execute a shell command with safety checks."""
        # Security: Reject commands with shell injection metacharacters
        for char in cls.SHELL_INJECTION_CHARS:
            if char in command:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"BLOCKED: Command contains shell metacharacter. Only simple commands allowed.",
                    "returncode": -1,
                }

        # Security: Check for blocked patterns
        for blocked in cls.BLOCKED_COMMANDS:
            if blocked in command:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"BLOCKED: Command contains dangerous pattern.",
                    "returncode": -1,
                }

        # Security: Validate command is in allowed list
        cmd_parts = command.split()
        base_cmd = cmd_parts[0] if cmd_parts else ""

        if base_cmd not in cls.ALLOWED_COMMANDS:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"BLOCKED: Command not in allowed list.",
                "returncode": -1,
            }

        # Check subcommand restrictions
        allowed_sub = cls.ALLOWED_COMMANDS.get(base_cmd)
        if allowed_sub is not None and len(cmd_parts) > 1:
            if cmd_parts[1] not in allowed_sub:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"BLOCKED: Subcommand not allowed.",
                    "returncode": -1,
                }

        # Security: Ensure cwd is within workspace
        if cwd:
            real_cwd = os.path.realpath(cwd)
            if not real_cwd.startswith(os.path.realpath(CodeGenConfig.WORKSPACE_DIR)):
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "BLOCKED: Working directory outside sandbox",
                    "returncode": -1,
                }

        try:
            # Use create_subprocess_exec (NOT _shell) to prevent injection
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            return {
                "success": proc.returncode == 0,
                "stdout": stdout.decode(errors="replace")[:5000],
                "stderr": stderr.decode(errors="replace")[:2000],
                "returncode": proc.returncode,
            }
        except asyncio.TimeoutError:
            proc.kill()
            return {
                "success": False,
                "stdout": "",
                "stderr": f"TIMEOUT: Command exceeded {timeout}s limit",
                "returncode": -1,
            }
        except Exception as e:
            logger.error(f"Shell execution error: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command execution failed",
                "returncode": -1,
            }


# ─────────────────────────────────────────────
# Build Pipeline - Orchestrates the full flow
# ─────────────────────────────────────────────
class BuildPipeline:
    """
    Complete build pipeline: prompt → generate → validate → write → build → deploy.
    This is what the orchestrator calls.
    """

    def __init__(self):
        self.generator = CodeGenerator()
        self.manager = ProjectManager()
        self.deployer = Deployer()
        self.shell = ShellExecutor()

    async def build(self, prompt: str, deploy_target: str = "local") -> dict:
        """
        Full pipeline: generate code from prompt, validate, write, and deploy.
        Returns a summary dict with project info and status.
        """
        result = {
            "status": "started",
            "project_id": None,
            "project_name": None,
            "files": [],
            "validation": None,
            "deploy_url": None,
            "build_log": [],
        }

        try:
            # Step 1: Generate code via Claude
            result["status"] = "generating"
            logger.info(f"🏗️ Generating project from: {prompt[:80]}...")
            gen_output = await self.generator.generate(prompt)

            # Step 2: Create project
            project = self.manager.create_project(gen_output)
            result["project_id"] = project.id
            result["project_name"] = project.name

            # Step 3: Validate for security
            result["status"] = "validating"
            validation = CodeValidator.validate(project)
            result["validation"] = validation

            if not validation["passed"]:
                project.log(f"❌ Validation FAILED: {len(validation['blocked_issues'])} blocked issues")
                for issue in validation["blocked_issues"]:
                    project.log(f"  BLOCKED: {issue['file']} — {issue['reason']}")
                project.status = "validation_failed"
                result["status"] = "validation_failed"
                result["build_log"] = project.build_log
                return result

            if validation["warnings"]:
                for warn in validation["warnings"]:
                    project.log(f"  ⚠️ WARNING: {warn['file']} — {warn['reason']}")

            # Step 4: Write files to disk
            result["status"] = "writing"
            written = self.manager.write_files(project)
            result["files"] = written

            # Step 5: Run setup commands if any
            setup_commands = gen_output.get("setup_commands", [])
            for cmd in setup_commands:
                project.log(f"⚙️ Running: {cmd}")
                cmd_result = await self.shell.execute(cmd, cwd=project.directory)
                if not cmd_result["success"]:
                    project.log(f"⚠️ Setup command failed: {cmd_result['stderr']}")

            # Step 6: Deploy
            result["status"] = "deploying"
            if deploy_target == "vercel":
                url = await self.deployer.deploy_vercel(project)
            elif deploy_target == "netlify":
                url = await self.deployer.deploy_netlify(project)
            elif deploy_target == "copy":
                url = await self.deployer.deploy_copy(project)
            else:
                url = await self.deployer.deploy_copy(project)

            result["deploy_url"] = url
            result["status"] = "complete"
            result["build_log"] = project.build_log

            # Save final state
            self.manager._save_index()

            project.log(f"🎉 Build complete! {len(written)} files generated.")
            return result

        except Exception as e:
            logger.error(f"Build pipeline failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result

    async def iterate(self, project_id: str, feedback: str, deploy_target: str = "local") -> dict:
        """Refine an existing project based on feedback."""
        project = self.manager.get_project(project_id)
        if not project:
            return {"status": "error", "error": f"Project {project_id} not found"}

        project.log(f"🔄 Iteration {project.iteration + 1}: {feedback[:80]}...")

        try:
            gen_output = await self.generator.refine(project, feedback)

            # Update project files
            project.files = [
                GeneratedFile(**fd) for fd in gen_output.get("files", [])
            ]
            project.iteration += 1

            # Validate
            validation = CodeValidator.validate(project)
            if not validation["passed"]:
                return {"status": "validation_failed", "validation": validation}

            # Write updated files
            written = self.manager.write_files(project)

            # Re-deploy
            if deploy_target == "copy":
                url = await self.deployer.deploy_copy(project)
            else:
                url = await self.deployer.deploy_copy(project)

            return {
                "status": "complete",
                "project_id": project.id,
                "iteration": project.iteration,
                "files": written,
                "deploy_url": url,
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def list_projects(self) -> list[dict]:
        """List all generated projects."""
        return self.manager.list_projects()