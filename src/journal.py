"""
ABOUT-ME JOURNAL SYSTEM
========================
Manages Mel's long-term memory about the user.

All files in data/about-me/ are loaded on startup and injected into
every Claude API call so Mel always knows who she's talking to.

You (the user) edit these files directly in VS Code or any text editor.
Mel can also append notes to notes.md via voice command.

Folder: data/about-me/
  personal.md    — name, family, lifestyle
  business.md    — company, projects, clients
  priorities.md  — current goals (update weekly)
  preferences.md — how you like things done
  notes.md       — misc (auto-populated by voice commands)
  [any .md / .txt you add]
"""

import os
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("journal")

# Resolve path relative to this file's location (src/ → data/about-me/)
_SRC_DIR = Path(__file__).parent
ABOUT_ME_DIR = _SRC_DIR.parent / "data" / "about-me"


def get_about_me_context() -> str:
    """Module-level convenience wrapper for AboutMe.load_all()."""
    return AboutMe.load_all()
NOTES_FILE = ABOUT_ME_DIR / "notes.md"


class AboutMe:
    """Loads and manages the user's personal context folder."""

    @staticmethod
    def load_all() -> str:
        """Read all .md and .txt files in about-me/, return combined text."""
        if not ABOUT_ME_DIR.exists():
            return ""

        parts = []
        for filepath in sorted(ABOUT_ME_DIR.iterdir()):
            if filepath.suffix in (".md", ".txt") and filepath.is_file():
                try:
                    content = filepath.read_text(encoding="utf-8").strip()
                    if content:
                        parts.append(f"### [{filepath.stem}]\n{content}")
                except Exception as e:
                    logger.warning(f"Could not read {filepath}: {e}")

        return "\n\n".join(parts)

    @staticmethod
    def get_summary(max_chars: int = 3000) -> str:
        """Return about-me context trimmed to max_chars for Claude injection."""
        text = AboutMe.load_all()
        if not text:
            return ""
        if len(text) > max_chars:
            return text[:max_chars] + "\n[...truncated]"
        return text

    @staticmethod
    def append_note(note: str, filename: str = "notes.md") -> bool:
        """Append a timestamped note to a file in about-me/."""
        try:
            ABOUT_ME_DIR.mkdir(parents=True, exist_ok=True)
            # Sanitize filename: only allow a bare filename with no path
            # components, so it can't escape ABOUT_ME_DIR via '..' or an
            # absolute path.
            safe_name = os.path.basename(filename)
            if not safe_name or safe_name in (".", "..") or safe_name != filename:
                logger.error(f"Rejected unsafe note filename: {filename!r}")
                return False
            target = (ABOUT_ME_DIR / safe_name).resolve()
            if target.parent != ABOUT_ME_DIR.resolve():
                logger.error(f"Rejected note filename outside ABOUT_ME_DIR: {filename!r}")
                return False
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            entry = f"\n- [{timestamp}] {note.strip()}"
            with open(target, "a", encoding="utf-8") as f:
                f.write(entry)
            logger.info(f"Note appended to {filename}: {note[:80]}")
            return True
        except Exception as e:
            logger.error(f"Could not append note: {e}")
            return False

    @staticmethod
    def get_recent_notes(days: int = 7) -> str:
        """Return notes from the last N days."""
        if not NOTES_FILE.exists():
            return ""
        try:
            lines = NOTES_FILE.read_text(encoding="utf-8").splitlines()
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(days=days)
            recent = []
            for line in lines:
                # Lines look like: - [2026-04-03 14:30] note text
                import re
                m = re.search(r'\[(\d{4}-\d{2}-\d{2})', line)
                if m:
                    try:
                        ts = datetime.strptime(m.group(1), "%Y-%m-%d")
                        if ts >= cutoff:
                            recent.append(line)
                    except ValueError:
                        continue
            return "\n".join(recent)
        except Exception as e:
            logger.error(f"Could not read notes: {e}")
            return ""

    @staticmethod
    def list_files() -> list:
        """List all files in the about-me folder."""
        if not ABOUT_ME_DIR.exists():
            return []
        return [
            {"name": f.name, "size": f.stat().st_size, "modified": f.stat().st_mtime}
            for f in sorted(ABOUT_ME_DIR.iterdir())
            if f.is_file() and f.suffix in (".md", ".txt")
        ]
