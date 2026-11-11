"""Traveller session IDs (LangGraph thread_id) shared by the CLI and Streamlit.

- A traveller who gives a name gets  "user_<name>", e.g. "user_priya".
- Otherwise a random guest ID is generated, e.g. "guest_3f9a2c71b8e4",
  and reused so the conversation context is kept between runs.
"""

import re
import uuid
from pathlib import Path

THREAD_ID_PATTERN = re.compile(r"^(user|guest)_[a-z0-9_]{1,40}$")

# CLI only: remembers the guest ID between runs (git-ignored)
CLI_SESSION_FILE = Path(__file__).resolve().parent / ".trippilot_session"


def thread_id_from_name(name: str | None) -> str | None:
    """'Priya Sharma' -> 'user_priya_sharma'. Returns None if the name is empty."""
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")[:40]
    return f"user_{slug}" if slug else None


def new_guest_id() -> str:
    return f"guest_{uuid.uuid4().hex[:12]}"


def is_valid_thread_id(value: str | None) -> bool:
    return bool(value) and bool(THREAD_ID_PATTERN.match(value))


def name_from_thread_id(thread_id: str | None) -> str:
    """'user_priya_sharma' -> 'priya sharma'; guest IDs -> ''."""
    if thread_id and thread_id.startswith("user_"):
        return thread_id[len("user_"):].replace("_", " ")
    return ""


def cli_guest_id() -> str:
    """Guest ID for the command line, created once and reused on later runs."""
    try:
        saved = CLI_SESSION_FILE.read_text(encoding="utf-8").strip()
        if is_valid_thread_id(saved):
            return saved
    except OSError:
        pass
    guest = new_guest_id()
    try:
        CLI_SESSION_FILE.write_text(guest, encoding="utf-8")
    except OSError:
        pass  # still works, just won't be remembered next time
    return guest
