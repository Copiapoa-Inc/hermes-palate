"""Read a Git checkout revision without importing the full Hermes CLI."""

from __future__ import annotations

from pathlib import Path


def _read_packed_ref(common_dir: Path, ref: str) -> str | None:
    """Look up one ref in ``.git/packed-refs`` without spawning Git."""
    try:
        text = (common_dir / "packed-refs").read_text(
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None
    for line in text.splitlines():
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].strip() == ref:
            return parts[0].strip()
    return None


def read_git_revision_fingerprint(repo_root: Path) -> str | None:
    """Return a cheap checkout fingerprint without spawning Git."""
    git_dir = repo_root / ".git"
    try:
        if git_dir.is_file():
            for line in git_dir.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines():
                key, _, value = line.partition(":")
                if key.strip() == "gitdir" and value.strip():
                    git_dir = (repo_root / value.strip()).resolve()
                    break

        common_dir = git_dir
        commondir_file = git_dir / "commondir"
        if commondir_file.exists():
            try:
                relative = commondir_file.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).strip()
                if relative:
                    common_dir = (git_dir / relative).resolve()
            except OSError:
                pass

        head = (git_dir / "HEAD").read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            for candidate in (git_dir, common_dir):
                ref_file = candidate / ref
                if ref_file.exists():
                    sha = ref_file.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).strip()
                    return f"git:{ref}:{sha}"
            packed_sha = _read_packed_ref(common_dir, ref)
            if packed_sha:
                return f"git:{ref}:{packed_sha}"
            return f"git:{ref}:unresolved"
        return f"git:HEAD:{head}"
    except OSError:
        return None
