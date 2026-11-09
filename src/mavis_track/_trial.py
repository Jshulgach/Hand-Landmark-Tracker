"""Portable trial names and auditable recording locations."""
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def trial_slug(name):
    if not str(name).strip():
        raise ValueError("Enter a trial name")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip()).strip("._")[:80]
    if not slug:
        raise ValueError("Trial name needs letters or numbers")
    if slug.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                                      *(f"LPT{i}" for i in range(1, 10))}:
        slug = "trial_" + slug
    return slug


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def trial_directory(parent, name):
    return Path(parent).expanduser().resolve() / f"{trial_slug(name)}_{utc_stamp()}_{uuid4().hex[:8]}"
