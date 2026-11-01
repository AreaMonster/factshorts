"""Per-job working folder with atomic JSON writes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


class WorkDir:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def subdir(self, name: str) -> Path:
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self, name: str) -> bool:
        return (self.root / name).is_file()

    def write_model(self, name: str, model: BaseModel) -> Path:
        return self.write_text(name, model.model_dump_json(indent=2))

    def read_model(self, name: str, cls: type[M]) -> M:
        return cls.model_validate_json((self.root / name).read_text(encoding="utf-8"))

    def write_json(self, name: str, data: object) -> Path:
        return self.write_text(name, json.dumps(data, indent=2, ensure_ascii=False, default=str))

    def write_text(self, name: str, text: str) -> Path:
        target = self.path(name)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
        return target
