from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Token:
    text: str
    start: float
    end: float


@dataclass
class Sentence:
    id: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Sentence":
        return cls(id=int(d["id"]), start=float(d["start"]), end=float(d["end"]), text=d["text"])
