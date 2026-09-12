"""Thin client for data/cron-service.mjs over one long-lived subprocess.

The eval never reimplements fire-time enumeration: cron-parser stays the single definition
of what an expression means.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

SERVICE = Path(__file__).resolve().parent.parent / "data" / "cron-service.mjs"


class CronService:
    def __init__(self, node: str = "node"):
        self.proc = subprocess.Popen(
            [node, str(SERVICE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def request(self, payload: dict) -> dict:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(f"{json.dumps(payload)}\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("cron-service died")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(f"cron-service: {response['error']}")
        return response

    def validate(self, crons: list[str]) -> list[dict]:
        return self.request({"op": "validate", "crons": crons})["results"]

    def fires(self, cron: str, n: int = 5, from_: str = "2026-01-01T00:00:00Z"):
        return self.request({"op": "fires", "cron": cron, "n": n, "from": from_})["times"]

    def semantic_equal(self, pairs: list[tuple[str, str]], n: int = 5) -> list[bool]:
        if not pairs:
            return []
        return self.request({"op": "semantic", "pairs": [list(p) for p in pairs], "n": n})["equal"]

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def __enter__(self) -> "CronService":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
