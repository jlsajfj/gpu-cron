"""Incremental automaton over the 5-field cron grammar.

The decoder masks every logit not in ``allowed()``, so a decoded string is well-formed
cron by construction — there is no repair pass or regex fixup anywhere. A field is
``term (',' term)*`` with ``term := '*' ['/' step] | value ['-' value] ['/' step]``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

GRAMMAR_PATH = Path(__file__).resolve().parent.parent / "grammar" / "cron-grammar.json"

EOS = "\x00"
ALPHABET_CHARS = "0123456789*,-/"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    min: int
    max: int


@dataclass(frozen=True)
class Grammar:
    fields: tuple[FieldSpec, ...]
    alphabet: str
    max_length: int

    @staticmethod
    def load(path: Path = GRAMMAR_PATH) -> "Grammar":
        raw = json.loads(path.read_text())
        fields = tuple(
            FieldSpec(f["name"], f["min"], f["max"])
            for f in sorted(raw["fields"], key=lambda f: f["index"])
        )
        return Grammar(fields, raw["alphabet"], raw["maxLength"])


@lru_cache(maxsize=4096)
def _leading_ok(digit: str, lo: int, hi: int) -> bool:
    return any(str(v).startswith(digit) for v in range(lo, hi + 1))


def _step_ok(value: int, spec: FieldSpec) -> bool:
    return 1 <= value <= spec.max


@lru_cache(maxsize=65536)
def _expand(term: str, spec: FieldSpec) -> set[int] | None:
    """Values a *complete* term covers, or None if the term is not complete."""
    if term == "*":
        return set(range(spec.min, spec.max + 1))

    def stepped(start: int, end: int, step: int) -> set[int]:
        return set(range(start, end + 1, step))

    m = re.fullmatch(r"\*/(\d+)", term)
    if m and _step_ok(int(m.group(1)), spec):
        return stepped(spec.min, spec.max, int(m.group(1)))
    m = re.fullmatch(r"(\d+)", term)
    if m and spec.min <= int(m.group(1)) <= spec.max:
        return {int(m.group(1))}
    m = re.fullmatch(r"(\d+)/(\d+)", term)
    if m and _step_ok(int(m.group(2)), spec) and spec.min <= int(m.group(1)) <= spec.max:
        return stepped(int(m.group(1)), spec.max, int(m.group(2)))
    m = re.fullmatch(r"(\d+)-(\d+)", term)
    if m and spec.min <= int(m.group(1)) <= int(m.group(2)) <= spec.max:
        return set(range(int(m.group(1)), int(m.group(2)) + 1))
    m = re.fullmatch(r"(\d+)-(\d+)/(\d+)", term)
    if (
        m
        and _step_ok(int(m.group(3)), spec)
        and spec.min <= int(m.group(1)) <= int(m.group(2)) <= spec.max
    ):
        return stepped(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


@lru_cache(maxsize=65536)
def _start_ok(term: str, threshold: int | None, spec: FieldSpec) -> bool:
    """Could this (possibly still-growing) term begin above everything already covered?

    Lists must be strictly ascending and non-overlapping — narrower than cron-parser, which
    allows `*/2,5`, but checkable one character at a time. A term's values all lie in
    [start, field max], so a start above the running maximum is enough for the whole term.
    """
    if threshold is None:
        return True
    if term and term[0] == "*":
        return False  # a bare star covers the field, so it overlaps anything before it
    candidates = range(spec.min, spec.max + 1)
    if not term:
        return any(v > threshold for v in candidates)
    m = re.match(r"\d+", term)
    if not m:
        return True
    digits = m.group()
    if len(digits) == 2 or m.end() < len(term):
        return int(digits) > threshold  # the start value is already fixed
    return any(str(v).startswith(digits) and v > threshold for v in candidates)


@lru_cache(maxsize=65536)
def _list_ok(s: str, spec: FieldSpec, *, trailing_complete: bool) -> bool:
    """Whether `s`'s comma terms are ascending and non-overlapping.

    `trailing_complete=False` asks only that the last term's *start* is placeable (it may
    still grow); `True` requires its whole expansion to clear the running maximum.
    """
    parts = s.split(",")
    threshold: int | None = None
    for term in parts[:-1]:
        values = _expand(term, spec)
        if values is None:
            return False
        if threshold is not None and min(values) <= threshold:
            return False
        threshold = max(values)

    trail = parts[-1]
    if not trailing_complete:
        return _start_ok(trail, threshold, spec)
    values = _expand(trail, spec)
    if values is None:
        return False
    return threshold is None or min(values) > threshold


@lru_cache(maxsize=65536)
def field_completion_len(text: str, spec: FieldSpec, max_extra: int = 3) -> int:
    """Fewest characters to append to `text` to reach a complete valid field.

    Breadth-first with an early exit: the usual answers (0 or 1) cost a handful of memoised
    `field_state` calls, and only an unreachable state pays for the full sweep.
    """
    if field_state(text, spec)[1]:
        return 0
    frontier = [""]
    for depth in range(1, max_extra + 1):
        nxt = [prefix + c for prefix in frontier for c in ALPHABET_CHARS]
        for ext in nxt:
            if field_state(text + ext, spec)[1]:
                return depth
        frontier = nxt
    return max_extra + 1


@lru_cache(maxsize=262144)
def field_state(s: str, spec: FieldSpec) -> tuple[bool, bool]:
    """Walk `s` as a partial field; returns (could still grow, is already complete)."""
    state = "TERM_START"
    lo = 0
    one = 0  # the single value held in V1 / R_HI1 / ST1
    i = 0
    n = len(s)

    while i < n:
        c = s[i]
        if state == "TERM_START":
            if c == "*":
                state = "AFTER_STAR"
            elif c.isdigit() and _leading_ok(c, spec.min, spec.max):
                one = int(c)
                state = "V1"
            else:
                return False, False
        elif state == "AFTER_STAR":
            if c == "/":
                state = "BEFORE_STEP"
            elif c == ",":
                state = "TERM_START"
            else:
                return False, False
        elif state == "V1":
            if c.isdigit():
                v = one * 10 + int(c)
                if not (spec.min <= v <= spec.max):
                    return False, False
                one = v
                state = "V2"
            elif c in "-/,":
                if not (spec.min <= one <= spec.max):
                    return False, False
                if c == "-":
                    if one >= spec.max:
                        return False, False
                    lo = one
                    state = "R_LO"
                elif c == "/":
                    state = "BEFORE_STEP"
                else:
                    state = "TERM_START"
            else:
                return False, False
        elif state == "V2":
            if c == "-":
                if not (spec.min <= one < spec.max):
                    return False, False
                lo = one
                state = "R_LO"
            elif c == "/":
                state = "BEFORE_STEP"
            elif c == ",":
                state = "TERM_START"
            else:
                return False, False
        elif state == "R_LO":
            if c.isdigit() and _leading_ok(c, lo + 1, spec.max):
                one = int(c)
                state = "R_HI1"
            else:
                return False, False
        elif state == "R_HI1":
            if c.isdigit():
                v = one * 10 + int(c)
                if not (lo < v <= spec.max):
                    return False, False
                one = v
                state = "R_HI2"
            elif c in "/,":
                if one <= lo:
                    return False, False
                state = "BEFORE_STEP" if c == "/" else "TERM_START"
            else:
                return False, False
        elif state == "R_HI2":
            if c in "/,":
                state = "BEFORE_STEP" if c == "/" else "TERM_START"
            else:
                return False, False
        elif state == "BEFORE_STEP":
            if c.isdigit() and _step_ok(int(c), spec):
                one = int(c)
                state = "ST1"
            else:
                return False, False
        elif state == "ST1":
            if c.isdigit():
                if not _step_ok(one * 10 + int(c), spec):
                    return False, False
                one = one * 10 + int(c)
                state = "ST2"
            elif c == ",":
                state = "TERM_START"
            else:
                return False, False
        elif state == "ST2":
            if c == ",":
                state = "TERM_START"
            else:
                return False, False
        i += 1

    if state == "TERM_START":
        syntactic = (len(s) > 0, False)
    elif state == "V1":
        syntactic = (True, spec.min <= one <= spec.max)
    elif state == "R_LO":
        syntactic = (True, False)
    elif state == "R_HI1":
        syntactic = (True, one > lo)
    elif state == "BEFORE_STEP":
        syntactic = (True, False)
    else:  # AFTER_STAR / V2 / R_HI2 / ST1 / ST2
        syntactic = (True, True)
    if not syntactic[0]:
        return False, False
    return (
        _list_ok(s, spec, trailing_complete=False),
        syntactic[1] and _list_ok(s, spec, trailing_complete=True),
    )


class CronAutomaton:
    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self._allowed: dict[tuple[int, str, int], frozenset[str]] = {}

    def start(self) -> tuple[int, str, int]:
        return (0, "", 0)

    def is_terminal(self, state: tuple[int, str, int]) -> bool:
        return state[0] >= len(self.grammar.fields)

    def completion_len(self, state: tuple[int, str]) -> int:
        """Length of the shortest completion that actually exists from `state`.

        A field left hanging on a comma needs two characters, not one: the next term has to
        open *and* close, and the list rule can force two digits (`5/7,27`). Gating
        `allowed()` on this keeps `completion_len(state) <= remaining budget` true at every
        reachable state, so the length cap can never strand the decoder.
        """
        fi, text = state
        last = len(self.grammar.fields) - 1
        return field_completion_len(text, self.grammar.fields[fi]) + 2 * (last - fi)

    def allowed(self, state: tuple[int, str, int]) -> frozenset[str]:
        if self.is_terminal(state):
            return frozenset()
        cached = self._allowed.get(state)
        if cached is not None:
            return cached
        fi, text, total = state
        last = len(self.grammar.fields) - 1
        spec = self.grammar.fields[fi]
        budget = self.grammar.max_length - total
        out = set()
        for c in self.grammar.alphabet:
            if c == " ":
                continue
            nxt = (fi, text + c)
            if field_state(nxt[1], spec)[0] and 1 + self.completion_len(nxt) <= budget:
                out.add(c)
        if field_state(text, spec)[1] and self.completion_len((fi, text)) <= budget:
            out.add(" " if fi < last else EOS)
        result = frozenset(out)
        self._allowed[state] = result
        return result

    def advance(self, state: tuple[int, str, int], ch: str) -> tuple[int, str, int] | None:
        if self.is_terminal(state) or ch not in self.allowed(state):
            return None
        fi, text, total = state
        if ch == EOS or ch == " ":
            return (fi + 1, "", total + 1)
        return (fi, text + ch, total + 1)

    def is_complete(self, state: tuple[int, str, int]) -> bool:
        if self.is_terminal(state):
            return True
        fi, text, _total = state
        if fi != len(self.grammar.fields) - 1:
            return False
        return field_state(text, self.grammar.fields[fi])[1]

    def walk(self, text: str) -> tuple[int, str, int] | None:
        state = self.start()
        for ch in text:
            state = self.advance(state, ch)
            if state is None:
                return None
        return state


@lru_cache(maxsize=1)
def default_grammar() -> Grammar:
    return Grammar.load()


@lru_cache(maxsize=1)
def default_automaton() -> CronAutomaton:
    return CronAutomaton(default_grammar())


def is_well_formed(text: str, automaton: CronAutomaton | None = None) -> bool:
    auto = automaton or default_automaton()
    state = auto.walk(text)
    return state is not None and auto.is_complete(state)
