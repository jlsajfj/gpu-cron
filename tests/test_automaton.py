"""The automaton accepts the corpus, rejects near misses, and never dead-ends on a walk."""

from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from cron_automaton import EOS, default_automaton, is_well_formed  # noqa: E402

from cron_semantics import CronService  # noqa: E402


class TestAutomaton(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.auto = default_automaton()
        cls.grammar = cls.auto.grammar

    def test_matches_conformance_fixture(self) -> None:
        """Replay the frozen fixture through Python as well as the browser.

        The fixture is *generated* from this automaton, so without a Python-side replay
        editing the automaton and forgetting `make conformance` leaves the TypeScript suite
        green against a stale recording while the two implementations have diverged — the
        exact failure the fixture exists to prevent.
        """
        fixture = json.loads((ROOT / "grammar" / "conformance.json").read_text())
        self.assertGreater(len(fixture["states"]), 12000)
        mismatches = []
        for case in fixture["states"]:
            state = (case["field"], case["text"], case["total"])
            allowed = self.auto.allowed(state)
            got = "".join(sorted(c for c in allowed if c != EOS))
            if got != case["allowed"] or (EOS in allowed) != case["canEnd"]:
                mismatches.append(case)
            if self.auto.completion_len((case["field"], case["text"])) != case["completionLen"]:
                mismatches.append(case)
        self.assertEqual(mismatches[:3], [], f"{len(mismatches)} fixture states disagree")
        for case in fixture["expressions"]:
            with self.subTest(text=case["text"]):
                self.assertEqual(is_well_formed(case["text"], self.auto), case["wellFormed"])

    def test_accepts_corpus_fixture(self) -> None:
        """The committed fixture covers every generation bucket, so this runs on a fresh
        clone — `data/out/` is gitignored and the full-corpus check below silently skipped
        there, which is exactly where the automaton's coverage most needs pinning."""
        fixture = ROOT / "tests" / "fixtures" / "corpus.txt"
        crons = [c for c in fixture.read_text().splitlines() if c.strip()]
        self.assertGreater(len(crons), 100)
        bad = [c for c in crons if not is_well_formed(c, self.auto)]
        self.assertEqual(bad[:5], [], f"{len(bad)} corpus expressions rejected by the automaton")

    def test_accepts_full_corpus_when_generated(self) -> None:
        path = ROOT / "data" / "out" / "canonical.jsonl"
        if not path.exists():
            self.skipTest("data/out not generated; the committed fixture covers this case")
        crons = [json.loads(l)["cron"] for l in path.read_text().splitlines() if l.strip()]
        bad = [c for c in crons if not is_well_formed(c, self.auto)]
        self.assertEqual(bad[:5], [], f"{len(bad)} corpus expressions rejected by the automaton")

    def test_rejects_near_misses(self) -> None:
        invalid = [
            "",
            "*",
            "* * * *",
            "* * * * * *",
            "60 * * * *",
            "0 24 * * *",
            "0 0 0 * *",
            "0 0 32 * *",
            "0 0 * 13 *",
            "0 0 * * 7",  # dow 7 is real cron but not in this dialect
            "*/0 * * * *",
            "*/-1 * * * *",
            "1- * * * *",
            "-5 * * * *",
            "5-2 * * * *",
            "1,,2 * * * *",
            "1, * * * *",
            "*, * * * *",
            "1/ * * * *",
            "*/ * * * *",
            " 1 * * * *",
            "1 * * * * ",
            "1  * * * *",
            "1 * * * *x",
            "1 * * * 1-5/two",
            "a * * * *",
            "1 * * * * *",
            "31 0 0 0 0",
            "0 0 5-10 * 1",
        ]
        # Last entry is the deliberate exception: DOM+DOW parses, it just means OR.
        for text in invalid[:-1]:
            with self.subTest(text=text):
                self.assertFalse(is_well_formed(text, self.auto), f"should reject {text!r}")
        self.assertTrue(is_well_formed(invalid[-1], self.auto))

    def test_never_dead_ends(self) -> None:
        """From any state reachable by legal moves, some legal move must remain."""
        rng = random.Random(7)
        seen = set()
        frontier = [self.auto.start()]
        for _ in range(4000):
            if not frontier:
                break
            state = frontier.pop(rng.randrange(len(frontier)))
            if state in seen or self.auto.is_terminal(state):
                continue
            seen.add(state)
            allowed = self.auto.allowed(state)
            self.assertTrue(allowed, f"dead end at {state!r}")
            for ch in allowed:
                nxt = self.auto.advance(state, ch)
                self.assertIsNotNone(nxt)
                if nxt not in seen and len(frontier) < 20000:
                    frontier.append(nxt)
        # EOS leads to the terminal state, the one place an empty allowed() is correct.
        state = self.auto.start()
        for ch in "* * * * *":
            state = self.auto.advance(state, ch)
            self.assertIsNotNone(state)
        self.assertIn(EOS, self.auto.allowed(state))
        terminal = self.auto.advance(state, EOS)
        self.assertIsNotNone(terminal)
        self.assertTrue(self.auto.is_terminal(terminal))
        self.assertEqual(self.auto.allowed(terminal), frozenset())
        self.assertTrue(self.auto.is_complete(terminal))
        self.assertGreater(len(seen), 500, "exploration was too shallow to prove anything")

    def test_adversarial_walks_never_dead_end(self) -> None:
        """Burns the length budget on purpose; a uniform walk never reaches the cap."""
        rng = random.Random(23)
        for _ in range(400):
            state = self.auto.start()
            for step in range(self.grammar.max_length + 5):
                allowed = self.auto.allowed(state)
                self.assertTrue(allowed, f"dead end at {state!r} after {step} steps")
                if EOS in allowed:
                    break
                scored = []
                for ch in allowed:
                    if ch == " ":
                        continue
                    nxt = self.auto.advance(state, ch)
                    assert nxt is not None
                    scored.append((self.auto.completion_len((nxt[0], nxt[1])), ch, nxt))
                if not scored:
                    # Only "end this field and move on" is available; take it.
                    state = self.auto.advance(state, " ")
                    self.assertIsNotNone(state)
                    continue
                scored.sort(reverse=True, key=lambda t: t[0])
                _cost, _ch, nxt = scored[rng.randrange(min(3, len(scored)))]
                state = nxt
                self.assertLessEqual(state[2], self.grammar.max_length)
            self.assertTrue(
                self.auto.is_complete(state) or EOS in self.auto.allowed(state),
                f"walk did not reach a finishable state: {state!r}",
            )

    def test_exhaustive_reachability_has_no_dead_ends(self) -> None:
        """Walk reachable states until the budget runs out, asserting a move always exists.

        The frontier is popped depth-first and capped, so this samples reachable states
        rather than proving the whole space; `exploration` in the browser suite makes the
        same trade.
        """
        seen = {self.auto.start()}
        frontier = [self.auto.start()]
        checked = 0
        while frontier and checked < 150_000:
            state = frontier.pop()
            checked += 1
            if self.auto.is_terminal(state):
                continue
            allowed = self.auto.allowed(state)
            self.assertTrue(allowed, f"dead end at reachable state {state!r}")
            for ch in allowed:
                nxt = self.auto.advance(state, ch)
                self.assertIsNotNone(nxt)
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        self.assertGreater(checked, 1000, "exploration was too shallow to prove anything")

    def test_random_walks_are_valid_cron(self) -> None:
        rng = random.Random(11)
        walks = []
        for _ in range(3000):
            state = self.auto.start()
            chars: list[str] = []
            for _ in range(self.grammar.max_length + 2):
                allowed = sorted(self.auto.allowed(state))
                self.assertTrue(allowed)
                ch = rng.choice(allowed)
                if ch == EOS:
                    break
                chars.append(ch)
                state = self.auto.advance(state, ch)
            text = "".join(chars)
            self.assertLessEqual(len(text), self.grammar.max_length)
            self.assertTrue(self.auto.is_complete(state), f"walk did not finish: {text!r}")
            walks.append(text)

        with CronService() as service:
            results = service.validate(walks)
        self.assertEqual(len(results), len(walks))
        broken = [v["cron"] for v in results if not v["parses"]]
        self.assertEqual(broken[:5], [], f"{len(broken)} walks produced cron-parser rejects")
        # Feb-30-style dates parse but never fire; counted, not asserted to be zero.
        never_fires = [v["cron"] for v in results if not v["fires"]]
        self.assertLess(len(never_fires), len(walks) * 0.02, f"too many non-firing: {never_fires[:5]}")


if __name__ == "__main__":
    unittest.main()
