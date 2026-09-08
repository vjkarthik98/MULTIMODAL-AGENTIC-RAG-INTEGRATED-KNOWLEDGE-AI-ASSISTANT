"""Regression coverage for the Recents transcript losing turns (2026-09-08).

Reported symptom: five questions were asked per modality, four were in the
transcript afterwards, and one surviving answer no longer matched its question.

Cause — two halves of the same turn disagreeing about whether it was stored:

1. app/api/api_routes.py's stream handler persists a turn only when the answer
   is non-empty AND was not refused. With an empty knowledge base almost every
   answer IS a refusal, so nothing was stored for those turns.
2. ChatPage.jsx then calls PATCH /rag/sessions/{id}/last-message regardless,
   and MongoMemory.patch_last_assistant_message() overwrote "the last assistant
   message" with no notion of which turn that was. With the current turn
   unstored, that message belonged to the PREVIOUS turn: its answer was
   destroyed and the current question never appeared at all.

The fix makes the patch turn-aware (`expected_query`) so it refuses to touch a
turn it wasn't asked to patch, and makes save_chat_turn() collapse a re-save of
the turn already at the end of the transcript instead of duplicating it — the
stream and the client's meta fallback both persist the same question seconds
apart.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

from app.memory.mongo_memory import MongoMemory


class FakeSessions:
    """Minimal stand-in for the chat_sessions collection — supports only the
    operators MongoMemory actually uses on it."""

    def __init__(self) -> None:
        self.docs: list[dict] = []

    def _match(self, flt: dict, doc: dict) -> bool:
        return all(doc.get(k) == v for k, v in flt.items())

    def find_one(self, flt: dict, projection: dict | None = None) -> dict | None:
        for d in self.docs:
            if self._match(flt, d):
                return copy.deepcopy(d)
        return None

    def update_one(self, flt: dict, update: dict, upsert: bool = False) -> None:
        target = next((d for d in self.docs if self._match(flt, d)), None)
        if target is None:
            if not upsert:
                return
            target = {"_id": len(self.docs) + 1}
            target.update({k: v for k, v in flt.items() if k != "_id"})
            target.update(copy.deepcopy(update.get("$setOnInsert", {})))
            self.docs.append(target)

        target.update(copy.deepcopy(update.get("$set", {})))
        for key, amount in update.get("$inc", {}).items():
            target[key] = target.get(key, 0) + amount
        for key, spec in update.get("$push", {}).items():
            arr = target.setdefault(key, [])
            arr.extend(copy.deepcopy(spec["$each"]))
            window = spec.get("$slice")
            if window is not None and window < 0:
                target[key] = arr[window:]


def _memory(fake: FakeSessions) -> MongoMemory:
    """A MongoMemory wired to the fake collection, without touching a server."""
    mem = MongoMemory.__new__(MongoMemory)
    mem._enabled = True
    mem._mongo_ok = True
    mem.sessions = fake
    mem._is_available = lambda: True  # type: ignore[method-assign]
    return mem


def _transcript(fake: FakeSessions) -> list[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in fake.docs[0]["messages"]]


SESSION = "s1"
USER = "u1"


class TestPatchIsTurnAware:

    def test_patch_refuses_to_overwrite_a_different_turn(self):
        fake = FakeSessions()
        mem = _memory(fake)
        mem.save_chat_turn(SESSION, USER, "What was Apple's revenue?", "Revenue was $391B.")

        # Turn 2 was refused, so nothing was persisted for it. The client
        # patches anyway — this must not touch turn 1.
        patched = mem.patch_last_assistant_message(
            SESSION,
            USER,
            "I could not find relevant information in your knowledge base.",
            [],
            None,
            expected_query="Summarize the risk factors.",
        )

        assert patched is False
        assert _transcript(fake) == [
            ("user", "What was Apple's revenue?"),
            ("assistant", "Revenue was $391B."),
        ]

    def test_patch_applies_when_it_is_the_current_turn(self):
        fake = FakeSessions()
        mem = _memory(fake)
        mem.save_chat_turn(SESSION, USER, "What was Apple's revenue?", "draft")

        patched = mem.patch_last_assistant_message(
            SESSION,
            USER,
            "Revenue was $391B.",
            [],
            "msg-42",
            expected_query="what was apple's REVENUE?",  # normalization is case/space insensitive
        )

        assert patched is True
        assert _transcript(fake)[-1] == ("assistant", "Revenue was $391B.")
        assert fake.docs[0]["messages"][-1]["msg_id"] == "msg-42"

    def test_patch_without_expected_query_keeps_legacy_behaviour(self):
        # An older frontend build sends no query; the previous unconditional
        # patch semantics must still work for it.
        fake = FakeSessions()
        mem = _memory(fake)
        mem.save_chat_turn(SESSION, USER, "q1", "a1")

        assert mem.patch_last_assistant_message(SESSION, USER, "a1-cleaned", [], None) is True
        assert _transcript(fake)[-1] == ("assistant", "a1-cleaned")


class TestRefusedTurnIsRecoverable:

    def test_missing_turn_can_be_appended_without_losing_the_previous_one(self):
        """The end-to-end shape of the API route's self-heal path."""
        fake = FakeSessions()
        mem = _memory(fake)
        mem.save_chat_turn(SESSION, USER, "Q1", "A1")

        refusal = "I could not find relevant information in your knowledge base."
        if not mem.patch_last_assistant_message(
            SESSION, USER, refusal, [], None, expected_query="Q2"
        ):
            mem.save_chat_turn(SESSION, USER, "Q2", refusal)

        assert _transcript(fake) == [
            ("user", "Q1"),
            ("assistant", "A1"),
            ("user", "Q2"),
            ("assistant", refusal),
        ]


class TestDuplicateTurnCollapse:

    def test_resaving_the_same_question_overwrites_instead_of_duplicating(self):
        # The stream persists first, then the client's meta fallback persists
        # the same question again seconds later.
        fake = FakeSessions()
        mem = _memory(fake)
        mem.save_chat_turn(SESSION, USER, "Q1", "streamed answer")
        mem.save_chat_turn(SESSION, USER, "Q1", "meta answer")

        assert _transcript(fake) == [("user", "Q1"), ("assistant", "meta answer")]

    def test_a_later_reask_gets_its_own_turn(self):
        fake = FakeSessions()
        mem = _memory(fake)
        mem.save_chat_turn(SESSION, USER, "Q1", "first answer")
        # Age the stored turn past the collapse window.
        stale = datetime.now(tz=timezone.utc) - timedelta(hours=2)
        fake.docs[0]["messages"][-1]["timestamp"] = stale

        mem.save_chat_turn(SESSION, USER, "Q1", "second answer")

        assert _transcript(fake) == [
            ("user", "Q1"),
            ("assistant", "first answer"),
            ("user", "Q1"),
            ("assistant", "second answer"),
        ]

    def test_a_different_question_always_appends(self):
        fake = FakeSessions()
        mem = _memory(fake)
        mem.save_chat_turn(SESSION, USER, "Q1", "A1")
        mem.save_chat_turn(SESSION, USER, "Q2", "A2")

        assert _transcript(fake) == [
            ("user", "Q1"),
            ("assistant", "A1"),
            ("user", "Q2"),
            ("assistant", "A2"),
        ]
