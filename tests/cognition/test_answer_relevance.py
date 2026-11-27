"""Regression tests: answers must address the question asked, not merely the same topic."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gajahmada_ai.api import server
from gajahmada_ai.cognition.answer_relevance import (
    extract_focus_terms,
    relevance_score,
    select_best_passage,
)

HISTORY = "Sejarah Indonesia berkembang dari kerajaan klasik hingga Proklamasi 17 Agustus 1945. Kini Indonesia memiliki 38 Provinsi."
CURRENT_PRESIDENT = "Presiden Republik Indonesia saat ini (2026) adalah Prabowo Subianto (Presiden ke-8)."


def test_focus_terms_ignore_question_words_and_keep_qualifiers():
    kinds = {t.term: t.kind for t in extract_focus_terms("Siapa presiden pertama Indonesia?")}
    assert "siapa" not in kinds
    assert kinds["pertama"] == "ordinal" and kinds["presiden"] == "word"
    assert "ibu kota" in {t.term for t in extract_focus_terms("Apa ibu kota Indonesia?")}


def test_off_target_answers_score_low_and_on_target_score_high():
    assert relevance_score("Apa ibu kota Indonesia?", HISTORY) < 0.6
    assert relevance_score("Siapa presiden pertama Indonesia?", CURRENT_PRESIDENT) < 0.6
    assert relevance_score("Siapa presiden Indonesia saat ini?", CURRENT_PRESIDENT) >= 0.6
    assert relevance_score("Berapa jumlah provinsi di Indonesia?", HISTORY) >= 0.6


def test_select_best_passage_returns_only_responsive_line():
    passage = "Daftar Presiden RI:\n1. Ir. Soekarno (1945–1967)\n2. Soeharto (1967–1998)\n3. B.J. Habibie (1998–1999)"
    best = select_best_passage("Siapa presiden pertama Indonesia?", [passage])
    assert best and "Soekarno" in best[0] and "Habibie" not in best[0]
    assert select_best_passage("Apa ibu kota Indonesia?", [HISTORY]) is None


@pytest.fixture(scope="module")
def client():
    server.ensure_initialized()
    return TestClient(server.app)


def _ask(client, q: str) -> str:
    return client.post("/ai/chat", json={"prompt": q}).json()["response"]


def test_first_president_is_not_answered_with_current_president(client):
    ans = _ask(client, "Siapa presiden pertama Indonesia?")
    assert "Soekarno" in ans
    assert "Prabowo" not in ans


def test_ordinal_president_question_returns_requested_ordinal(client):
    ans = _ask(client, "Siapa presiden ke-3 Indonesia?")
    assert "Habibie" in ans and "Prabowo" not in ans


def test_current_president_question_still_answered(client):
    assert "Prabowo" in _ask(client, "Siapa presiden Indonesia saat ini?")


def test_unrelated_history_dump_is_not_given_for_capital_question(client):
    ans = _ask(client, "Apa ibu kota Indonesia?")
    assert "ibu kota" in ans.lower() or "belum menemukan" in ans.lower()
    assert "Budi Utomo" not in ans  # old behaviour: generic history paragraph


def test_existing_correct_answers_unchanged(client):
    assert "Gajah Mada" in _ask(client, "Siapa Mahapatih Gajah Mada?")
    assert "84" in _ask(client, "Berapa 12 * 7?")
    assert "38" in _ask(client, "Berapa jumlah provinsi di Indonesia?")
    assert "Halo" in _ask(client, "Apa kabar?")
