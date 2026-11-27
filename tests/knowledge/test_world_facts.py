"""World & space fact base: integrity + that questions get the responsive fact (offline)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from gajahmada_ai.api import server
from gajahmada_ai.knowledge.world_facts import get_world_knowledge_corpus


def test_corpus_is_unique_nonempty_sentences():
    docs = get_world_knowledge_corpus()
    assert len(docs) >= 250
    assert len(set(docs)) == len(docs)
    assert all(d.strip().endswith((".", ")")) and len(d) > 20 for d in docs)


@pytest.fixture(scope="module")
def client():
    server.ensure_initialized()
    return TestClient(server.app)


@pytest.mark.parametrize(
    "question, must_contain",
    [
        ("Apa ibu kota Indonesia?", "Jakarta"),
        ("Apa ibu kota Prancis?", "Paris"),
        ("Apa ibu kota Jepang?", "Tokyo"),
        ("Apa mata uang Jepang?", "Yen"),
        ("Planet apa yang terbesar di tata surya?", "Jupiter"),
        ("Siapa manusia pertama yang berjalan di Bulan?", "Armstrong"),
        ("Siapa manusia pertama di luar angkasa?", "Gagarin"),
        ("Berapa jarak rata-rata Bumi ke Bulan?", "384.400"),
        ("Berapa kecepatan cahaya?", "299.792.458"),
        ("Apa puncak tertinggi di dunia?", "Everest"),
        ("Siapa pencipta lagu Indonesia Raya?", "Supratman"),
        ("Berapa jumlah tulang manusia dewasa?", "206"),
    ],
)
def test_question_gets_responsive_fact(client, question, must_contain):
    ans = client.post("/ai/chat", json={"prompt": question}).json()["response"]
    assert must_contain in ans, f"{question!r} -> {ans[:200]!r}"


def test_unknown_fact_is_declined_not_fabricated(client):
    ans = client.post("/ai/chat", json={"prompt": "Apa warna favorit kucing tetangga Zorgblat?"}).json()["response"]
    assert "Zorgblat" in ans and ("belum" in ans.lower() or "tidak" in ans.lower())
