"""Episodic memory uses one table for writes, reads, decay and consolidation; old databases are migrated safely.

The legacy database is built here with the pre-fix schema: episodes in `episodic_memory` (written by the agent but
never read) and a few in `episodic_memories` (read but never written by current code), all older than 30 days.
"""
import json
import sqlite3
import time

import pytest

from core.memory import MemoryManager

DAY = 86400.0


def legacy_db(path):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE episodic_memories (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, episode_type TEXT,
            content TEXT, importance REAL);
        CREATE TABLE episodic_memory (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, episode_type TEXT,
            content TEXT, importance REAL, last_accessed REAL, created_at REAL);
    """)
    old = time.time() - 90 * DAY
    for i in range(5):
        conn.execute("INSERT INTO episodic_memory (timestamp, episode_type, content, importance, last_accessed, created_at)"
                     " VALUES (?, 'communication', ?, 0.3, ?, ?)", (old + i, json.dumps({"n": i}), old, old))
    conn.execute("INSERT INTO episodic_memories (timestamp, episode_type, content, importance) VALUES (?, ?, ?, ?)",
                 (old, 'conversation_summary', json.dumps({"summary": "legacy summary"}), 0.5))
    conn.commit()
    conn.close()


@pytest.fixture
def mm(tmp_path):
    legacy_db(tmp_path / "memory.db")
    m = MemoryManager({"memory_dir": str(tmp_path)})
    yield m
    m.stop()


def test_migration_merges_tables_and_reads_see_written_memories(mm):
    conn = mm._get_connection()
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "episodic_memories" not in names
    assert conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 6
    assert len(mm.retrieve_episodic_memories(limit=10)) == 6
    assert mm.get_last_summary_time() is not None
    assert [s["summary"] for s in mm.get_conversation_summaries()] == ["legacy summary"]


def test_decay_spares_memories_that_were_never_readable(mm):
    assert mm.decay_memories(threshold_days=30) == 0
    assert mm._get_connection().execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 6


def test_decay_removes_memories_unused_for_30_days(mm):
    conn = mm._get_connection()
    conn.execute("UPDATE episodic_memory SET last_accessed = ? WHERE id = 1", (time.time() - 40 * DAY,))
    conn.commit()
    assert mm.decay_memories(threshold_days=30) == 1


def test_migration_runs_once(tmp_path, mm):
    conn = mm._get_connection()
    conn.execute("UPDATE episodic_memory SET last_accessed = 1")
    conn.commit()
    mm.stop()
    again = MemoryManager({"memory_dir": str(tmp_path)})
    assert again._get_connection().execute("SELECT MIN(last_accessed) FROM episodic_memory").fetchone()[0] == 1
    again.stop()


def test_retrieval_counts_accesses_and_consolidation_works(tmp_path):
    m = MemoryManager({"memory_dir": str(tmp_path)})
    ep = {"timestamp": 5.0, "episode_type": "perception_audio", "importance": 0.9,
          "content": json.dumps({"type": "audio", "interpretation": "someone said hello"})}
    m._store_episodic_memory(ep)
    for _ in range(4):
        m.retrieve_episodic_memories(limit=5)
    conn = m._get_connection()
    assert conn.execute("SELECT access_count FROM episodic_memory").fetchone()[0] == 4
    m._consolidate_memories()
    op = m.memory_queue.get_nowait()  # consolidation queues the semantic write for the memory worker
    assert op["type"] == "store_semantic"
    m._store_semantic_memory(op["data"])
    rows = conn.execute("SELECT concept, confidence FROM semantic_memory").fetchall()
    assert rows and rows[0][0] == "learned_audio_5" and rows[0][1] == pytest.approx(0.72)
    m.stop()
