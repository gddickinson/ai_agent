"""Unit tests for the MemoryManager module."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import MemoryManager


@pytest.fixture
def memory_manager(tmp_path):
    """Create a MemoryManager with a temporary directory."""
    config = {
        "memory_dir": str(tmp_path / "memory"),
        "perception_capacity": 10,
        "working_memory_capacity": 5,
    }
    mm = MemoryManager(config)
    yield mm
    try:
        mm.stop()
    except Exception:
        pass


class TestMemoryManagerInit:
    """Tests for MemoryManager initialization."""

    def test_creates_directories(self, memory_manager, tmp_path):
        """Test that required directories are created."""
        mem_dir = tmp_path / "memory"
        assert mem_dir.exists()
        assert (mem_dir / "episodic").exists()
        assert (mem_dir / "semantic").exists()

    def test_db_created(self, memory_manager, tmp_path):
        """Test that the SQLite database file is created."""
        db_path = tmp_path / "memory" / "memory.db"
        assert db_path.exists()

    def test_initial_working_memory_empty(self, memory_manager):
        """Test that working memory starts empty."""
        assert len(memory_manager.working_memory) == 0


class TestWorkingMemory:
    """Tests for working memory operations."""

    def test_add_to_working_memory(self, memory_manager):
        """Test adding an item to working memory."""
        item = {"type": "test", "content": "hello"}
        memory_manager.add_to_working_memory(item, importance=0.5)
        assert len(memory_manager.working_memory) == 1

    def test_working_memory_capacity(self, memory_manager):
        """Test that working memory respects capacity limits."""
        for i in range(20):
            item = {"type": "test", "content": f"item_{i}"}
            memory_manager.add_to_working_memory(item, importance=0.5)
        assert len(memory_manager.working_memory) <= memory_manager.working_memory_capacity
