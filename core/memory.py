"""
Memory Manager Module
Handles different types of memory for the agent, including:
- Perception memory (short-term sensory buffer)
- Working memory (active processing)
- Episodic memory (past experiences)
- Semantic memory (knowledge and facts)
"""

import logging
import time
import json
import os
import threading
import queue
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3

class MemoryManager:
    """
    Manages various memory systems for the agent.
    """

    def __init__(self, config):
        """Initialize the memory manager."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.memory_dir = config.get('memory_dir', 'memory')

        # Create directories if they don't exist
        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(os.path.join(self.memory_dir, 'episodic'), exist_ok=True)
        os.makedirs(os.path.join(self.memory_dir, 'semantic'), exist_ok=True)

        # Database connection is thread-local
        self.db_path = os.path.join(self.memory_dir, 'memory.db')
        self._local = threading.local()
        self._local.conn = None

        # Initialize database
        self._get_connection()

        # Set capacity limits
        self.perception_capacity = config.get('perception_capacity', 100)
        self.working_memory_capacity = config.get('working_memory_capacity', 20)

        # Initialize working memory and perception memory
        self.working_memory = []
        self.perception_memory = []

        # Initialize memory queue for background processing
        self.memory_queue = queue.Queue()

        # Memory worker thread
        self.memory_worker_thread = None

        # Flag for running state
        self.running = False

        self.logger.info(f"Memory manager initialized with storage at {self.memory_dir}")


    def start(self):
        """Start the memory manager."""
        if self.running:
            self.logger.warning("Memory manager is already running")
            return

        self.running = True

        # Start memory worker thread
        self.memory_worker_thread = threading.Thread(
            target=self._memory_worker,
            name="memory_worker_thread",
            daemon=True
        )
        self.memory_worker_thread.start()

        self.logger.info("Memory manager started")

    def stop(self):
        """Stop the memory manager."""
        if not self.running:
            return

        self.running = False

        # Wait for memory worker to finish
        if self.memory_worker_thread and self.memory_worker_thread.is_alive():
            try:
                self.memory_worker_thread.join(timeout=2.0)
            except Exception as e:
                self.logger.error(f"Error stopping memory worker thread: {e}")

        # Close all database connections
        if hasattr(self._local, 'conn') and self._local.conn:
            try:
                self._local.conn.close()
                self._local.conn = None
            except Exception as e:
                self.logger.error(f"Error closing memory database connection: {e}")

        self.logger.info("Memory manager stopped")

    def add_perception(self, perception: Dict[str, Any]):
        """
        Add a perception to short-term memory.

        Args:
            perception: Perception data
        """
        # Add to perception memory
        self.perception_memory.append(perception)

        # Trim if over capacity
        while len(self.perception_memory) > self.perception_capacity:
            self.perception_memory.pop(0)

        # Schedule for episodic storage
        self.memory_queue.put({
            'type': 'store_episodic',
            'data': {
                'timestamp': perception.get('timestamp', time.time()),
                'episode_type': f"perception_{perception.get('type', 'unknown')}",
                'content': json.dumps(perception),
                'importance': 0.3  # Default importance for perceptions
            }
        })

    def add_to_working_memory(self, item: Dict[str, Any], importance: float = 0.5):
        """
        Add an item to working memory.

        Args:
            item: Memory item
            importance: Importance score (0-1)
        """
        # Add timestamp if not present
        if 'timestamp' not in item:
            item['timestamp'] = time.time()

        # Add importance
        item['importance'] = importance

        # Add to working memory
        self.working_memory.append(item)

        # Sort by importance
        self.working_memory.sort(key=lambda x: x.get('importance', 0), reverse=True)

        # Trim if over capacity
        while len(self.working_memory) > self.working_memory_capacity:
            removed = self.working_memory.pop()

            # Store important items to episodic memory
            if removed.get('importance', 0) > 0.4:
                self.memory_queue.put({
                    'type': 'store_episodic',
                    'data': {
                        'timestamp': removed.get('timestamp', time.time()),
                        'episode_type': removed.get('type', 'working_memory'),
                        'content': json.dumps(removed),
                        'importance': removed.get('importance', 0.5)
                    }
                })

    def get_working_memory(self) -> List[Dict[str, Any]]:
        """
        Get current working memory.

        Returns:
            List of working memory items
        """
        return self.working_memory

    def get_recent_perceptions(self, limit=5):
        """Get the most recent perceptions from memory."""
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return []

            cursor = conn.cursor()

            # Query for recent perceptions
            cursor.execute("""
                SELECT id, timestamp, sensor_type, sensor_name, data, interpretation
                FROM perceptions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            # Process results
            perceptions = []
            for row in cursor.fetchall():
                try:
                    # Try to parse data and interpretation as JSON
                    data = json.loads(row[4]) if row[4] else {}
                    interpretation = json.loads(row[5]) if row[5] else ""

                    perception = {
                        'id': row[0],
                        'timestamp': row[1],
                        'type': row[2],
                        'sensor': row[3],
                        'data': data,
                        'interpretation': interpretation
                    }

                    perceptions.append(perception)
                except json.JSONDecodeError:
                    # Fallback to raw strings
                    perception = {
                        'id': row[0],
                        'timestamp': row[1],
                        'type': row[2],
                        'sensor': row[3],
                        'data': row[4],
                        'interpretation': row[5]
                    }

                    perceptions.append(perception)

            self.logger.debug(f"Retrieved {len(perceptions)} recent perceptions")
            return perceptions

        except Exception as e:
            self.logger.error(f"Error getting recent perceptions: {e}")
            return []

    def get_perceptions_by_type(self, perception_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent perceptions of specific type.

        Args:
            perception_type: Type of perception
            limit: Maximum number to return

        Returns:
            List of perceptions
        """
        matching = [p for p in self.perception_memory if p.get('type') == perception_type]
        return matching[-limit:]

    def store_episodic_memory(self, memory_data):
        """Store an episodic memory."""
        # Use the memory queue to handle this asynchronously
        self.memory_queue.put({
            'type': 'store_episodic',
            'data': memory_data
        })

    def _store_episodic_memory(self, episode: Dict[str, Any]):
        """Internal method to store episodic memory in database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                '''
                INSERT INTO episodic_memory
                (timestamp, episode_type, content, importance, last_accessed, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                (
                    episode.get('timestamp', time.time()),
                    episode.get('episode_type', 'unknown'),
                    episode.get('content', '{}'),
                    episode.get('importance', 0.5),
                    time.time(),
                    time.time()
                )
            )

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error storing episodic memory: {e}")

    def retrieve_episodic_memories(self, query=None, min_importance=0.0, limit=10):
        """Retrieve episodic memories based on criteria."""
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return []

            cursor = conn.cursor()

            # Build query
            sql = """
                SELECT id, timestamp, episode_type, content, importance
                FROM episodic_memories
                WHERE importance >= ?
            """
            params = [min_importance]

            if query:
                sql += " AND content LIKE ?"
                params.append(f"%{query}%")

            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            # Execute query
            cursor.execute(sql, params)

            # Process results
            memories = []
            for row in cursor.fetchall():
                memories.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'episode_type': row[2],
                    'content': row[3],
                    'importance': row[4]
                })

            self.logger.debug(f"Retrieved {len(memories)} episodic memories")
            return memories

        except Exception as e:
            self.logger.error(f"Error retrieving episodic memories: {e}")
            return []

    def store_semantic_memory(self, concept: str, content: Dict[str, Any], confidence: float = 0.5):
        """
        Store a semantic memory (knowledge).

        Args:
            concept: The concept name/key
            content: Concept definition/information
            confidence: Confidence in this information (0-1)
        """
        self.memory_queue.put({
            'type': 'store_semantic',
            'data': {
                'concept': concept,
                'content': json.dumps(content),
                'confidence': confidence
            }
        })

    def _store_semantic_memory(self, memory: Dict[str, Any]):
        """Internal method to store semantic memory in database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Check if concept already exists
            cursor.execute(
                "SELECT id FROM semantic_memory WHERE concept = ?",
                (memory.get('concept'),)
            )

            existing = cursor.fetchone()

            if existing:
                # Update existing concept
                cursor.execute(
                    '''
                    UPDATE semantic_memory
                    SET content = ?, confidence = ?, last_updated = ?
                    WHERE id = ?
                    ''',
                    (
                        memory.get('content', '{}'),
                        memory.get('confidence', 0.5),
                        time.time(),
                        existing[0]
                    )
                )
            else:
                # Insert new concept
                cursor.execute(
                    '''
                    INSERT INTO semantic_memory
                    (concept, content, confidence, last_updated, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ''',
                    (
                        memory.get('concept', ''),
                        memory.get('content', '{}'),
                        memory.get('confidence', 0.5),
                        time.time(),
                        time.time()
                    )
                )

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Error storing semantic memory: {e}")

    def retrieve_semantic_memories(
        self,
        concept: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve semantic memories.

        Args:
            concept: Filter by concept (supports SQL LIKE patterns)
            min_confidence: Minimum confidence score
            limit: Maximum number to return

        Returns:
            List of matching semantic memories
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM semantic_memory WHERE confidence >= ?"
            params = [min_confidence]

            if concept:
                if '%' in concept:
                    # Pattern matching
                    query += " AND concept LIKE ?"
                    params.append(concept)
                else:
                    # Exact match
                    query += " AND concept = ?"
                    params.append(concept)

            query += " ORDER BY confidence DESC, last_updated DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert to dictionaries
            result = []
            for row in rows:
                memory = dict(row)
                # Parse JSON content
                try:
                    memory['content'] = json.loads(memory['content'])
                except:
                    pass
                result.append(memory)

            conn.close()
            return result

        except Exception as e:
            self.logger.error(f"Error retrieving semantic memories: {e}")
            return []

    def _consolidate_memories(self):
        """
        Consolidate memories - move important episodic memories to semantic memory.
        This simulates learning and knowledge acquisition from experiences.
        """
        self.logger.info("Consolidating memories")

        try:
            # Get highly accessed important memories
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Find important and frequently accessed episodic memories
            cursor.execute(
                '''
                SELECT * FROM episodic_memory
                WHERE importance > 0.7 AND access_count > 3
                ORDER BY importance DESC, access_count DESC
                LIMIT 10
                '''
            )

            important_memories = cursor.fetchall()

            for memory in important_memories:
                # Consider this for semantic memory
                try:
                    content = json.loads(memory['content'])

                    # Generate a concept name based on content
                    if 'type' in content and 'interpretation' in content:
                        # For perceptions
                        concept = f"learned_{content['type']}_{int(memory['timestamp'])}"
                        semantic_content = {
                            'derived_from': 'episodic',
                            'original_id': memory['id'],
                            'type': content['type'],
                            'knowledge': content['interpretation'],
                            'consolidated_at': time.time()
                        }

                        # Store as semantic memory
                        self.store_semantic_memory(
                            concept=concept,
                            content=semantic_content,
                            confidence=memory['importance'] * 0.8  # Slightly lower confidence
                        )

                except Exception as e:
                    self.logger.error(f"Error consolidating memory {memory['id']}: {e}")

            conn.close()

        except Exception as e:
            self.logger.error(f"Error in memory consolidation: {e}")

    def schedule_consolidation(self):
        """Schedule a memory consolidation operation."""
        self.memory_queue.put({'type': 'consolidate'})

    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            try:
                self._local.conn = sqlite3.connect(self.db_path)
                self._init_db_tables(self._local.conn)
            except Exception as e:
                self.logger.error(f"Error connecting to database: {e}")
                return None
        return self._local.conn

    def _init_db_tables(self, conn):
        """Initialize database tables if they don't exist."""
        try:
            cursor = conn.cursor()

            # Create perceptions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS perceptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    sensor_type TEXT,
                    sensor_name TEXT,
                    data TEXT,
                    interpretation TEXT
                )
            """)

            # Create episodic memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    episode_type TEXT,
                    content TEXT,
                    importance REAL
                )
            """)

            # Create semantic memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept TEXT,
                    description TEXT,
                    relationships TEXT,
                    importance REAL,
                    last_updated REAL
                )
            """)

            conn.commit()
            self.logger.debug("Database tables initialized")

        except Exception as e:
            self.logger.error(f"Error initializing database tables: {e}")

    def store_perception(self, sensor_type, sensor_name, data, interpretation=""):
        """
        Store a perception in memory.

        Args:
            sensor_type: Type of sensor
            sensor_name: Name of sensor
            data: Sensor data (JSON string)
            interpretation: Optional interpretation of data

        Returns:
            ID of stored perception
        """
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return None

            cursor = conn.cursor()

            # Insert into database
            cursor.execute("""
                INSERT INTO perceptions
                (timestamp, sensor_type, sensor_name, data, interpretation)
                VALUES (?, ?, ?, ?, ?)
            """, (time.time(), sensor_type, sensor_name, data, interpretation))

            perception_id = cursor.lastrowid

            # Commit changes
            conn.commit()

            # Trim old perceptions if needed
            self._trim_perceptions()

            self.logger.debug(f"Stored perception from {sensor_name} ({sensor_type})")

            return perception_id

        except Exception as e:
            self.logger.error(f"Error storing perception: {e}")
            return None

    def _trim_perceptions(self):
        """Trim old perceptions if over capacity."""
        try:
            conn = self._get_connection()
            if not conn:
                return

            cursor = conn.cursor()

            # Count perceptions
            cursor.execute("SELECT COUNT(*) FROM perceptions")
            count = cursor.fetchone()[0]

            # If over capacity, trim
            if count > self.perception_capacity:
                # Delete oldest perceptions
                excess = count - self.perception_capacity
                cursor.execute("""
                    DELETE FROM perceptions
                    WHERE id IN (
                        SELECT id FROM perceptions
                        ORDER BY timestamp ASC
                        LIMIT ?
                    )
                """, (excess,))

                conn.commit()
                self.logger.debug(f"Trimmed {excess} old perceptions")

        except Exception as e:
            self.logger.error(f"Error trimming perceptions: {e}")

    def get_memories_about(self, entity, limit=5):
        """
        Get memories related to a specific entity (e.g., person, object).

        Args:
            entity: The entity to search for
            limit: Maximum number of memories to return

        Returns:
            List of memory items
        """
        memories = []

        try:
            # First check episodic memories
            episodic = self.retrieve_episodic_memories(
                query=entity,
                min_importance=0.3,
                limit=limit
            )

            for memory in episodic:
                memories.append({
                    'type': 'episodic',
                    'content': memory.get('content', ''),
                    'timestamp': memory.get('timestamp', 0)
                })

            # Then check working memory
            for item in self.working_memory:
                content = str(item.get('content', ''))
                if entity.lower() in content.lower():
                    memories.append({
                        'type': 'working',
                        'content': content,
                        'timestamp': item.get('timestamp', 0)
                    })

            # Sort by timestamp (most recent first)
            memories.sort(key=lambda x: x.get('timestamp', 0), reverse=True)

            # Limit results
            memories = memories[:limit]

            return memories

        except Exception as e:
            self.logger.error(f"Error getting memories about {entity}: {e}")
            return []

    def store_user_info(self, user_id, user_name, metadata=None):
        """
        Store information about a user.

        Args:
            user_id: Unique identifier for the user
            user_name: Name of the user
            metadata: Additional metadata about the user

        Returns:
            Success flag
        """
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return False

            cursor = conn.cursor()

            # Create users table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    metadata TEXT,
                    last_interaction REAL
                )
            """)

            # Convert metadata to JSON if it's a dict
            if metadata is not None and not isinstance(metadata, str):
                metadata = json.dumps(metadata)

            # Insert or update user
            cursor.execute("""
                INSERT OR REPLACE INTO users
                (id, name, metadata, last_interaction)
                VALUES (?, ?, ?, ?)
            """, (user_id, user_name, metadata, time.time()))

            conn.commit()
            self.logger.debug(f"Stored user info for {user_name} ({user_id})")
            return True

        except Exception as e:
            self.logger.error(f"Error storing user info: {e}")
            return False

    def get_user_info(self, user_id=None, user_name=None):
        """
        Get information about a user.

        Args:
            user_id: Unique identifier for the user
            user_name: Name of the user

        Returns:
            User info dict or None if not found
        """
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return None

            cursor = conn.cursor()

            # Check if users table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='users'
            """)

            if not cursor.fetchone():
                return None

            # Query based on provided parameters
            if user_id:
                cursor.execute("""
                    SELECT id, name, metadata, last_interaction
                    FROM users
                    WHERE id = ?
                """, (user_id,))
            elif user_name:
                cursor.execute("""
                    SELECT id, name, metadata, last_interaction
                    FROM users
                    WHERE name = ?
                """, (user_name,))
            else:
                return None

            row = cursor.fetchone()

            if not row:
                return None

            # Parse metadata
            metadata = row[2]
            try:
                if metadata:
                    metadata = json.loads(metadata)
            except:
                pass

            return {
                'id': row[0],
                'name': row[1],
                'metadata': metadata,
                'last_interaction': row[3]
            }

        except Exception as e:
            self.logger.error(f"Error getting user info: {e}")
            return None

    def update_perception(self, perception_id, interpretation=None):
        """
        Update a perception with interpretation.

        Args:
            perception_id: ID of the perception to update
            interpretation: Interpretation to add

        Returns:
            Success flag
        """
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return False

            cursor = conn.cursor()

            # Update perception
            cursor.execute("""
                UPDATE perceptions
                SET interpretation = ?
                WHERE id = ?
            """, (interpretation, perception_id))

            conn.commit()
            self.logger.debug(f"Updated perception {perception_id} with interpretation")
            return True

        except Exception as e:
            self.logger.error(f"Error updating perception: {e}")
            return False

    def store_conversation_chunk(self, speaker, content, importance=0.6):
        """
        Store a chunk of conversation.

        Args:
            speaker: Who said it
            content: What was said
            importance: Importance score

        Returns:
            Success flag
        """
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return False

            cursor = conn.cursor()

            # Create conversations table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    speaker TEXT,
                    content TEXT,
                    importance REAL
                )
            """)

            # Insert conversation
            cursor.execute("""
                INSERT INTO conversations
                (timestamp, speaker, content, importance)
                VALUES (?, ?, ?, ?)
            """, (time.time(), speaker, content, importance))

            conn.commit()
            return True

        except Exception as e:
            self.logger.error(f"Error storing conversation: {e}")
            return False

    def get_recent_conversation(self, limit=5):
        """
        Get the most recent conversation chunks.

        Args:
            limit: Maximum number of chunks to return

        Returns:
            List of conversation chunks
        """
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return []

            cursor = conn.cursor()

            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='conversations'
            """)

            if not cursor.fetchone():
                return []

            # Get recent conversation
            cursor.execute("""
                SELECT id, timestamp, speaker, content, importance
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            # Process results
            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'speaker': row[2],
                    'content': row[3],
                    'importance': row[4]
                })

            # Reverse to get chronological order
            conversations.reverse()

            return conversations

        except Exception as e:
            self.logger.error(f"Error getting recent conversation: {e}")
            return []

    def store_fact(self, entity, attribute, value, source="conversation"):
        """
        Store a fact about an entity.

        Args:
            entity: The entity the fact is about (e.g., 'user:12345')
            attribute: The attribute (e.g., 'age')
            value: The value (e.g., '42')
            source: Where this fact came from

        Returns:
            Success flag
        """
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return False

            cursor = conn.cursor()

            # Create facts table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity TEXT,
                    attribute TEXT,
                    value TEXT,
                    source TEXT,
                    timestamp REAL,
                    confidence REAL
                )
            """)

            # Check if fact already exists
            cursor.execute("""
                SELECT id FROM facts
                WHERE entity = ? AND attribute = ?
            """, (entity, attribute))

            existing_id = cursor.fetchone()

            if existing_id:
                # Update existing fact
                cursor.execute("""
                    UPDATE facts
                    SET value = ?, source = ?, timestamp = ?, confidence = ?
                    WHERE id = ?
                """, (value, source, time.time(), 0.9, existing_id[0]))
            else:
                # Insert new fact
                cursor.execute("""
                    INSERT INTO facts
                    (entity, attribute, value, source, timestamp, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (entity, attribute, value, source, time.time(), 0.9))

            conn.commit()
            return True

        except Exception as e:
            self.logger.error(f"Error storing fact: {e}")
            return False

    def get_facts(self, entity=None, attribute=None, limit=10):
        """
        Get facts about an entity.

        Args:
            entity: The entity to get facts about (optional)
            attribute: The attribute to get (optional)
            limit: Maximum number of facts to return

        Returns:
            List of facts
        """
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return []

            cursor = conn.cursor()

            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='facts'
            """)

            if not cursor.fetchone():
                return []

            # Build query
            sql = "SELECT id, entity, attribute, value, source, timestamp, confidence FROM facts"
            params = []

            where_clauses = []
            if entity:
                where_clauses.append("entity = ?")
                params.append(entity)

            if attribute:
                where_clauses.append("attribute = ?")
                params.append(attribute)

            if where_clauses:
                sql += " WHERE " + " AND ".join(where_clauses)

            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            # Execute query
            cursor.execute(sql, params)

            # Process results
            facts = []
            for row in cursor.fetchall():
                facts.append({
                    'id': row[0],
                    'entity': row[1],
                    'attribute': row[2],
                    'value': row[3],
                    'source': row[4],
                    'timestamp': row[5],
                    'confidence': row[6]
                })

            return facts

        except Exception as e:
            self.logger.error(f"Error getting facts: {e}")
            return []

    def get_last_summary_time(self):
        """
        Get the timestamp of the last conversation summary.

        Returns:
            Timestamp of last summary or None if no summaries exist
        """
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return None

            cursor = conn.cursor()

            # Query for last summary
            cursor.execute("""
                SELECT timestamp
                FROM episodic_memories
                WHERE episode_type = 'conversation_summary'
                ORDER BY timestamp DESC
                LIMIT 1
            """)

            row = cursor.fetchone()

            if row:
                return row[0]

            return None

        except Exception as e:
            self.logger.error(f"Error getting last summary time: {e}")
            return None

    def get_conversation_summaries(self, user_id=None, limit=5):
        """
        Get conversation summaries for a user.

        Args:
            user_id: User ID (optional)
            limit: Maximum number of summaries to return

        Returns:
            List of conversation summaries
        """
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return []

            cursor = conn.cursor()

            # Query for summaries
            if user_id:
                cursor.execute("""
                    SELECT id, timestamp, content
                    FROM episodic_memories
                    WHERE episode_type = 'conversation_summary'
                    AND content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (f'%"user_id": "{user_id}"%', limit))
            else:
                cursor.execute("""
                    SELECT id, timestamp, content
                    FROM episodic_memories
                    WHERE episode_type = 'conversation_summary'
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

            summaries = []
            for row in cursor.fetchall():
                try:
                    content = json.loads(row[2])

                    summaries.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'user_id': content.get('user_id'),
                        'conversation_id': content.get('conversation_id'),
                        'summary': content.get('summary')
                    })
                except:
                    # Skip corrupted entries
                    continue

            return summaries

        except Exception as e:
            self.logger.error(f"Error getting conversation summaries: {e}")
            return []

    def get_related_memories(self, query, limit=5):
        """
        Get memories related to a specific topic or query.

        Args:
            query: The topic or query to search for
            limit: Maximum number of memories to return

        Returns:
            List of related memories
        """
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return []

            # First, search in conversation summaries
            cursor = conn.cursor()

            # Prepare query terms
            query_terms = query.lower().split()

            # Build SQL query with OR conditions for each term
            sql_query = """
                SELECT id, timestamp, content, importance
                FROM episodic_memories
                WHERE episode_type = 'conversation_summary'
                AND (
            """

            like_conditions = []
            params = []

            for term in query_terms:
                if len(term) > 3:  # Skip short terms
                    like_conditions.append("content LIKE ?")
                    params.append(f"%{term}%")

            if not like_conditions:
                # If no valid terms, use the whole query
                like_conditions.append("content LIKE ?")
                params.append(f"%{query}%")

            sql_query += " OR ".join(like_conditions)
            sql_query += ") ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            # Run query
            cursor.execute(sql_query, params)

            # Process results
            memories = []
            for row in cursor.fetchall():
                try:
                    content = json.loads(row[2])

                    memories.append({
                        'type': 'conversation_summary',
                        'id': row[0],
                        'timestamp': row[1],
                        'content': content.get('summary', ''),
                        'importance': row[3]
                    })
                except:
                    # Skip corrupted entries
                    continue

            # If we found enough memories, return them
            if len(memories) >= limit:
                return memories

            # Otherwise, also search in facts
            remaining = limit - len(memories)

            # Search in facts
            sql_query = """
                SELECT id, entity, attribute, value, timestamp
                FROM facts
                WHERE (
            """

            like_conditions = []
            params = []

            for term in query_terms:
                if len(term) > 3:  # Skip short terms
                    like_conditions.append("value LIKE ?")
                    params.append(f"%{term}%")

            if not like_conditions:
                # If no valid terms, use the whole query
                like_conditions.append("value LIKE ?")
                params.append(f"%{query}%")

            sql_query += " OR ".join(like_conditions)
            sql_query += ") ORDER BY timestamp DESC LIMIT ?"
            params.append(remaining)

            # Run query
            cursor.execute(sql_query, params)

            # Process results
            for row in cursor.fetchall():
                memories.append({
                    'type': 'fact',
                    'id': row[0],
                    'entity': row[1],
                    'attribute': row[2],
                    'value': row[3],
                    'timestamp': row[4]
                })

            return memories

        except Exception as e:
            self.logger.error(f"Error getting related memories: {e}")
            return []

    def decay_memories(self, threshold_days=30):
        """
        Apply memory decay to old, low-importance memories.

        Args:
            threshold_days: Age threshold in days

        Returns:
            Number of decayed memories
        """
        try:
            conn = self._get_connection()
            if not conn:
                self.logger.error("No database connection available")
                return 0

            cursor = conn.cursor()

            # Calculate threshold timestamp
            threshold_timestamp = time.time() - (threshold_days * 24 * 60 * 60)

            # Decay episodic memories
            cursor.execute("""
                DELETE FROM episodic_memories
                WHERE timestamp < ?
                AND importance < 0.7  -- Only decay low-importance memories
                AND episode_type != 'conversation_summary'  -- Keep summaries
            """, (threshold_timestamp,))

            # Count how many were deleted
            episodic_count = cursor.rowcount

            # Decay facts
            cursor.execute("""
                DELETE FROM facts
                WHERE timestamp < ?
                AND attribute LIKE 'conversation_%'  -- Only decay conversation facts
            """, (threshold_timestamp,))

            # Count how many were deleted
            fact_count = cursor.rowcount

            # Commit changes
            conn.commit()

            total_count = episodic_count + fact_count
            self.logger.info(f"Decayed {total_count} old memories ({episodic_count} episodic, {fact_count} facts)")

            return total_count

        except Exception as e:
            self.logger.error(f"Error decaying memories: {e}")
            return 0

    def _memory_worker(self):
        """Worker thread for processing memory operations."""
        while self.running:
            try:
                # Get operation from queue
                try:
                    operation = self.memory_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Process operation
                try:
                    op_type = operation.get('type')
                    if op_type == 'store_episodic':
                        self._store_episodic_memory(operation.get('data', {}))
                    elif op_type == 'store_semantic':
                        self._store_semantic_memory(operation.get('data', {}))
                    elif op_type == 'consolidate':
                        self._consolidate_memories()
                    else:
                        self.logger.warning(f"Unknown memory operation: {op_type}")

                except Exception as e:
                    self.logger.error(f"Error processing memory operation: {e}")

                finally:
                    self.memory_queue.task_done()

            except Exception as e:
                self.logger.error(f"Error in memory worker: {e}")
                time.sleep(0.5)  # Prevent thrashing on error
