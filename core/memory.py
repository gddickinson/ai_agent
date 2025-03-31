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
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize memory systems.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configure memory paths
        self.memory_dir = Path(config.get('memory_dir', 'memory'))
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure database
        self.db_path = self.memory_dir / 'memory.db'
        self.initialize_database()
        
        # Memory capacity limits
        self.perception_capacity = config.get('perception_capacity', 100)
        self.working_memory_capacity = config.get('working_memory_capacity', 20)
        
        # In-memory storage for quick access
        self.perception_memory = []  # Recent perceptions
        self.working_memory = []     # Currently relevant items
        
        # Queue for memory operations
        self.memory_queue = queue.Queue()
        self.running = False
        self.memory_thread = None
        
        self.logger.info(f"Memory manager initialized with storage at {self.memory_dir}")
    
    def initialize_database(self):
        """Initialize the SQLite database schema."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Episodic memory table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                episode_type TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                last_accessed REAL,
                access_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            )
            ''')
            
            # Semantic memory table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                last_updated REAL NOT NULL,
                created_at REAL NOT NULL
            )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic_memory(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodic_type ON episodic_memory(episode_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_episodic_importance ON episodic_memory(importance)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_semantic_concept ON semantic_memory(concept)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Memory database initialized")
        
        except Exception as e:
            self.logger.error(f"Error initializing memory database: {e}")
            raise
    
    def start(self):
        """Start the memory manager thread."""
        if self.running:
            return
            
        self.running = True
        self.memory_thread = threading.Thread(
            target=self._memory_worker,
            name="memory_thread",
            daemon=True
        )
        self.memory_thread.start()
        self.logger.info("Memory manager started")
    
    def stop(self):
        """Stop the memory manager thread."""
        if not self.running:
            return
            
        self.running = False
        if self.memory_thread and self.memory_thread.is_alive():
            self.memory_thread.join(timeout=2.0)
            
        self.logger.info("Memory manager stopped")
    
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
    
    def get_recent_perceptions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent perceptions.
        
        Args:
            limit: Maximum number of perceptions to return
            
        Returns:
            List of recent perceptions
        """
        return self.perception_memory[-limit:]
    
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
    
    def store_episodic_memory(self, episode: Dict[str, Any]):
        """
        Store an episodic memory.
        
        Args:
            episode: Episode data with fields:
                - timestamp: When the episode occurred
                - episode_type: Type of episode
                - content: Episode content (JSON serializable)
                - importance: Importance score (0-1)
        """
        self.memory_queue.put({
            'type': 'store_episodic',
            'data': episode
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
    
    def retrieve_episodic_memories(
        self, 
        episode_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        min_importance: float = 0.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve episodic memories matching criteria.
        
        Args:
            episode_type: Filter by episode type
            start_time: Minimum timestamp
            end_time: Maximum timestamp
            min_importance: Minimum importance score
            limit: Maximum number to return
            
        Returns:
            List of matching episodic memories
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM episodic_memory WHERE importance >= ?"
            params = [min_importance]
            
            if episode_type:
                query += " AND episode_type = ?"
                params.append(episode_type)
                
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
                
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
                
            query += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Update access information
            for row in rows:
                cursor.execute(
                    "UPDATE episodic_memory SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                    (time.time(), row['id'])
                )
            
            conn.commit()
            
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
