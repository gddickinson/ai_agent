"""
LLM Manager Module
Handles loading, managing, and running inference on multiple LLM models.
"""

import logging
import threading
import queue
import time
from typing import Dict, Any, List, Callable, Optional, Tuple

import requests

class LLMTask:
    """Represents a task to be processed by an LLM."""

    def __init__(
        self,
        model_name: str,
        prompt: str,
        callback: Optional[Callable] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        task_id: Optional[str] = None
    ):
        """
        Initialize an LLM task.

        Args:
            model_name: Name of the model to use
            prompt: The prompt to send to the model
            callback: Function to call with results
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            task_id: Optional ID for this task
        """
        self.model_name = model_name
        self.prompt = prompt
        self.callback = callback
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.task_id = task_id or str(id(self))
        self.result = None
        self.completed = False
        self.error = None
        self.created_at = time.time()

class LLMModel:
    """Represents an LLM model that can process tasks."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize an LLM model.

        Args:
            name: Model name
            config: Model configuration
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.busy = False
        self.current_task = None

        # Stats
        self.tasks_processed = 0
        self.total_processing_time = 0

        # Configure model based on type
        self.model_type = config.get('type', 'ollama')

        if self.model_type == 'ollama':
            self.api_base = config.get('api_base', 'http://localhost:11434/api')
            self.model_id = config.get('model_id', 'llama3')
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.logger.info(f"Initialized model {self.name} ({self.model_type})")

    def process_task(self, task: LLMTask) -> Dict[str, Any]:
        """
        Process a task using this model.

        Args:
            task: The task to process

        Returns:
            Result dictionary
        """
        self.busy = True
        self.current_task = task
        start_time = time.time()

        try:
            if self.model_type == 'ollama':
                return self._process_ollama(task)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        finally:
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.tasks_processed += 1
            self.busy = False
            self.current_task = None

    def _process_ollama(self, task: LLMTask) -> Dict[str, Any]:
        """Process a task using Ollama API."""
        self.logger.debug(f"Processing task with Ollama model {self.model_id}")

        url = f"{self.api_base}/generate"
        payload = {
            "model": self.model_id,
            "prompt": task.prompt,
            "stream": False,
            "options": {
                "temperature": task.temperature,
                "num_predict": task.max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return {
                "text": result.get("response", ""),
                "model": self.model_id,
                "task_id": task.task_id,
                "usage": {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }
            }
        except Exception as e:
            self.logger.error(f"Error in Ollama API call: {e}")
            raise

class LLMManager:
    """
    Manages multiple LLM models and distributes tasks efficiently.
    Handles parallel processing using worker threads.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM manager.

        Args:
            config: Configuration dictionary for LLM setup
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.models: Dict[str, LLMModel] = {}
        self.task_queue = queue.Queue()
        self.workers: List[threading.Thread] = []
        self.running = False
        self.num_workers = config.get('num_workers', 3)

        # Initialize models
        self._initialize_models()

        # Start workers
        self._start_workers()

        self.logger.info(
            f"LLM Manager initialized with {len(self.models)} models and {self.num_workers} workers"
        )

    def _initialize_models(self):
        """Initialize LLM models from configuration."""
        for model_name, model_config in self.config.get('models', {}).items():
            self.models[model_name] = LLMModel(model_name, model_config)
            self.logger.info(f"Initialized model: {model_name}")

    def _start_workers(self):
        """Start worker threads for processing LLM tasks."""
        self.running = True
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"llm_worker_{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            self.logger.debug(f"Started worker {i}")

    def _worker_loop(self):
        """Worker thread function for processing tasks."""
        while self.running:
            try:
                # Get task from queue
                task = self.task_queue.get(timeout=0.5)

                # Find the appropriate model for this task
                model = self.models.get(task.model_name)
                if not model:
                    self.logger.error(f"Model not found: {task.model_name}")
                    task.error = f"Model not found: {task.model_name}"
                    if task.callback:
                        task.callback(None, task)
                    self.task_queue.task_done()
                    continue

                # Process the task
                self.logger.debug(f"Processing task {task.task_id} with model {task.model_name}")
                try:
                    result = model.process_task(task)
                    task.result = result
                    task.completed = True

                    # Call callback if provided
                    if task.callback:
                        task.callback(result, task)

                except Exception as e:
                    self.logger.error(f"Error processing task {task.task_id}: {e}")
                    task.error = str(e)

                    # Call callback with error
                    if task.callback:
                        task.callback(None, task)

                finally:
                    self.task_queue.task_done()

            except queue.Empty:
                # No tasks in queue
                pass

            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
                time.sleep(0.5)  # Prevent thrashing on error

    def submit_task(
        self,
        model_name: str,
        prompt: str,
        callback: Optional[Callable] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        task_id: Optional[str] = None
    ) -> LLMTask:
        """
        Submit a task for processing by an LLM.

        Args:
            model_name: Name of the model to use
            prompt: The prompt to send to the model
            callback: Function to call with results (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            task_id: Optional ID for this task

        Returns:
            LLMTask object for tracking progress
        """
        # Create task
        task = LLMTask(
            model_name=model_name,
            prompt=prompt,
            callback=callback,
            max_tokens=max_tokens,
            temperature=temperature,
            task_id=task_id
        )

        # Add to queue
        self.task_queue.put(task)
        self.logger.debug(f"Submitted task {task.task_id} to model {model_name}")

        return task

    def submit_task_sync(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Submit a task and wait for the result synchronously.

        Args:
            model_name: Name of the model to use
            prompt: The prompt to send to the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: Maximum time to wait for a result

        Returns:
            Result dictionary

        Raises:
            TimeoutError: If the task takes longer than the timeout
        """
        result_container = {}
        event = threading.Event()

        def callback(result, task):
            result_container['result'] = result
            result_container['error'] = task.error
            event.set()

        # Submit the task
        task = self.submit_task(
            model_name=model_name,
            prompt=prompt,
            callback=callback,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Wait for the result
        if not event.wait(timeout):
            raise TimeoutError(f"Task {task.task_id} timed out after {timeout} seconds")

        # Check for errors
        if result_container.get('error'):
            raise RuntimeError(result_container['error'])

        return result_container['result']

    def submit_batch(
        self,
        tasks: List[Tuple[str, str, Dict[str, Any]]]
    ) -> List[LLMTask]:
        """
        Submit a batch of tasks.

        Args:
            tasks: List of tuples (model_name, prompt, kwargs)

        Returns:
            List of LLMTask objects
        """
        results = []
        for model_name, prompt, kwargs in tasks:
            task = self.submit_task(model_name, prompt, **kwargs)
            results.append(task)
        return results

    def stop(self):
        """Stop the manager and all workers."""
        self.running = False
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=2.0)
        self.logger.info("LLM Manager stopped")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about LLM usage.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "queue_size": self.task_queue.qsize(),
            "models": {}
        }

        for name, model in self.models.items():
            stats["models"][name] = {
                "tasks_processed": model.tasks_processed,
                "total_processing_time": model.total_processing_time,
                "avg_processing_time": (
                    model.total_processing_time / model.tasks_processed
                    if model.tasks_processed > 0 else 0
                ),
                "busy": model.busy
            }

        return stats


    def submit_task_sync(self, model_name, prompt, timeout=30.0, **kwargs):
        """
        Submit a task to an LLM and wait for the result (synchronous).

        Args:
            model_name: Name of the model to use
            prompt: Prompt to send to the model
            timeout: Maximum time to wait for response (seconds)
            **kwargs: Additional parameters for the model

        Returns:
            Result dictionary or None if timeout or error
        """
        result_container = [None]
        event = threading.Event()

        def callback(result, task):
            result_container[0] = result
            event.set()

        # Submit task asynchronously
        self.submit_task(model_name, prompt, callback, **kwargs)

        # Wait for result with timeout
        if event.wait(timeout):
            return result_container[0]
        else:
            self.logger.warning(f"Timeout waiting for LLM response from {model_name}")
            return None
