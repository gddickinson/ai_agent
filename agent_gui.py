#!/usr/bin/env python3
"""
Agent GUI
A graphical user interface for interacting with the Embodied AI Agent
"""

import sys
import os
import time
import threading
import queue
import json
import logging
import argparse
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from typing import Dict, Any, List, Optional
import cv2
from PIL import Image, ImageTk
import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.agent import EmbodiedAgent
from utils.config import load_config
from utils.logging import setup_logging


class AgentGUI:
    """GUI for interacting with the Embodied AI Agent."""

    def __init__(self, master, agent):
        """
        Initialize the GUI.

        Args:
            master: Tkinter root window
            agent: EmbodiedAgent instance
        """
        self.master = master
        self.agent = agent
        self.response_queue = queue.Queue()
        self.master.title("Embodied AI Agent Interface")
        self.master.geometry("1200x800")
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        # Setup styles
        self.setup_styles()

        # Create the main frame for the interface
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create paned window for main layout
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Create left panel for chat and interaction
        self.left_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_panel, weight=2)

        # Create right panel for monitoring
        self.right_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_panel, weight=1)

        # Create components in left panel
        self.create_chat_interface(self.left_panel)

        # Create components in right panel
        self.create_monitoring_interface(self.right_panel)

        # Register callbacks for agent
        self.register_agent_callbacks()

        # Setup update timers
        self.setup_timers()

        # Flag for running state
        self.running = True

    def setup_styles(self):
        """Setup custom styles for the interface."""
        self.style = ttk.Style()

        # Configure different styles for different message types
        self.style.configure("Human.TLabel", foreground="blue", font=("Helvetica", 10, "bold"))
        self.style.configure("Agent.TLabel", foreground="green", font=("Helvetica", 10, "bold"))
        self.style.configure("System.TLabel", foreground="gray", font=("Helvetica", 9))
        self.style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))

        # Configure button styles
        self.style.configure("Send.TButton", font=("Helvetica", 10, "bold"))

        # Configure notebook styles
        self.style.configure("TNotebook", padding=5)
        self.style.configure("TNotebook.Tab", font=("Helvetica", 9, "bold"))

    def create_chat_interface(self, parent):
        """Create the chat interface in the left panel."""
        # Create a frame for the chat box
        chat_frame = ttk.LabelFrame(parent, text="Conversation")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create chat history display
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state=tk.DISABLED, height=20)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure tags for different message types
        self.chat_display.tag_configure("human", foreground="blue")
        self.chat_display.tag_configure("agent", foreground="green")
        self.chat_display.tag_configure("system", foreground="gray", font=("Helvetica", 9))
        self.chat_display.tag_configure("thought", foreground="purple", font=("Helvetica", 9, "italic"))

        # Create a frame for the message input
        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create message input box
        self.message_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=3)
        self.message_input.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=5, pady=5)
        self.message_input.bind("<Return>", self.send_message_event)
        self.message_input.bind("<Shift-Return>", lambda e: None)  # Allow Shift+Enter for newline

        # Create send button
        self.send_button = ttk.Button(input_frame, text="Send", style="Send.TButton", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def create_monitoring_interface(self, parent):
        """Create the monitoring interface in the right panel."""
        # Create a notebook for different monitoring views
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create monologue tab
        self.monologue_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.monologue_tab, text="Internal Monologue")
        self.create_monologue_view(self.monologue_tab)

        # Create camera feed tab
        self.camera_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.camera_tab, text="Camera Feed")
        self.create_camera_view(self.camera_tab)

        # Create state tab
        self.state_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.state_tab, text="Agent State")
        self.create_state_view(self.state_tab)

        # Create memory tab
        self.memory_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.memory_tab, text="Memory")
        self.create_memory_view(self.memory_tab)

        # Create autonomy tab
        self.autonomy_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.autonomy_tab, text="Autonomy")
        self.create_autonomy_view(self.autonomy_tab)

    def create_monologue_view(self, parent):
        """Create the internal monologue view."""
        # Create monologue display
        self.monologue_display = scrolledtext.ScrolledText(parent, wrap=tk.WORD, state=tk.DISABLED)
        self.monologue_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Configure tags
        self.monologue_display.tag_configure("timestamp", foreground="gray")
        self.monologue_display.tag_configure("thought", foreground="purple")

    def create_camera_view(self, parent):
        """Create the camera feed view."""
        # Create a frame for camera controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create camera feed label
        self.camera_label = ttk.Label(parent)
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create camera description
        self.camera_description = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=5, state=tk.DISABLED)
        self.camera_description.pack(fill=tk.X, padx=5, pady=5)

        # Initialize camera placeholder
        self.update_camera_feed(None)

    def create_state_view(self, parent):
        """Create the agent state view."""
        # Create notebook for different state aspects
        state_notebook = ttk.Notebook(parent)
        state_notebook.pack(fill=tk.BOTH, expand=True)

        # Create emotions tab
        emotions_tab = ttk.Frame(state_notebook)
        state_notebook.add(emotions_tab, text="Emotions")

        # Create emotional state display
        self.emotions_display = scrolledtext.ScrolledText(emotions_tab, wrap=tk.WORD, state=tk.DISABLED)
        self.emotions_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create drives tab
        drives_tab = ttk.Frame(state_notebook)
        state_notebook.add(drives_tab, text="Drives")

        # Create drives display
        self.drives_display = scrolledtext.ScrolledText(drives_tab, wrap=tk.WORD, state=tk.DISABLED)
        self.drives_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create goals tab
        goals_tab = ttk.Frame(state_notebook)
        state_notebook.add(goals_tab, text="Goals")

        # Create goals display
        self.goals_display = scrolledtext.ScrolledText(goals_tab, wrap=tk.WORD, state=tk.DISABLED)
        self.goals_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_memory_view(self, parent):
        """Create the memory view."""
        # Create notebook for different memory types
        memory_notebook = ttk.Notebook(parent)
        memory_notebook.pack(fill=tk.BOTH, expand=True)

        # Create working memory tab
        working_memory_tab = ttk.Frame(memory_notebook)
        memory_notebook.add(working_memory_tab, text="Working Memory")

        # Create working memory display
        self.working_memory_display = scrolledtext.ScrolledText(working_memory_tab, wrap=tk.WORD, state=tk.DISABLED)
        self.working_memory_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create facts tab
        facts_tab = ttk.Frame(memory_notebook)
        memory_notebook.add(facts_tab, text="Facts")

        # Create facts display
        self.facts_display = scrolledtext.ScrolledText(facts_tab, wrap=tk.WORD, state=tk.DISABLED)
        self.facts_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create perceptions tab
        perceptions_tab = ttk.Frame(memory_notebook)
        memory_notebook.add(perceptions_tab, text="Perceptions")

        # Create perceptions display
        self.perceptions_display = scrolledtext.ScrolledText(perceptions_tab, wrap=tk.WORD, state=tk.DISABLED)
        self.perceptions_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_autonomy_view(self, parent):
        """Create the autonomy view."""
        # Create notebook for different autonomy aspects
        autonomy_notebook = ttk.Notebook(parent)
        autonomy_notebook.pack(fill=tk.BOTH, expand=True)

        # Create topics tab
        topics_tab = ttk.Frame(autonomy_notebook)
        autonomy_notebook.add(topics_tab, text="Topics of Interest")

        # Create topics display
        self.topics_display = scrolledtext.ScrolledText(topics_tab, wrap=tk.WORD, state=tk.DISABLED)
        self.topics_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create questions tab
        questions_tab = ttk.Frame(autonomy_notebook)
        autonomy_notebook.add(questions_tab, text="Questions")

        # Create questions display
        self.questions_display = scrolledtext.ScrolledText(questions_tab, wrap=tk.WORD, state=tk.DISABLED)
        self.questions_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create thinking mode tab
        thinking_tab = ttk.Frame(autonomy_notebook)
        autonomy_notebook.add(thinking_tab, text="Thinking Mode")

        # Create thinking mode display
        self.thinking_mode_display = scrolledtext.ScrolledText(thinking_tab, wrap=tk.WORD, state=tk.DISABLED)
        self.thinking_mode_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def register_agent_callbacks(self):
        """Register callbacks for agent events."""
        self.agent.consciousness.register_message_callback('sent', self._message_sent_callback)

    def _message_sent_callback(self, message):
        """Callback for when a message is sent by the agent."""
        self.response_queue.put(message)

    def setup_timers(self):
        """Setup update timers for GUI components."""
        # Update chat display with new messages
        self.master.after(100, self.check_message_queue)

        # Update monitoring views
        self.master.after(1000, self.update_monitoring_views)

    def check_message_queue(self):
        """Check for new messages in the response queue."""
        try:
            # Check for agent responses
            try:
                while True:
                    message = self.response_queue.get_nowait()
                    self.display_agent_message(message)
                    self.response_queue.task_done()
            except queue.Empty:
                pass

            # Reschedule the check
            if self.running:
                self.master.after(100, self.check_message_queue)

        except Exception as e:
            logging.error(f"Error checking message queue: {e}")
            if self.running:
                self.master.after(100, self.check_message_queue)

    def update_monitoring_views(self):
        """Update all monitoring views."""
        try:
            # Update monologue display
            self.update_monologue_display()

            # Update agent state displays
            self.update_state_displays()

            # Update memory displays
            self.update_memory_displays()

            # Update autonomy displays
            self.update_autonomy_displays()

            # Update camera feed
            self.update_camera_from_agent()

            # Reschedule the update
            if self.running:
                self.master.after(1000, self.update_monitoring_views)

        except Exception as e:
            logging.error(f"Error updating monitoring views: {e}")
            if self.running:
                self.master.after(1000, self.update_monitoring_views)

    def update_monologue_display(self):
        """Update the internal monologue display."""
        monologue = self.agent.consciousness.get_internal_monologue()

        # Only update if there are new entries
        if len(monologue) > 0:
            # Enable editing
            self.monologue_display.configure(state=tk.NORMAL)

            # Clear current content
            self.monologue_display.delete(1.0, tk.END)

            # Add each monologue entry
            for entry in monologue:
                # Parse timestamp and thought
                parts = entry.split('] ', 1)
                if len(parts) == 2:
                    timestamp = parts[0] + ']'
                    thought = parts[1]

                    # Insert timestamp
                    self.monologue_display.insert(tk.END, timestamp + ' ', "timestamp")

                    # Insert thought
                    self.monologue_display.insert(tk.END, thought + '\n', "thought")
                else:
                    # If can't parse, insert the whole entry
                    self.monologue_display.insert(tk.END, entry + '\n')

            # Disable editing
            self.monologue_display.configure(state=tk.DISABLED)

            # Scroll to bottom
            self.monologue_display.see(tk.END)

    def update_state_displays(self):
        """Update the agent state displays."""
        try:
            internal_state = self.agent.cognition.get_internal_state()

            # Update emotions display
            self.emotions_display.configure(state=tk.NORMAL)
            self.emotions_display.delete(1.0, tk.END)

            self.emotions_display.insert(tk.END, "Current Emotional State:\n\n")
            for emotion, value in internal_state.get('emotions', {}).items():
                bar_length = int(value * 20)  # Scale to 20 characters
                bar = '█' * bar_length + '░' * (20 - bar_length)
                self.emotions_display.insert(tk.END, f"{emotion.capitalize()}: {value:.2f} {bar}\n")

            self.emotions_display.configure(state=tk.DISABLED)

            # Update drives display
            self.drives_display.configure(state=tk.NORMAL)
            self.drives_display.delete(1.0, tk.END)

            self.drives_display.insert(tk.END, "Current Drives:\n\n")
            for drive, value in internal_state.get('drives', {}).items():
                bar_length = int(value * 20)  # Scale to 20 characters
                bar = '█' * bar_length + '░' * (20 - bar_length)
                self.drives_display.insert(tk.END, f"{drive.capitalize()}: {value:.2f} {bar}\n")

            self.drives_display.configure(state=tk.DISABLED)

            # Update goals display
            self.goals_display.configure(state=tk.NORMAL)
            self.goals_display.delete(1.0, tk.END)

            self.goals_display.insert(tk.END, "Current Goals:\n\n")
            for i, goal in enumerate(internal_state.get('current_goals', [])):
                self.goals_display.insert(tk.END, f"{i+1}. {goal}\n")

            # Add current plan if available
            current_plan = internal_state.get('current_plan', {})
            if current_plan:
                self.goals_display.insert(tk.END, "\nCurrent Plan:\n\n")

                # Short term actions
                self.goals_display.insert(tk.END, "Short-term:\n")
                for action in current_plan.get('short_term', []):
                    self.goals_display.insert(tk.END, f"- {action}\n")

                # Medium term actions
                self.goals_display.insert(tk.END, "\nMedium-term:\n")
                for action in current_plan.get('medium_term', []):
                    self.goals_display.insert(tk.END, f"- {action}\n")

                # Long term actions
                self.goals_display.insert(tk.END, "\nLong-term:\n")
                for action in current_plan.get('long_term', []):
                    self.goals_display.insert(tk.END, f"- {action}\n")

            self.goals_display.configure(state=tk.DISABLED)

        except Exception as e:
            logging.error(f"Error updating state displays: {e}")

    def update_memory_displays(self):
        """Update the memory displays."""
        try:
            # Update working memory display
            working_memory = self.agent.memory.get_working_memory()

            self.working_memory_display.configure(state=tk.NORMAL)
            self.working_memory_display.delete(1.0, tk.END)

            self.working_memory_display.insert(tk.END, f"Working Memory Items ({len(working_memory)}):\n\n")

            for item in working_memory:
                item_type = item.get('type', 'unknown')
                timestamp = item.get('timestamp', 0)
                importance = item.get('importance', 0)

                time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')

                self.working_memory_display.insert(tk.END, f"[{time_str}] Type: {item_type} (Importance: {importance:.2f})\n")

                content = item.get('content', {})
                if isinstance(content, dict):
                    # For structured content, show a summary
                    if 'type' in content:
                        self.working_memory_display.insert(tk.END, f"Content type: {content.get('type')}\n")
                    if 'content' in content and isinstance(content['content'], str):
                        self.working_memory_display.insert(tk.END, f"Content: {content['content'][:100]}...\n")
                elif isinstance(content, str):
                    self.working_memory_display.insert(tk.END, f"Content: {content[:100]}...\n")

                self.working_memory_display.insert(tk.END, "\n")

            self.working_memory_display.configure(state=tk.DISABLED)

            # Update facts display
            facts = self.agent.memory.get_facts(limit=20)

            self.facts_display.configure(state=tk.NORMAL)
            self.facts_display.delete(1.0, tk.END)

            self.facts_display.insert(tk.END, f"Known Facts ({len(facts)}):\n\n")

            for fact in facts:
                entity = fact.get('entity', '')
                attribute = fact.get('attribute', '')
                value = fact.get('value', '')
                timestamp = fact.get('timestamp', 0)

                time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

                self.facts_display.insert(tk.END, f"[{time_str}] {entity}: {attribute} = {value}\n\n")

            self.facts_display.configure(state=tk.DISABLED)

            # Update perceptions display
            perceptions = self.agent.memory.get_recent_perceptions(limit=5)

            self.perceptions_display.configure(state=tk.NORMAL)
            self.perceptions_display.delete(1.0, tk.END)

            self.perceptions_display.insert(tk.END, f"Recent Perceptions ({len(perceptions)}):\n\n")

            for perception in perceptions:
                p_type = perception.get('type', 'unknown')
                timestamp = perception.get('timestamp', 0)

                time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')

                self.perceptions_display.insert(tk.END, f"[{time_str}] Type: {p_type}\n")

                # Try to display interpretation
                interp = perception.get('interpretation', '')
                if interp:
                    if isinstance(interp, str):
                        try:
                            # Try to parse as JSON
                            interp_data = json.loads(interp)
                            if 'description' in interp_data:
                                self.perceptions_display.insert(tk.END, f"Description: {interp_data['description']}\n")
                        except:
                            # Just show the string
                            self.perceptions_display.insert(tk.END, f"Interpretation: {interp[:100]}...\n")

                self.perceptions_display.insert(tk.END, "\n")

            self.perceptions_display.configure(state=tk.DISABLED)

        except Exception as e:
            logging.error(f"Error updating memory displays: {e}")

    def update_autonomy_displays(self):
        """Update the autonomy displays."""
        try:
            # Check if autonomy module is available
            if not hasattr(self.agent, 'autonomy') or self.agent.autonomy is None:
                self.topics_display.configure(state=tk.NORMAL)
                self.topics_display.delete(1.0, tk.END)
                self.topics_display.insert(tk.END, "Autonomy module not enabled.")
                self.topics_display.configure(state=tk.DISABLED)

                self.questions_display.configure(state=tk.NORMAL)
                self.questions_display.delete(1.0, tk.END)
                self.questions_display.insert(tk.END, "Autonomy module not enabled.")
                self.questions_display.configure(state=tk.DISABLED)

                self.thinking_mode_display.configure(state=tk.NORMAL)
                self.thinking_mode_display.delete(1.0, tk.END)
                self.thinking_mode_display.insert(tk.END, "Autonomy module not enabled.")
                self.thinking_mode_display.configure(state=tk.DISABLED)

                return

            # Update topics display
            topics = self.agent.autonomy.get_topics_of_interest()

            self.topics_display.configure(state=tk.NORMAL)
            self.topics_display.delete(1.0, tk.END)

            self.topics_display.insert(tk.END, "Topics of Interest:\n\n")

            # Sort topics by interest level
            sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)

            for topic, score in sorted_topics:
                bar_length = int(score * 20)  # Scale to 20 characters
                bar = '█' * bar_length + '░' * (20 - bar_length)
                self.topics_display.insert(tk.END, f"{topic.replace('_', ' ').capitalize()}: {score:.2f} {bar}\n")

            self.topics_display.configure(state=tk.DISABLED)

            # Update questions display
            questions = self.agent.autonomy.get_current_questions()

            self.questions_display.configure(state=tk.NORMAL)
            self.questions_display.delete(1.0, tk.END)

            self.questions_display.insert(tk.END, "Current Questions:\n\n")

            for i, question in enumerate(questions):
                self.questions_display.insert(tk.END, f"{i+1}. {question}\n\n")

            self.questions_display.configure(state=tk.DISABLED)

            # Update thinking mode display
            current_mode = self.agent.autonomy.get_current_thinking_mode()

            self.thinking_mode_display.configure(state=tk.NORMAL)
            self.thinking_mode_display.delete(1.0, tk.END)

            self.thinking_mode_display.insert(tk.END, "Current Thinking Mode:\n\n")
            self.thinking_mode_display.insert(tk.END, f"{current_mode.capitalize()}\n\n")

            # Add description based on mode
            if current_mode == 'analytical':
                description = "Analyzing patterns, drawing logical conclusions, and examining evidence rationally."
            elif current_mode == 'creative':
                description = "Exploring novel connections, imaginative possibilities, and unconventional perspectives."
            elif current_mode == 'reflective':
                description = "Considering past experiences, evaluating self-knowledge, and contemplating deeper meanings."
            elif current_mode == 'exploratory':
                description = "Investigating new concepts, formulating questions, and seeking to expand understanding."
            elif current_mode == 'emotional':
                description = "Processing feelings, examining motivations, and exploring emotional responses."
            elif current_mode == 'predictive':
                description = "Forecasting possibilities, anticipating outcomes, and projecting future scenarios."
            else:
                description = "Engaging in general thinking processes."

            self.thinking_mode_display.insert(tk.END, f"Description: {description}\n\n")

            # Add all available modes
            self.thinking_mode_display.insert(tk.END, "Available Thinking Modes:\n\n")
            all_modes = self.agent.autonomy.thinking_modes
            for mode in all_modes:
                if mode == current_mode:
                    self.thinking_mode_display.insert(tk.END, f"• {mode.capitalize()} (ACTIVE)\n")
                else:
                    self.thinking_mode_display.insert(tk.END, f"• {mode.capitalize()}\n")

            self.thinking_mode_display.configure(state=tk.DISABLED)

        except Exception as e:
            logging.error(f"Error updating autonomy displays: {e}")

    def update_camera_from_agent(self):
        """Update camera feed from agent's camera sensor if available."""
        try:
            # Get camera data from agent
            webcam_data = None
            sensor_data = self.agent.sensors.get_data('webcam')

            if sensor_data and 'description' in sensor_data:
                # Update camera description
                self.camera_description.configure(state=tk.NORMAL)
                self.camera_description.delete(1.0, tk.END)
                self.camera_description.insert(tk.END, f"Camera Description: {sensor_data['description']}")
                self.camera_description.configure(state=tk.DISABLED)

                # Update camera placeholder with description
                self.update_camera_feed(None, sensor_data['description'])
        except Exception as e:
            logging.error(f"Error updating camera feed: {e}")

    def update_camera_feed(self, frame=None, description=None):
        """
        Update the camera feed display.

        Args:
            frame: CV2 frame or None for placeholder
            description: Optional description text
        """
        try:
            if frame is not None:
                # Convert OpenCV frame to Tkinter-compatible image
                cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv2_img)
                tk_img = ImageTk.PhotoImage(image=pil_img)

                # Update label
                self.camera_label.configure(image=tk_img)
                self.camera_label.image = tk_img  # Keep a reference
            else:
                # Create a placeholder image with description
                width, height = 320, 240
                pil_img = Image.new('RGB', (width, height), color=(240, 240, 240))

                # Add description text if available

                if description:
                    from PIL import ImageDraw, ImageFont
                    draw = ImageDraw.Draw(pil_img)

                    # Wrap text to fit in the image
                    wrapped_text = []
                    words = description.split()
                    line = ""
                    for word in words:
                        test_line = line + word + " "
                        # Check if line is too long
                        if draw.textlength(test_line, font=None) > width - 20:
                            wrapped_text.append(line)
                            line = word + " "
                        else:
                            line = test_line
                    wrapped_text.append(line)

                    # Draw text
                    y_position = 20
                    for line in wrapped_text:
                        draw.text((10, y_position), line, fill=(0, 0, 0))
                        y_position += 20

                # Convert to Tkinter image
                tk_img = ImageTk.PhotoImage(image=pil_img)

                # Update label
                self.camera_label.configure(image=tk_img)
                self.camera_label.image = tk_img  # Keep a reference

        except Exception as e:
            logging.error(f"Error updating camera feed: {e}")
            # Create error placeholder
            pil_img = Image.new('RGB', (320, 240), color=(240, 200, 200))

            try:
                # Add error text
                from PIL import ImageDraw
                draw = ImageDraw.Draw(pil_img)
                draw.text((10, 100), "Error loading camera feed", fill=(255, 0, 0))

                # Convert to Tkinter image
                tk_img = ImageTk.PhotoImage(image=pil_img)

                # Update label
                self.camera_label.configure(image=tk_img)
                self.camera_label.image = tk_img  # Keep a reference
            except:
                # Last resort fallback
                self.camera_label.configure(image='', text="Camera error")

    def send_message(self):
        """Send a message to the agent."""
        # Get message text
        message_text = self.message_input.get(1.0, tk.END).strip()

        # Skip if message is empty
        if not message_text:
            return

        # Clear input
        self.message_input.delete(1.0, tk.END)

        # Display message in chat
        self.display_human_message(message_text)

        # Send to agent
        self.display_system_message("Sending message to agent...")

        message = {
            'id': f"msg_{int(time.time())}",
            'sender': 'human',
            'content': message_text,
            'timestamp': time.time()
        }

        # Send to agent's consciousness
        self.agent.consciousness.receive_message(message)

    def send_message_event(self, event):
        """Handle Enter key for sending message."""
        # Don't add newline
        if not event.state & 0x1:  # Check if shift is not pressed
            self.send_message()
            return "break"  # Prevent default behavior

    def display_human_message(self, message):
        """Display a message from the human in the chat."""
        # Enable editing
        self.chat_display.configure(state=tk.NORMAL)

        # Add timestamp and sender
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.chat_display.insert(tk.END, f"[{timestamp}] You: ", "human")

        # Add message content
        self.chat_display.insert(tk.END, f"{message}\n\n")

        # Disable editing
        self.chat_display.configure(state=tk.DISABLED)

        # Scroll to bottom
        self.chat_display.see(tk.END)

    def display_agent_message(self, message):
        """Display a message from the agent in the chat."""
        # Enable editing
        self.chat_display.configure(state=tk.NORMAL)

        # Add timestamp and sender
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.chat_display.insert(tk.END, f"[{timestamp}] Agent: ", "agent")

        # Add message content
        self.chat_display.insert(tk.END, f"{message.get('content', '')}\n", "agent")

        # Add thought if available
        thought = message.get('thought')
        if thought:
            self.chat_display.insert(tk.END, f"(Internal thought: {thought})\n", "thought")

        # Add extra newline
        self.chat_display.insert(tk.END, "\n")

        # Disable editing
        self.chat_display.configure(state=tk.DISABLED)

        # Scroll to bottom
        self.chat_display.see(tk.END)

    def display_system_message(self, message):
        """Display a system message in the chat."""
        # Enable editing
        self.chat_display.configure(state=tk.NORMAL)

        # Add message
        self.chat_display.insert(tk.END, f"{message}\n", "system")

        # Disable editing
        self.chat_display.configure(state=tk.DISABLED)

        # Scroll to bottom
        self.chat_display.see(tk.END)

    def on_close(self):
        """Handle window close event."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.running = False
            self.master.destroy()


def main():
    """Main entry point for the GUI application."""
    parser = argparse.ArgumentParser(description="GUI for Embodied AI Agent")
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--hardware",
        choices=["mac", "robot"],
        default="mac",
        help="Hardware platform to run on"
    )
    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting Agent GUI")

    try:
        # Load configuration
        config = load_config(args.config)

        # Override hardware platform from command line if specified
        if args.hardware:
            config['hardware']['platform'] = args.hardware

        # Initialize Tkinter
        root = tk.Tk()

        # Create and initialize agent
        agent = EmbodiedAgent(config)

        # Start agent
        agent.start()
        logger.info("Agent started")

        # Create GUI
        gui = AgentGUI(root, agent)

        # Start GUI main loop
        root.mainloop()

        # Cleanup
        logger.info("Stopping agent")
        agent.stop()
        logger.info("Agent stopped")

    except Exception as e:
        logger.exception(f"Error in main: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
