"""
Main Agent GUI application.

Provides the AgentGUI class and main() entry point for the Tkinter-based interface.
"""

import sys
import os
import time
import queue
import logging
import argparse
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import EmbodiedAgent
from utils.config import load_config
from utils.logging import setup_logging

from gui import monitoring_views
from gui import camera_view


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
        self.logger = logging.getLogger(__name__)
        self.response_queue: queue.Queue = queue.Queue()
        self.master.title("Embodied AI Agent Interface")
        self.master.geometry("1200x800")
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        self.setup_styles()

        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        self.left_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_panel, weight=2)

        self.right_panel = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_panel, weight=1)

        self._create_chat_interface(self.left_panel)
        self._create_monitoring_interface(self.right_panel)
        self._register_agent_callbacks()
        self._setup_timers()

        self.running = True

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_styles(self):
        """Setup custom styles for the interface."""
        self.style = ttk.Style()
        self.style.configure("Human.TLabel", foreground="blue", font=("Helvetica", 10, "bold"))
        self.style.configure("Agent.TLabel", foreground="green", font=("Helvetica", 10, "bold"))
        self.style.configure("System.TLabel", foreground="gray", font=("Helvetica", 9))
        self.style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))
        self.style.configure("Send.TButton", font=("Helvetica", 10, "bold"))
        self.style.configure("TNotebook", padding=5)
        self.style.configure("TNotebook.Tab", font=("Helvetica", 9, "bold"))

    def _create_chat_interface(self, parent):
        """Create the chat interface in the left panel."""
        chat_frame = ttk.LabelFrame(parent, text="Conversation")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state=tk.DISABLED, height=20)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_display.tag_configure("human", foreground="blue")
        self.chat_display.tag_configure("agent", foreground="green")
        self.chat_display.tag_configure("system", foreground="gray", font=("Helvetica", 9))
        self.chat_display.tag_configure("thought", foreground="purple", font=("Helvetica", 9, "italic"))

        input_frame = ttk.Frame(parent)
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.message_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=3)
        self.message_input.pack(fill=tk.X, side=tk.LEFT, expand=True, padx=5, pady=5)
        self.message_input.bind("<Return>", self._send_message_event)
        self.message_input.bind("<Shift-Return>", lambda e: None)

        self.send_button = ttk.Button(input_frame, text="Send", style="Send.TButton", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def _create_monitoring_interface(self, parent):
        """Create the monitoring interface in the right panel."""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.monologue_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.monologue_tab, text="Internal Monologue")
        monitoring_views.create_monologue_view(self, self.monologue_tab)

        self.camera_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.camera_tab, text="Camera Feed")
        camera_view.create_camera_view(self, self.camera_tab)

        self.state_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.state_tab, text="Agent State")
        monitoring_views.create_state_view(self, self.state_tab)

        self.memory_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.memory_tab, text="Memory")
        monitoring_views.create_memory_view(self, self.memory_tab)

        self.autonomy_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.autonomy_tab, text="Autonomy")
        monitoring_views.create_autonomy_view(self, self.autonomy_tab)

    def _register_agent_callbacks(self):
        """Register callbacks for agent events."""
        self.agent.consciousness.register_message_callback('sent', self._message_sent_callback)

    def _message_sent_callback(self, message):
        """Callback for when a message is sent by the agent."""
        self.response_queue.put(message)

    def _setup_timers(self):
        """Setup update timers for GUI components."""
        self.master.after(100, self._check_message_queue)
        self.master.after(1000, self._update_monitoring_views)

    # ------------------------------------------------------------------
    # Update loops
    # ------------------------------------------------------------------

    def _check_message_queue(self):
        """Check for new messages in the response queue."""
        try:
            while True:
                message = self.response_queue.get_nowait()
                self.display_agent_message(message)
                self.response_queue.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            logging.error(f"Error checking message queue: {e}")
        if self.running:
            self.master.after(100, self._check_message_queue)

    def _update_monitoring_views(self):
        """Update all monitoring views."""
        try:
            monitoring_views.update_monologue_display(self)
            monitoring_views.update_state_displays(self)
            monitoring_views.update_memory_displays(self)
            monitoring_views.update_autonomy_displays(self)
            camera_view.update_camera_from_agent(self)
        except Exception as e:
            logging.error(f"Error updating monitoring views: {e}")
        if self.running:
            self.master.after(1000, self._update_monitoring_views)

    # ------------------------------------------------------------------
    # Chat messaging
    # ------------------------------------------------------------------

    def send_message(self):
        """Send a message to the agent."""
        message_text = self.message_input.get(1.0, tk.END).strip()
        if not message_text:
            return
        self.message_input.delete(1.0, tk.END)
        self.display_human_message(message_text)
        self.display_system_message("Sending message to agent...")
        message = {
            'id': f"msg_{int(time.time())}",
            'sender': 'human',
            'content': message_text,
            'timestamp': time.time(),
        }
        self.agent.consciousness.receive_message(message)

    def _send_message_event(self, event):
        """Handle Enter key for sending message."""
        if not event.state & 0x1:
            self.send_message()
            return "break"

    def display_human_message(self, message: str):
        """Display a message from the human in the chat."""
        self.chat_display.configure(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.chat_display.insert(tk.END, f"[{timestamp}] You: ", "human")
        self.chat_display.insert(tk.END, f"{message}\n\n")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def display_agent_message(self, message: Dict[str, Any]):
        """Display a message from the agent in the chat."""
        self.chat_display.configure(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.chat_display.insert(tk.END, f"[{timestamp}] Agent: ", "agent")
        self.chat_display.insert(tk.END, f"{message.get('content', '')}\n", "agent")
        thought = message.get('thought')
        if thought:
            self.chat_display.insert(tk.END, f"(Internal thought: {thought})\n", "thought")
        self.chat_display.insert(tk.END, "\n")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def display_system_message(self, message: str):
        """Display a system message in the chat."""
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{message}\n", "system")
        self.chat_display.configure(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def _reset_memory_database(self):
        """Reset the memory database after confirmation."""
        if messagebox.askyesno(
            "Reset Memory Database",
            "Are you sure you want to reset the memory database? "
            "This will erase all stored facts, conversations, and experiences.",
        ):
            try:
                success = self.agent.memory.reset_database()
                if success:
                    messagebox.showinfo("Success", "Memory database has been reset successfully.")
                    self.display_system_message("Memory database has been reset.")
                    monitoring_views.update_memory_stats(self)
                else:
                    messagebox.showerror("Error", "Failed to reset memory database.")
            except Exception as e:
                messagebox.showerror("Error", f"Error resetting memory database: {e}")

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def on_close(self):
        """Handle window close event."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.running = False
            self.master.destroy()


def main():
    """Main entry point for the GUI application."""
    parser = argparse.ArgumentParser(description="GUI for Embodied AI Agent")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--hardware", choices=["mac", "robot"], default="mac", help="Hardware platform to run on")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting Agent GUI")

    try:
        config = load_config(args.config)
        if args.hardware:
            config['hardware']['platform'] = args.hardware

        root = tk.Tk()
        agent = EmbodiedAgent(config)
        agent.start()
        logger.info("Agent started")

        gui = AgentGUI(root, agent)
        root.mainloop()

        logger.info("Stopping agent")
        agent.stop()
        logger.info("Agent stopped")

    except Exception as e:
        logger.exception(f"Error in main: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
