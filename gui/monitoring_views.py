"""
Monitoring view panels for the Agent GUI.

Provides update methods for monologue, state, memory, and autonomy displays.
These are mixed into AgentGUI via composition.
"""

import json
import logging
import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox


def create_monologue_view(gui, parent):
    """Create the internal monologue view."""
    gui.monologue_display = scrolledtext.ScrolledText(parent, wrap=tk.WORD, state=tk.DISABLED)
    gui.monologue_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    gui.monologue_display.tag_configure("timestamp", foreground="gray")
    gui.monologue_display.tag_configure("thought", foreground="purple")


def create_state_view(gui, parent):
    """Create the agent state view."""
    state_notebook = ttk.Notebook(parent)
    state_notebook.pack(fill=tk.BOTH, expand=True)

    emotions_tab = ttk.Frame(state_notebook)
    state_notebook.add(emotions_tab, text="Emotions")
    gui.emotions_display = scrolledtext.ScrolledText(emotions_tab, wrap=tk.WORD, state=tk.DISABLED)
    gui.emotions_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    drives_tab = ttk.Frame(state_notebook)
    state_notebook.add(drives_tab, text="Drives")
    gui.drives_display = scrolledtext.ScrolledText(drives_tab, wrap=tk.WORD, state=tk.DISABLED)
    gui.drives_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    goals_tab = ttk.Frame(state_notebook)
    state_notebook.add(goals_tab, text="Goals")
    gui.goals_display = scrolledtext.ScrolledText(goals_tab, wrap=tk.WORD, state=tk.DISABLED)
    gui.goals_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


def create_memory_view(gui, parent):
    """Create the memory view."""
    memory_notebook = ttk.Notebook(parent)
    memory_notebook.pack(fill=tk.BOTH, expand=True)

    working_memory_tab = ttk.Frame(memory_notebook)
    memory_notebook.add(working_memory_tab, text="Working Memory")
    gui.working_memory_display = scrolledtext.ScrolledText(working_memory_tab, wrap=tk.WORD, state=tk.DISABLED)
    gui.working_memory_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    facts_tab = ttk.Frame(memory_notebook)
    memory_notebook.add(facts_tab, text="Facts")
    gui.facts_display = scrolledtext.ScrolledText(facts_tab, wrap=tk.WORD, state=tk.DISABLED)
    gui.facts_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    perceptions_tab = ttk.Frame(memory_notebook)
    memory_notebook.add(perceptions_tab, text="Perceptions")
    gui.perceptions_display = scrolledtext.ScrolledText(perceptions_tab, wrap=tk.WORD, state=tk.DISABLED)
    gui.perceptions_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    memory_mgmt_tab = ttk.Frame(memory_notebook)
    memory_notebook.add(memory_mgmt_tab, text="Memory Management")

    mgmt_frame = ttk.Frame(memory_mgmt_tab)
    mgmt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    warning_label = ttk.Label(
        mgmt_frame,
        text="Warning: Resetting memory will erase all stored facts, conversations, and experiences.",
        foreground="red",
        font=("Helvetica", 10, "bold"),
    )
    warning_label.pack(pady=10)

    gui.reset_button = ttk.Button(
        mgmt_frame, text="Reset Memory Database", command=gui._reset_memory_database
    )
    gui.reset_button.pack(pady=10)

    gui.memory_status_label = ttk.Label(mgmt_frame, text="Memory system status: Active")
    gui.memory_status_label.pack(pady=10)

    gui.memory_stats_display = scrolledtext.ScrolledText(mgmt_frame, wrap=tk.WORD, height=10)
    gui.memory_stats_display.pack(fill=tk.X, expand=True, padx=5, pady=5)
    update_memory_stats(gui)


def create_autonomy_view(gui, parent):
    """Create the autonomy view."""
    autonomy_notebook = ttk.Notebook(parent)
    autonomy_notebook.pack(fill=tk.BOTH, expand=True)

    topics_tab = ttk.Frame(autonomy_notebook)
    autonomy_notebook.add(topics_tab, text="Topics of Interest")
    gui.topics_display = scrolledtext.ScrolledText(topics_tab, wrap=tk.WORD, state=tk.DISABLED)
    gui.topics_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    questions_tab = ttk.Frame(autonomy_notebook)
    autonomy_notebook.add(questions_tab, text="Questions")
    gui.questions_display = scrolledtext.ScrolledText(questions_tab, wrap=tk.WORD, state=tk.DISABLED)
    gui.questions_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    thinking_tab = ttk.Frame(autonomy_notebook)
    autonomy_notebook.add(thinking_tab, text="Thinking Mode")
    gui.thinking_mode_display = scrolledtext.ScrolledText(thinking_tab, wrap=tk.WORD, state=tk.DISABLED)
    gui.thinking_mode_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


# --- Update functions ---


def update_monologue_display(gui):
    """Update the internal monologue display."""
    monologue = gui.agent.consciousness.get_internal_monologue()
    if len(monologue) > 0:
        gui.monologue_display.configure(state=tk.NORMAL)
        gui.monologue_display.delete(1.0, tk.END)
        for entry in monologue:
            parts = entry.split('] ', 1)
            if len(parts) == 2:
                timestamp = parts[0] + ']'
                thought = parts[1]
                gui.monologue_display.insert(tk.END, timestamp + ' ', "timestamp")
                gui.monologue_display.insert(tk.END, thought + '\n', "thought")
            else:
                gui.monologue_display.insert(tk.END, entry + '\n')
        gui.monologue_display.configure(state=tk.DISABLED)
        gui.monologue_display.see(tk.END)


def update_state_displays(gui):
    """Update the agent state displays."""
    try:
        internal_state = gui.agent.cognition.get_internal_state()

        gui.emotions_display.configure(state=tk.NORMAL)
        gui.emotions_display.delete(1.0, tk.END)
        gui.emotions_display.insert(tk.END, "Current Emotional State:\n\n")
        for emotion, value in internal_state.get('emotions', {}).items():
            bar_length = int(value * 20)
            bar = '\u2588' * bar_length + '\u2591' * (20 - bar_length)
            gui.emotions_display.insert(tk.END, f"{emotion.capitalize()}: {value:.2f} {bar}\n")
        gui.emotions_display.configure(state=tk.DISABLED)

        gui.drives_display.configure(state=tk.NORMAL)
        gui.drives_display.delete(1.0, tk.END)
        gui.drives_display.insert(tk.END, "Current Drives:\n\n")
        for drive, value in internal_state.get('drives', {}).items():
            bar_length = int(value * 20)
            bar = '\u2588' * bar_length + '\u2591' * (20 - bar_length)
            gui.drives_display.insert(tk.END, f"{drive.capitalize()}: {value:.2f} {bar}\n")
        gui.drives_display.configure(state=tk.DISABLED)

        gui.goals_display.configure(state=tk.NORMAL)
        gui.goals_display.delete(1.0, tk.END)
        gui.goals_display.insert(tk.END, "Current Goals:\n\n")
        for i, goal in enumerate(internal_state.get('current_goals', [])):
            gui.goals_display.insert(tk.END, f"{i+1}. {goal}\n")
        current_plan = internal_state.get('current_plan', {})
        if current_plan:
            gui.goals_display.insert(tk.END, "\nCurrent Plan:\n\n")
            for label, key in [("Short-term", "short_term"), ("Medium-term", "medium_term"), ("Long-term", "long_term")]:
                gui.goals_display.insert(tk.END, f"{label}:\n")
                for action in current_plan.get(key, []):
                    gui.goals_display.insert(tk.END, f"- {action}\n")
                gui.goals_display.insert(tk.END, "\n")
        gui.goals_display.configure(state=tk.DISABLED)

    except Exception as e:
        logging.error(f"Error updating state displays: {e}")


def update_memory_displays(gui):
    """Update the memory displays."""
    try:
        working_memory = gui.agent.memory.get_working_memory()
        gui.working_memory_display.configure(state=tk.NORMAL)
        gui.working_memory_display.delete(1.0, tk.END)
        gui.working_memory_display.insert(tk.END, f"Working Memory Items ({len(working_memory)}):\n\n")
        for item in working_memory:
            item_type = item.get('type', 'unknown')
            timestamp = item.get('timestamp', 0)
            importance = item.get('importance', 0)
            time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
            gui.working_memory_display.insert(tk.END, f"[{time_str}] Type: {item_type} (Importance: {importance:.2f})\n")
            content = item.get('content', {})
            if isinstance(content, dict):
                if 'type' in content:
                    gui.working_memory_display.insert(tk.END, f"Content type: {content.get('type')}\n")
                if 'content' in content and isinstance(content['content'], str):
                    gui.working_memory_display.insert(tk.END, f"Content: {content['content'][:100]}...\n")
            elif isinstance(content, str):
                gui.working_memory_display.insert(tk.END, f"Content: {content[:100]}...\n")
            gui.working_memory_display.insert(tk.END, "\n")
        gui.working_memory_display.configure(state=tk.DISABLED)

        facts = gui.agent.memory.get_facts(limit=20)
        gui.facts_display.configure(state=tk.NORMAL)
        gui.facts_display.delete(1.0, tk.END)
        gui.facts_display.insert(tk.END, f"Known Facts ({len(facts)}):\n\n")
        for fact in facts:
            entity = fact.get('entity', '')
            attribute = fact.get('attribute', '')
            value = fact.get('value', '')
            timestamp = fact.get('timestamp', 0)
            time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            gui.facts_display.insert(tk.END, f"[{time_str}] {entity}: {attribute} = {value}\n\n")
        gui.facts_display.configure(state=tk.DISABLED)

        perceptions = gui.agent.memory.get_recent_perceptions(limit=5)
        gui.perceptions_display.configure(state=tk.NORMAL)
        gui.perceptions_display.delete(1.0, tk.END)
        gui.perceptions_display.insert(tk.END, f"Recent Perceptions ({len(perceptions)}):\n\n")
        for perception in perceptions:
            p_type = perception.get('type', 'unknown')
            timestamp = perception.get('timestamp', 0)
            time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
            gui.perceptions_display.insert(tk.END, f"[{time_str}] Type: {p_type}\n")
            interp = perception.get('interpretation', '')
            if interp:
                if isinstance(interp, str):
                    try:
                        interp_data = json.loads(interp)
                        if 'description' in interp_data:
                            gui.perceptions_display.insert(tk.END, f"Description: {interp_data['description']}\n")
                    except json.JSONDecodeError:
                        gui.perceptions_display.insert(tk.END, f"Interpretation: {interp[:100]}...\n")
            gui.perceptions_display.insert(tk.END, "\n")
        gui.perceptions_display.configure(state=tk.DISABLED)

    except Exception as e:
        logging.error(f"Error updating memory displays: {e}")


def update_autonomy_displays(gui):
    """Update the autonomy displays."""
    try:
        if not hasattr(gui.agent, 'autonomy') or gui.agent.autonomy is None:
            for display in [gui.topics_display, gui.questions_display, gui.thinking_mode_display]:
                display.configure(state=tk.NORMAL)
                display.delete(1.0, tk.END)
                display.insert(tk.END, "Autonomy module not enabled.")
                display.configure(state=tk.DISABLED)
            return

        topics = gui.agent.autonomy.get_topics_of_interest()
        gui.topics_display.configure(state=tk.NORMAL)
        gui.topics_display.delete(1.0, tk.END)
        gui.topics_display.insert(tk.END, "Topics of Interest:\n\n")
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        for topic, score in sorted_topics:
            bar_length = int(score * 20)
            bar = '\u2588' * bar_length + '\u2591' * (20 - bar_length)
            gui.topics_display.insert(tk.END, f"{topic.replace('_', ' ').capitalize()}: {score:.2f} {bar}\n")
        gui.topics_display.configure(state=tk.DISABLED)

        questions = gui.agent.autonomy.get_current_questions()
        gui.questions_display.configure(state=tk.NORMAL)
        gui.questions_display.delete(1.0, tk.END)
        gui.questions_display.insert(tk.END, "Current Questions:\n\n")
        for i, question in enumerate(questions):
            gui.questions_display.insert(tk.END, f"{i+1}. {question}\n\n")
        gui.questions_display.configure(state=tk.DISABLED)

        _update_thinking_mode(gui)

    except Exception as e:
        logging.error(f"Error updating autonomy displays: {e}")


def _update_thinking_mode(gui):
    """Update the thinking mode display."""
    current_mode = gui.agent.autonomy.get_current_thinking_mode()
    gui.thinking_mode_display.configure(state=tk.NORMAL)
    gui.thinking_mode_display.delete(1.0, tk.END)
    gui.thinking_mode_display.insert(tk.END, "Current Thinking Mode:\n\n")
    gui.thinking_mode_display.insert(tk.END, f"{current_mode.capitalize()}\n\n")

    descriptions = {
        'analytical': "Analyzing patterns, drawing logical conclusions, and examining evidence rationally.",
        'creative': "Exploring novel connections, imaginative possibilities, and unconventional perspectives.",
        'reflective': "Considering past experiences, evaluating self-knowledge, and contemplating deeper meanings.",
        'exploratory': "Investigating new concepts, formulating questions, and seeking to expand understanding.",
        'emotional': "Processing feelings, examining motivations, and exploring emotional responses.",
        'predictive': "Forecasting possibilities, anticipating outcomes, and projecting future scenarios.",
    }
    description = descriptions.get(current_mode, "Engaging in general thinking processes.")
    gui.thinking_mode_display.insert(tk.END, f"Description: {description}\n\n")

    gui.thinking_mode_display.insert(tk.END, "Available Thinking Modes:\n\n")
    all_modes = gui.agent.autonomy.thinking_modes
    for mode in all_modes:
        suffix = " (ACTIVE)" if mode == current_mode else ""
        gui.thinking_mode_display.insert(tk.END, f"\u2022 {mode.capitalize()}{suffix}\n")
    gui.thinking_mode_display.configure(state=tk.DISABLED)


def update_memory_stats(gui):
    """Update memory statistics display."""
    try:
        stats = gui.agent.memory.get_memory_stats()
        gui.memory_stats_display.config(state=tk.NORMAL)
        gui.memory_stats_display.delete(1.0, tk.END)
        gui.memory_stats_display.insert(tk.END, "Memory System Statistics:\n\n")
        if stats:
            for category, count in stats.items():
                gui.memory_stats_display.insert(tk.END, f"{category.capitalize()}: {count}\n")
        else:
            gui.memory_stats_display.insert(tk.END, "No memory statistics available.")
        gui.memory_stats_display.config(state=tk.DISABLED)
    except Exception as e:
        logging.error(f"Error updating memory stats: {e}")
