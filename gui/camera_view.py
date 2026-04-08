"""
Camera view panel for the Agent GUI.

Handles camera feed display, snapshot capture, and vision model processing.
"""

import time
import logging
import base64
import tkinter as tk
from tkinter import ttk, scrolledtext

import cv2
from PIL import Image, ImageTk


def create_camera_view(gui, parent):
    """Create the camera feed view."""
    control_frame = ttk.Frame(parent)
    control_frame.pack(fill=tk.X, padx=5, pady=5)

    gui.camera_button = ttk.Button(control_frame, text="Stop Camera", command=lambda: toggle_camera(gui))
    gui.camera_button.pack(side=tk.LEFT, padx=5, pady=5)

    gui.snapshot_button = ttk.Button(control_frame, text="Take Snapshot", command=lambda: take_snapshot(gui))
    gui.snapshot_button.pack(side=tk.LEFT, padx=5, pady=5)

    gui.camera_label = ttk.Label(parent)
    gui.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    gui.camera_description = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=5, state=tk.DISABLED)
    gui.camera_description.pack(fill=tk.X, padx=5, pady=5)

    gui.camera_active = True
    gui.current_frame = None
    update_camera_feed(gui, None)


def toggle_camera(gui):
    """Toggle the camera feed on/off."""
    if gui.camera_active:
        gui.camera_active = False
        gui.camera_button.configure(text="Start Camera")
        update_camera_feed(gui, None, "Camera feed paused")
    else:
        gui.camera_active = True
        gui.camera_button.configure(text="Stop Camera")
        update_camera_from_agent(gui)


def take_snapshot(gui):
    """Take a snapshot of the current camera feed and process it with vision model."""
    if hasattr(gui, 'current_frame') and gui.current_frame is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        try:
            cv2.imwrite(filename, gui.current_frame)
            gui.display_system_message(f"Snapshot saved as {filename}")
            if gui.agent:
                gui.display_system_message("Processing image with vision model...")
                gui.display_system_message("Note: Vision processing may take some time or time out on resource-constrained systems.")
                _, buffer = cv2.imencode('.jpg', gui.current_frame)
                base64_image = base64.b64encode(buffer).decode('utf-8')
                snapshot_message = {
                    'id': f"snapshot_{int(time.time())}",
                    'sender': 'human',
                    'content': "I just took a snapshot. Can you describe what you see in the image?",
                    'timestamp': time.time(),
                    'snapshot': True,
                    'processing': True,
                }
                gui.agent.consciousness.receive_snapshot_message(snapshot_message, base64_image)
        except Exception as e:
            gui.display_system_message(f"Error saving or processing snapshot: {e}")
    else:
        gui.display_system_message("No camera frame available for snapshot")


def update_camera_from_agent(gui):
    """Update camera feed from agent's camera sensor if available."""
    try:
        if not gui.camera_active:
            return
        sensor_data = gui.agent.sensors.get_data('webcam')
        if sensor_data:
            if sensor_data.get('simulated', False):
                if 'description' in sensor_data:
                    gui.camera_description.configure(state=tk.NORMAL)
                    gui.camera_description.delete(1.0, tk.END)
                    gui.camera_description.insert(tk.END, f"Camera Description: {sensor_data['description']}")
                    gui.camera_description.configure(state=tk.DISABLED)
                    update_camera_feed(gui, None, sensor_data['description'])
            else:
                frame = sensor_data.get('frame')
                description = sensor_data.get('description')
                if frame is not None:
                    gui.current_frame = frame
                    update_camera_feed(gui, frame, description)
                    if description:
                        gui.camera_description.configure(state=tk.NORMAL)
                        gui.camera_description.delete(1.0, tk.END)
                        gui.camera_description.insert(tk.END, f"Camera Description: {description}")
                        gui.camera_description.configure(state=tk.DISABLED)
    except Exception as e:
        logging.error(f"Error updating camera feed: {e}")


def update_camera_feed(gui, frame=None, description=None):
    """
    Update the camera feed display.

    Args:
        gui: AgentGUI instance
        frame: CV2 frame or None for placeholder
        description: Optional description text
    """
    try:
        if frame is not None:
            cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            display_width = 480
            display_height = 360
            h, w = frame.shape[:2]
            aspect = w / h
            if aspect > display_width / display_height:
                resize_width = display_width
                resize_height = int(display_width / aspect)
            else:
                resize_height = display_height
                resize_width = int(display_height * aspect)
            cv2_img_resized = cv2.resize(cv2_img, (resize_width, resize_height))
            pil_img = Image.fromarray(cv2_img_resized)
            tk_img = ImageTk.PhotoImage(image=pil_img)
            gui.camera_label.configure(image=tk_img)
            gui.camera_label.image = tk_img
            gui.current_frame = frame
            if description:
                gui.camera_description.configure(state=tk.NORMAL)
                gui.camera_description.delete(1.0, tk.END)
                gui.camera_description.insert(tk.END, f"Camera Description: {description}")
                gui.camera_description.configure(state=tk.DISABLED)
        else:
            width, height = 480, 360
            pil_img = Image.new('RGB', (width, height), color=(240, 240, 240))
            if description:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(pil_img)
                wrapped_text = []
                words = description.split()
                line = ""
                for word in words:
                    test_line = line + word + " "
                    if draw.textlength(test_line, font=None) > width - 20:
                        wrapped_text.append(line)
                        line = word + " "
                    else:
                        line = test_line
                wrapped_text.append(line)
                y_position = 20
                for line_text in wrapped_text:
                    draw.text((10, y_position), line_text, fill=(0, 0, 0))
                    y_position += 20
            tk_img = ImageTk.PhotoImage(image=pil_img)
            gui.camera_label.configure(image=tk_img)
            gui.camera_label.image = tk_img
            gui.current_frame = None
    except Exception as e:
        logging.error(f"Error updating camera feed: {e}")
        pil_img = Image.new('RGB', (480, 360), color=(240, 200, 200))
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(pil_img)
            draw.text((10, 100), f"Error loading camera feed: {str(e)[:50]}", fill=(255, 0, 0))
            tk_img = ImageTk.PhotoImage(image=pil_img)
            gui.camera_label.configure(image=tk_img)
            gui.camera_label.image = tk_img
        except Exception:
            gui.camera_label.configure(image='', text="Camera error")
