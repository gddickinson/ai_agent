"""The sensor layer imports and simulates without OpenCV (only a real camera needs cv2)."""
import importlib
import sys


def test_simulated_sensors_without_cv2(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cv2', None)  # makes 'import cv2' raise ImportError
    import hardware.sensors as sensors
    sensors = importlib.reload(sensors)
    try:
        assert sensors.cv2 is None
        cam = sensors.CameraSensor({'name': 'webcam', 'type': 'camera', 'simulate': False, 'frame_interval': 0})
        cam.start()
        assert cam.simulate is True
        data = cam.get_data()
        assert data['simulated'] is True and data['description']
        assert cam.test_camera_connection()['success'] is False
        cam.stop()
    finally:
        monkeypatch.delitem(sys.modules, 'cv2')
        importlib.reload(sensors)
