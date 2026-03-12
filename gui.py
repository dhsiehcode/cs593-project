

'''
import sys
import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox



class CamViewer(QWidget):
    def __init__(self, camera_name: int = 0, fps: int = 25):
        super().__init__()
        self.setWindowTitle("Webcam Stream - QTimer")
        self.camera_name = camera_name
        self.frame_interval_ms = int(1000 / max(1, fps))

## only main thread

# filename: webcam_qt_timer.py
import sys
import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox


class WebcamViewer(QWidget):
    def __init__(self, camera_index: int = 0, fps: int = 30):
        super().__init__()
        self.setWindowTitle("Webcam Stream - QTimer")
        self.camera_index = camera_index
        self.frame_interval_ms = int(1000 / max(1, fps))

        # UI
        self.label = QLabel("Starting camera...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(640, 480)
        self.btn_toggle = QPushButton("Stop")
        self.btn_toggle.clicked.connect(self.toggle_stream)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.btn_toggle)
        self.setLayout(layout)

        # OpenCV capture
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows; safe on others
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", f"Cannot open camera index {self.camera_index}")
            sys.exit(1)

        # Timer to grab frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.frame_interval_ms)

    def toggle_stream(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_toggle.setText("Start")
        else:
            self.timer.start(self.frame_interval_ms)
            self.btn_toggle.setText("Stop")

    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return  # could also show a message or try to reconnect

        # Convert BGR (OpenCV) -> RGB (Qt), then to QImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Optionally scale to label size while keeping aspect ratio
        pix = QPixmap.fromImage(qimg).scaled(
            self.label.width(), self.label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.label.setPixmap(pix)

    def closeEvent(self, event):
        # Cleanup
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = WebcamViewer(camera_index=0, fps=30)
    w.show()
    sys.exit(app.exec())
'''

# filename: webcam_qt_thread.py

## threaded
import sys
import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QMessageBox, QHBoxLayout
from ultralytics import YOLO
from mapping import Mapper


class CameraWorker(QThread):
    frameReady = pyqtSignal(object)   # emits numpy array (BGR)
    cameraError = pyqtSignal(str)

    def __init__(self, camera_index: int = 0, fps: int = 15):
        super().__init__()
        self.camera_index = camera_index
        self.fps = fps
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.cameraError.emit(f"Cannot open camera index {self.camera_index}")
            return

        delay_ms = max(1, int(1000 / self.fps))
        while self._running:
            ok, frame = cap.read()
            if not ok or frame is None:
                # Emit an error or continue (some webcams glitch occasionally)
                continue
            self.frameReady.emit(frame)
            # QThread.msleep uses milliseconds (static method)
            QThread.msleep(delay_ms)

        cap.release()

    def stop(self):
        self._running = False
        self.wait(1000)  # wait up to 1s for clean exit


class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self.clicked.emit(int(pos.x()), int(pos.y()))
        super().mousePressEvent(event)


class WebcamViewer(QWidget):
    DEFAULT_FOV_X = 90
    DEFAULT_FOV_Y = 60
    DEFAULT_CAM_TO_ROBOT = 0.5

    def __init__(self, camera_index: int = 0, fps: int = 15, confidence: float = 0.7):
        super().__init__()
        self.setWindowTitle("Webcam Stream - QThread")
        self.label = ClickableLabel("Starting camera...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(640, 480)
        self.label.clicked.connect(self.on_label_clicked)
        self.btn_toggle = QPushButton("Stop")
        self.btn_toggle.clicked.connect(self.toggle_stream)
        self.btn_mode = QPushButton("Mode: Relative")
        self.btn_mode.clicked.connect(self.toggle_mode)

        self.sidebar_title = QLabel("Clicked Pixel")
        self.sidebar_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sidebar_coords = QLabel("x: -, y: -")
        self.sidebar_coords.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sidebar_sample = QLabel("color: -")
        self.sidebar_sample.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sidebar_box = QLabel("box: -")
        self.sidebar_box.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sidebar_move = QLabel("movement: -")
        self.sidebar_move.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sidebar_mode = QLabel("mode: get_relative_movement")
        self.sidebar_mode.setAlignment(Qt.AlignmentFlag.AlignLeft)

        sidebar = QVBoxLayout()
        sidebar.addWidget(self.sidebar_title)
        sidebar.addWidget(self.sidebar_coords)
        sidebar.addWidget(self.sidebar_sample)
        sidebar.addWidget(self.sidebar_box)
        sidebar.addWidget(self.sidebar_move)
        sidebar.addWidget(self.sidebar_mode)
        sidebar.addStretch(1)

        main = QHBoxLayout()
        main.addWidget(self.label, 1)
        main.addLayout(sidebar)

        root = QVBoxLayout()
        root.addLayout(main)
        root.addWidget(self.btn_toggle)
        root.addWidget(self.btn_mode)
        self.setLayout(root)

        self.worker = CameraWorker(camera_index=camera_index, fps=fps)
        self.worker.frameReady.connect(self.on_frame)
        self.worker.cameraError.connect(self.on_camera_error)
        self.worker.start()
        self._last_frame = None
        self._frame_count = 0
        self._last_boxes = []
        self.confidence = confidence
        self.mapper = None
        self.use_relative = True
        try:
            self.detector = YOLO("yolov8n.pt")
        except Exception as exc:
            print(exc)
            self.detector = None
            QMessageBox.warning(self, "Model Load Error", f"Could not load yolov8n-face.pt: {exc}")

    @pyqtSlot(object)
    def on_frame(self, frame_bgr):
        self._last_frame = frame_bgr
        self._frame_count += 1

        if self.detector is not None and self._frame_count % 5 == 0:
            results = self.detector(frame_bgr, conf = self.confidence, verbose=False)
            boxes = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for idx, b in enumerate(results[0].boxes.xyxy.cpu().tolist(), start=1):
                    x1, y1, x2, y2 = map(int, b)
                    boxes.append({"id": idx, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            self._last_boxes = boxes

        display_frame = frame_bgr.copy()
        for box in self._last_boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"ID {box['id']}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.label.width(), self.label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.label.setPixmap(pix)

    @pyqtSlot(int, int)
    def on_label_clicked(self, x, y):
        if self._last_frame is None:
            self.sidebar_coords.setText(f"x: {x}, y: {y}")
            self.sidebar_sample.setText("color: -")
            self.sidebar_box.setText("box: -")
            self.sidebar_move.setText("movement: -")
            return

        mapped = self._map_label_to_frame(x, y)
        if mapped is None:
            self.sidebar_coords.setText(f"x: {x}, y: {y} (outside frame)")
            self.sidebar_sample.setText("color: out of bounds")
            self.sidebar_title.setText("Clicked Pixel")
            self.sidebar_box.setText("box: -")
            self.sidebar_move.setText("movement: -")
            return

        fx, fy = mapped
        self.sidebar_coords.setText(f"x: {fx}, y: {fy}")

        h, w, _ = self._last_frame.shape
        if 0 <= fx < w and 0 <= fy < h:
            b, g, r = self._last_frame[fy, fx]
            self.sidebar_sample.setText(f"color: r={int(r)}, g={int(g)}, b={int(b)}")
        else:
            self.sidebar_sample.setText("color: out of bounds")

        hit = self._box_at_point(fx, fy)
        if hit is not None:
            self.sidebar_title.setText("Clicked Pixel (IN FACE)")
            self.sidebar_box.setText(
                f"box: id={hit['id']} [{hit['x1']},{hit['y1']}] - [{hit['x2']},{hit['y2']}]"
            )
            self._ensure_mapper()
            if self.mapper is not None:
                if self.use_relative:
                    pitch, yaw, roll = self.mapper.get_relative_movement(fx, fy, hit)
                else:
                    pitch, yaw, roll = self.mapper.get_absolute_movement(fx, fy, hit)
                self.sidebar_move.setText(
                    f"movement: pitch={pitch:.3f}, yaw={yaw:.3f}, roll={roll:.3f}"
                )
        else:
            self.sidebar_title.setText("Clicked Pixel")
            self.sidebar_box.setText("box: -")
            self.sidebar_move.setText("movement: -")

    def _map_label_to_frame(self, x, y):
        if self._last_frame is None:
            return None
        frame_h, frame_w, _ = self._last_frame.shape
        label_w = max(1, self.label.width())
        label_h = max(1, self.label.height())

        scale = min(label_w / frame_w, label_h / frame_h)
        disp_w = int(frame_w * scale)
        disp_h = int(frame_h * scale)
        off_x = (label_w - disp_w) / 2
        off_y = (label_h - disp_h) / 2

        if x < off_x or y < off_y or x >= off_x + disp_w or y >= off_y + disp_h:
            return None

        fx = int((x - off_x) / scale)
        fy = int((y - off_y) / scale)
        return fx, fy

    def _box_at_point(self, x, y):
        for box in self._last_boxes:
            if box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"]:
                return box
        return None

    def _ensure_mapper(self):
        if self.mapper is not None or self._last_frame is None:
            return
        h, w, _ = self._last_frame.shape
        self.mapper = Mapper(
            height=h,
            width=w,
            fov_x=self.DEFAULT_FOV_X,
            fov_y=self.DEFAULT_FOV_Y,
            cam_to_robot=self.DEFAULT_CAM_TO_ROBOT,
        )

    @pyqtSlot(str)
    def on_camera_error(self, msg: str):
        QMessageBox.critical(self, "Camera Error", msg)

    def toggle_stream(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.btn_toggle.setText("Start")
        else:
            self.worker = CameraWorker()  # restart with same params if needed
            self.worker.frameReady.connect(self.on_frame)
            self.worker.cameraError.connect(self.on_camera_error)
            self.worker.start()
            self.btn_toggle.setText("Stop")

    def toggle_mode(self):
        self.use_relative = not self.use_relative
        mode = "Relative" if self.use_relative else "Absolute"
        self.btn_mode.setText(f"Mode: {mode}")
        label = "get_relative_movement" if self.use_relative else "get_absolute_movement"
        self.sidebar_mode.setText(f"mode: {label}")

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
        event.accept()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Webcam viewer")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--fps", type=int, default=15, help="Target FPS")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    w = WebcamViewer(camera_index=args.camera, fps=args.fps)
    w.show()
    sys.exit(app.exec())
        
