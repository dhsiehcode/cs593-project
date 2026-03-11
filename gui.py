

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


class CameraWorker(QThread):
    frameReady = pyqtSignal(object)   # emits numpy array (BGR)
    cameraError = pyqtSignal(str)

    def __init__(self, camera_index: int = 0, fps: int = 30):
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
    def __init__(self, camera_index: int = 0, fps: int = 30):
        super().__init__()
        self.setWindowTitle("Webcam Stream - QThread")
        self.label = ClickableLabel("Starting camera...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(640, 480)
        self.label.clicked.connect(self.on_label_clicked)
        self.btn_toggle = QPushButton("Stop")
        self.btn_toggle.clicked.connect(self.toggle_stream)

        self.sidebar_title = QLabel("Clicked Pixel")
        self.sidebar_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sidebar_coords = QLabel("x: -, y: -")
        self.sidebar_coords.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sidebar_sample = QLabel("color: -")
        self.sidebar_sample.setAlignment(Qt.AlignmentFlag.AlignLeft)

        sidebar = QVBoxLayout()
        sidebar.addWidget(self.sidebar_title)
        sidebar.addWidget(self.sidebar_coords)
        sidebar.addWidget(self.sidebar_sample)
        sidebar.addStretch(1)

        main = QHBoxLayout()
        main.addWidget(self.label, 1)
        main.addLayout(sidebar)

        root = QVBoxLayout()
        root.addLayout(main)
        root.addWidget(self.btn_toggle)
        self.setLayout(root)

        self.worker = CameraWorker(camera_index=camera_index, fps=fps)
        self.worker.frameReady.connect(self.on_frame)
        self.worker.cameraError.connect(self.on_camera_error)
        self.worker.start()
        self._last_frame = None

    @pyqtSlot(object)
    def on_frame(self, frame_bgr):
        self._last_frame = frame_bgr
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
        self.sidebar_coords.setText(f"x: {x}, y: {y}")
        if self._last_frame is None:
            self.sidebar_sample.setText("color: -")
            return

        h, w, _ = self._last_frame.shape
        if 0 <= x < w and 0 <= y < h:
            b, g, r = self._last_frame[y, x]
            self.sidebar_sample.setText(f"color: r={int(r)}, g={int(g)}, b={int(b)}")
        else:
            self.sidebar_sample.setText("color: out of bounds")

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

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = WebcamViewer(camera_index=0, fps=30)
    w.show()
    sys.exit(app.exec())
        
