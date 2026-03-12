# filename: webcam_qt_thread.py

## threaded
import sys
import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QHBoxLayout,
    QLineEdit,
)
from ultralytics import YOLO
from mapping import Mapper
from furhat_control import FurhatController


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
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("Furhat IP (e.g., 192.168.1.50)")
        self.btn_connect = QPushButton("Connect Furhat")
        self.btn_connect.clicked.connect(self.connect_furhat)
        self.lbl_furhat_status = QLabel("Furhat: disconnected")

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
        self.sidebar_furhat = QLabel("furhat: disconnected")
        self.sidebar_furhat.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sidebar_furhat_move = QLabel("furhat move: -")
        self.sidebar_furhat_move.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sidebar_furhat_pose = QLabel("furhat pose: -")
        self.sidebar_furhat_pose.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.sidebar_furhat_can_move = QLabel("furhat can move: -")
        self.sidebar_furhat_can_move.setAlignment(Qt.AlignmentFlag.AlignLeft)

        sidebar = QVBoxLayout()
        sidebar.addWidget(self.sidebar_title)
        sidebar.addWidget(self.sidebar_coords)
        sidebar.addWidget(self.sidebar_sample)
        sidebar.addWidget(self.sidebar_box)
        sidebar.addWidget(self.sidebar_move)
        sidebar.addWidget(self.sidebar_furhat)
        sidebar.addWidget(self.sidebar_furhat_move)
        sidebar.addWidget(self.sidebar_furhat_pose)
        sidebar.addWidget(self.sidebar_furhat_can_move)
        sidebar.addStretch(1)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setMinimumWidth(320)

        main = QHBoxLayout()
        main.addWidget(self.label, 1)
        main.addWidget(sidebar_widget)

        root = QVBoxLayout()
        root.addLayout(main)
        root.addWidget(self.btn_toggle)
        root.addWidget(self.ip_input)
        root.addWidget(self.btn_connect)
        root.addWidget(self.lbl_furhat_status)
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
        self.furhat: FurhatController | None = None
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
        self._update_furhat_can_move()

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
                #pitch, yaw, roll = self.mapper.get_relative_movement(fx, fy, hit)
                pitch, yaw, roll = self.mapper.get_absolute_movement(fx, fy, hit)
                self.sidebar_move.setText(
                    f"movement: pitch={pitch:.3f}, yaw={yaw:.3f}, roll={roll:.3f}"
                )
                if self.furhat is not None:
                    try:
                        fut = self.furhat.submit(
                            #self.furhat.move_head_relative(yaw=yaw, pitch=pitch, roll=roll)
                            self.furhat.move_head_absolute(yaw=yaw, pitch=pitch, roll=roll)
                        )
                        fut.result()
                        self.sidebar_furhat_move.setText("furhat move: ok")
                        pose = self.furhat.get_head_pose()
                        if pose is None:
                            self.sidebar_furhat_pose.setText("furhat pose: -")
                        else:
                            self.sidebar_furhat_pose.setText(
                                f"furhat pose: pitch={pose.pitch:.3f}, yaw={pose.yaw:.3f}, roll={pose.roll:.3f}"
                            )
                    except Exception as exc:
                        self.sidebar_furhat_move.setText(f"furhat move: failed ({exc})")
                        self.sidebar_furhat_pose.setText("furhat pose: -")
                        self._update_furhat_can_move()
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

    def connect_furhat(self):
        ip = self.ip_input.text().strip()
        if not ip:
            QMessageBox.warning(self, "Missing IP", "Enter the Furhat IP address.")
            return
        try:
            if self.furhat is not None:
                fut = self.furhat.submit(self.furhat.disconnect())
                fut.result()
            self.furhat = FurhatController(ip)
            fut = self.furhat.submit(self.furhat.connect())
            fut.result()
            self.lbl_furhat_status.setText(f"Furhat: connected to {ip}")
            self.sidebar_furhat.setText(f"furhat: connected to {ip}")
            self._update_furhat_can_move()
        except Exception as exc:
            self.furhat = None
            self.lbl_furhat_status.setText("Furhat: disconnected")
            self.sidebar_furhat.setText("furhat: disconnected")
            self.sidebar_furhat_can_move.setText("furhat can move: -")
            QMessageBox.critical(self, "Furhat Connection Error", str(exc))

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
        if self.furhat is not None:
            try:
                fut = self.furhat.submit(self.furhat.disconnect())
                fut.result()
            except Exception:
                pass
        event.accept()

    def _update_furhat_can_move(self):
        if self.furhat is None:
            self.sidebar_furhat_can_move.setText("furhat can move: -")
            return
        if self.furhat.can_move_now():
            self.sidebar_furhat_can_move.setText("furhat can move: yes")
        else:
            remaining = self.furhat.time_until_move()
            self.sidebar_furhat_can_move.setText(
                f"furhat can move: no ({remaining:.2f}s)"
            )


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
        
    ## default furhat for HAILabRouter4: 192.168.0.199
