# filename: webcam_qt_thread.py

## threaded
import sys
import time
import cv2
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal, pyqtSlot
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
from names import Face, KnownFaces
from gesture_detector import GestureDetector

_GESTURE_HYSTERESIS_S = 2.0


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
    DEFAULT_CAM_TO_ROBOT = 0.3

    def __init__(self, camera_index: int = 0, fps: int = 15, confidence: float = 0.7):
        super().__init__()
        self.setWindowTitle("Webcam Stream - QThread")
        self.label = ClickableLabel("Starting camera...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(640, 480)
        self.label.clicked.connect(self.on_label_clicked)
        self.btn_toggle = QPushButton("Stop")
        self.btn_toggle.clicked.connect(self.toggle_stream)
        self.btn_tracking = QPushButton("Enable Tracking")
        self.btn_tracking.setCheckable(True)
        self.btn_tracking.clicked.connect(self.toggle_tracking)
        self.btn_greeting = QPushButton("Enable Greeting")
        self.btn_greeting.setCheckable(True)
        self.btn_greeting.clicked.connect(self.toggle_greeting)
        self.btn_gesture = QPushButton("Enable Gesture Attention")
        self.btn_gesture.setCheckable(True)
        self.btn_gesture.clicked.connect(self.toggle_gesture)
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
        #sidebar.addWidget(self.sidebar_furhat_move)
        #sidebar.addWidget(self.sidebar_furhat_pose)
        sidebar.addWidget(self.sidebar_furhat_can_move)
        sidebar.addStretch(1)

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar)
        sidebar_widget.setMinimumWidth(320)


        self.left_sidebar_title = QLabel("Controls")
        self.left_sidebar_title.setAlignment(Qt.AlignmentFlag.AlignLeft)

        left_sidebar = QVBoxLayout()
        left_sidebar.addWidget(self.left_sidebar_title)
        left_sidebar.addWidget(self.btn_tracking)
        left_sidebar.addWidget(self.btn_greeting)
        left_sidebar.addWidget(self.btn_gesture)
        left_sidebar.addStretch(1)

        left_sidebar_widget = QWidget()
        left_sidebar_widget.setLayout(left_sidebar)
        left_sidebar_widget.setMinimumWidth(160)

        main = QHBoxLayout()
        main.addWidget(left_sidebar_widget)
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
        self._tracked_face_center: tuple[int, int] | None = None
        self._known_faces: KnownFaces = KnownFaces()
        self._greeting_enabled = False
        self._gesture_mode = False
        self._gesture_last_id: int | None = None
        self._gesture_last_switch: float = 0.0
        self._gesture_detector = GestureDetector()
        self._tracking_timer = QTimer(self)
        self._tracking_timer.setInterval(500)
        self._tracking_timer.timeout.connect(self._track_face)
        try:
            self.detector = YOLO("yolov8n-pose.pt")
        except Exception as exc:
            print(exc)
            self.detector = None
            QMessageBox.warning(self, "Model Load Error", f"Could not load yolov8n-pose.pt: {exc}")

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
                    if self._greeting_enabled and self._known_faces.get_by_id(idx) is None:
                        face = Face(
                            id=idx,
                            bbox=[x1, y1, x2, y2],
                            face_center=((x1 + x2) // 2, (y1 + y2) // 2),
                        )
                        self._known_faces.add(face)
                        if self.furhat is not None:
                            self.furhat.submit(self._greet_face(face))
            self._last_boxes = boxes

            if self._gesture_mode:
                self._gesture_detector.update_scores(results, self._known_faces, frame_bgr.shape[0])
                salient: Face | None = self._gesture_detector.most_salient(self._known_faces)

                if salient is not None:
                    now = time.monotonic()
                    if (salient.id != self._gesture_last_id and
                            now - self._gesture_last_switch < _GESTURE_HYSTERESIS_S):
                        salient = None
                    else:
                        self._gesture_last_id = salient.id
                        self._gesture_last_switch = now

                if salient is not None:
                    cx, cy = salient.face_center
                    self._tracked_face_center = (cx, cy)
                    self.sidebar_title.setText(
                        f"Gesture → {salient.name} ID {salient.id} "
                        f"(score {salient.gesture_score:.2f})"
                    )
                    self._ensure_mapper()
                    if self.furhat is not None and self.mapper is not None and self.furhat.can_move_now():
                        bbox_dict = {"x1": salient.bbox[0], "y1": salient.bbox[1],
                                     "x2": salient.bbox[2], "y2": salient.bbox[3]}
                        pitch, yaw, roll = self.mapper.get_absolute_movement(cx, cy, bbox_dict)
                        fut = self.furhat.submit(
                            self.furhat.move_head_relative(yaw=yaw, pitch=pitch, roll=roll)
                        )
                        fut.result()
                else:
                    self.sidebar_title.setText("Gesture: no salient target")

        display_frame = frame_bgr.copy()
        tracked_id = None
        if self._tracked_face_center is not None and self._last_boxes:
            tx, ty = self._tracked_face_center
            tracked_id = min(
                self._last_boxes,
                key=lambda b: (((b["x1"] + b["x2"]) // 2 - tx) ** 2 +
                               ((b["y1"] + b["y2"]) // 2 - ty) ** 2),
            )["id"]
        for box in self._last_boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            color = (0, 165, 255) if box["id"] == tracked_id else (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            known = self._known_faces.get_by_id(box["id"])
            if known is None or known.name == "unknown":
                box_label = f"ID {box['id']}"
            elif known.name in ("-", "greeting"):
                box_label = "..."
            else:
                box_label = known.name
            cv2.putText(
                display_frame,
                box_label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            if self._gesture_mode:
                face = self._known_faces.get_by_id(box["id"])
                if face is not None and face.gesture_score >= GestureDetector.RAISED_HAND_THRESHOLD:
                    cv2.putText(
                        display_frame,
                        f"HAND ({face.gesture_score:.2f})",
                        (x1, max(0, y1 - 22)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
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
            cx = (hit["x1"] + hit["x2"]) // 2
            cy = (hit["y1"] + hit["y2"]) // 2
            self._tracked_face_center = (cx, cy)
            if self.btn_tracking.isChecked() and not self._tracking_timer.isActive():
                self._tracking_timer.start()
            self.sidebar_title.setText("Clicked Pixel (IN FACE, tracking)")
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
                            self.furhat.move_head_relative(yaw=yaw, pitch=pitch, roll=roll)
                            #self.furhat.move_head_absolute(yaw=yaw, pitch=pitch, roll=roll)
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
            self._tracked_face_center = None
            self._tracking_timer.stop()
            self.sidebar_title.setText("Clicked Pixel")
            self.sidebar_box.setText("box: -")
            self.sidebar_move.setText("movement: -")

    def toggle_tracking(self, checked: bool):
        if checked:
            self.btn_tracking.setText("Disable Tracking")
            if self._tracked_face_center is not None and not self._tracking_timer.isActive():
                self._tracking_timer.start()
        else:
            self.btn_tracking.setText("Enable Tracking")
            self._tracking_timer.stop()
            self._tracked_face_center = None
            self.sidebar_title.setText("Clicked Pixel")

    def toggle_greeting(self, checked: bool):
        self._greeting_enabled = checked
        if checked:
            self.btn_greeting.setText("Disable Greeting")
        else:
            self.btn_greeting.setText("Enable Greeting")

    def toggle_gesture(self, checked: bool):
        self._gesture_mode = checked
        self.btn_gesture.setText(
            "Disable Gesture Attention" if checked else "Enable Gesture Attention"
        )
        if not checked:
            for face in self._known_faces.faces:
                face.gesture_score = 0.0
            self.sidebar_title.setText("Clicked Pixel")

    def _track_face(self):
        if self._tracked_face_center is None or self.furhat is None or self.mapper is None:
            return
        if not self._last_boxes:
            return
        tx, ty = self._tracked_face_center
        best = min(
            self._last_boxes,
            key=lambda b: (((b["x1"] + b["x2"]) // 2 - tx) ** 2 +
                           ((b["y1"] + b["y2"]) // 2 - ty) ** 2),
        )
        cx = (best["x1"] + best["x2"]) // 2
        cy = (best["y1"] + best["y2"]) // 2
        self._tracked_face_center = (cx, cy)
        self._ensure_mapper()
        pitch, yaw, roll = self.mapper.get_absolute_movement(cx, cy, best)
        self.sidebar_move.setText(f"movement: pitch={pitch:.3f}, yaw={yaw:.3f}, roll={roll:.3f}")
        try:
            fut = self.furhat.submit(
                self.furhat.move_head_relative(yaw=yaw, pitch=pitch, roll=roll)
            )
            fut.result()
            self.sidebar_furhat_move.setText("furhat move: ok (tracking)")
            pose = self.furhat.get_head_pose()
            if pose is not None:
                self.sidebar_furhat_pose.setText(
                    f"furhat pose: pitch={pose.pitch:.3f}, yaw={pose.yaw:.3f}, roll={pose.roll:.3f}"
                )
        except Exception as exc:
            self.sidebar_furhat_move.setText(f"furhat move: failed ({exc})")

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

    async def _greet_face(self, face: Face) -> None:
        face.name = "greeting"
        face.name = await self.furhat.greet_and_name()

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
        self._tracking_timer.stop()
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
