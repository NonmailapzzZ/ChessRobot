# final2.py
# Patched UI: build UI first, then try to init main.init() safely (no crash if missing)

import sys
import math
import time
import traceback

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QGroupBox, QGridLayout, QTextEdit,
    QDoubleSpinBox, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QLocale
from PyQt6.QtGui import QImage, QPixmap

# use coordinate module for FK/IK
import coordinate

# import main motor driver (may or may not have init())
import main

# OpenCV + numpy (camera)
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except Exception:
    HAS_OPENCV = False
    import types
    np = types.SimpleNamespace(array=lambda *a, **k: None)

# --- SCARA minimal (θ1, θ2, d4) ---
JOINT_LIMITS = {
    'theta1': (-180.0, 180.0),
    'theta2': (-180.0, 180.0),
    'd4': (0.0, 200.0)
}
L1 = 200.0
L2 = 150.0

# ---------------- Camera thread ----------------
class CameraThread(QThread):
    frame_ready = pyqtSignal(object)   # emits numpy RGB array
    error_occurred = pyqtSignal(str)

    def __init__(self, device_index=0):
        super().__init__()
        self.device_index = int(device_index)
        self.camera = None
        self.running = False
        self.brightness = 0    # -100..100
        self.sharpness = 0.0   # 0..5
        self.fps_sleep_ms = 30

    def set_brightness(self, v):
        try:
            self.brightness = int(v)
        except:
            pass

    def set_sharpness(self, v):
        try:
            self.sharpness = float(v)
        except:
            pass

    def run(self):
        if not HAS_OPENCV:
            self.error_occurred.emit("OpenCV not installed")
            return
        try:
            idx = self.device_index
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(idx)
            self.camera = cap
            if not cap or not cap.isOpened():
                self.error_occurred.emit(f"Cannot open camera {idx}")
                return

            self.running = True
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.error_occurred.emit("Failed to read camera frame")
                    break

                # Convert BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Brightness (beta)
                beta = float(self.brightness)
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)

                # Sharpening (unsharp mask style)
                s = float(self.sharpness)
                if s > 0.0:
                    blurred = cv2.GaussianBlur(frame, (0,0), sigmaX=1.0)
                    sharpened = cv2.addWeighted(frame, 1.0 + s*0.2, blurred, -s*0.2, 0)
                    frame = np.clip(sharpened, 0, 255).astype(np.uint8)

                self.frame_ready.emit(frame)
                self.msleep(self.fps_sleep_ms)
        except Exception as e:
            tb = traceback.format_exc()
            self.error_occurred.emit(f"Camera error: {e}\n{tb}")
        finally:
            try:
                if self.camera and self.camera.isOpened():
                    self.camera.release()
            except:
                pass
            self.running = False

    def stop(self):
        self.running = False

# ---------------- Main combined UI ----------------
class ScaraCameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCARA Control (θ1, θ2, d4) + Camera")
        self.resize(1100, 700)

        # state: theta1, theta2, d4
        self.joints = {k: 0.0 for k in JOINT_LIMITS.keys()}
        self.camera_thread = None

        # Build UI first (so self.log exists) then try to init motor hardware safely
        self._build_ui()
        self._sync_widgets()

        # Try to call main.init() if present, but don't crash if missing
        try:
            init_func = getattr(main, "init", None)
            if callable(init_func):
                ok = main.init()
                if ok:
                    self._log("[SYSTEM] Motor driver initialized (hardware).")
                else:
                    self._log("[SYSTEM] Motor driver initialized (SIMULATION mode or hw not ready).")
            else:
                print("[final2] main.init() not found — running in SIMULATION mode.")
                try:
                    self._log("[SYSTEM] Motor driver initialized (SIMULATION mode: main.init missing).")
                except Exception:
                    pass
        except Exception as e:
            print(f"[final2] Motor init failed: {e}")
            try:
                self._log(f"[ERR] Motor init failed: {e}")
            except Exception:
                pass

        # initial scan for cameras
        self._scan_cameras(max_devices=8)

        # timer (FK updates)
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start()

    def _build_ui(self):
        root = QWidget()
        main_h = QHBoxLayout(root)
        main_h.setContentsMargins(12,12,12,12)
        main_h.setSpacing(6)   # reduced spacing to bring panels closer

        # ---------------- Left: Controls + Log (wider) ----------------
        left_v = QVBoxLayout()
        left_v.setSpacing(10)

        # Joint group (taller rows, Arabic numerals)
        joint_g = QGroupBox("Joint Jog")
        joint_layout = QGridLayout()
        joint_g.setLayout(joint_layout)
        joint_g.setStyleSheet(self._group_style())

        # columns: label small, slider big, spinbox small
        joint_layout.setColumnStretch(0, 1)
        joint_layout.setColumnStretch(1, 5)   # slightly narrower slider column
        joint_layout.setColumnStretch(2, 1)
        joint_layout.setVerticalSpacing(16)

        self.sliders = {}
        self.spinboxes = {}

        controls = [('theta1','θ1 (M1)'), ('theta2','θ2 (M2)'), ('d4','d4 (M4)')]
        row = 0
        for key, label in controls:
            minv, maxv = JOINT_LIMITS[key]

            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight:700; padding-left:6px; font-size:14px; color:#c8efe6;")  # changed color
            joint_layout.addWidget(lbl, row, 0)

            # slightly slimmer slider visual
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 1000)
            s.setFixedHeight(20)   # slightly smaller
            self.sliders[key] = s
            joint_layout.addWidget(s, row, 1)

            sp = QDoubleSpinBox()
            sp.setDecimals(2)
            sp.setRange(minv, maxv)
            sp.setSingleStep(0.5)
            sp.setLocale(QLocale(QLocale.Language.English))  # force Arabic numerals
            if key != 'd4':
                sp.setSuffix('°')
            else:
                sp.setSuffix(' mm')
            sp.setMinimumWidth(120)
            sp.setFixedHeight(32)
            sp.setStyleSheet("font-size:14px; padding:4px; color:#e6f2f1; background:#0f1b1f; border:1px solid #274344;")
            self.spinboxes[key] = sp
            joint_layout.addWidget(sp, row, 2)

            row += 1

        joint_g.setMinimumHeight(200)
        left_v.addWidget(joint_g)

        # action buttons
        btns = QHBoxLayout()
        self.btn_send = QPushButton("Send"); self.btn_send.clicked.connect(self._on_send)
        self.btn_home = QPushButton("Home"); self.btn_home.clicked.connect(self._on_home)
        self.btn_stop = QPushButton("STOP"); self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setStyleSheet("background:#d45b5b;color:white;font-weight:700; font-size:13px; padding:8px; border:2px solid #9e3a3a;")
        for b in (self.btn_send, self.btn_home):
            b.setStyleSheet("font-size:13px; padding:8px; border:2px solid rgba(43,109,255,0.12);")
        btns.addWidget(self.btn_send); btns.addWidget(self.btn_home); btns.addWidget(self.btn_stop)
        left_v.addLayout(btns)

        # Console / Log (taller)
        left_v.addWidget(QLabel("Console / Log"))
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(360)
        self.log.setStyleSheet("font-size:13px; color:#dff0ee; background:#071214; border:2px solid #213c3a;")
        left_v.addWidget(self.log)

        left_v.addStretch(1)
        main_h.addLayout(left_v, 2)   # left weight reduced to 2 (was 3) -> brings mid panel closer

        # ---------------- Middle: Cartesian + Camera ----------------
        mid_v = QVBoxLayout()
        mid_v.setSpacing(12)

        cart_g = QGroupBox("Cartesian Pose (FK)")
        cart_layout = QGridLayout()
        cart_g.setLayout(cart_layout)
        cart_g.setStyleSheet(self._group_style(bold=True))  # bolder border for cart box

        lbl_x = QLabel("X (mm)"); self.val_x = QLabel("0.00")
        lbl_y = QLabel("Y (mm)"); self.val_y = QLabel("0.00")
        lbl_z = QLabel("Z (mm)"); self.val_z = QLabel("0.00")
        lbl_yaw = QLabel("Yaw (°)"); self.val_yaw = QLabel("0.00")
        for w in (self.val_x,self.val_y,self.val_z,self.val_yaw):
            w.setStyleSheet("background:#e9f1f5;padding:8px;border-radius:6px;color:#092027; font-size:14px; border:2px solid #c6e2df;")
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # change label color to match theme (not black)
        lbl_x.setStyleSheet("color:#c8efe6; font-size:14px;")
        lbl_y.setStyleSheet("color:#c8efe6; font-size:14px;")
        lbl_z.setStyleSheet("color:#c8efe6; font-size:14px;")
        lbl_yaw.setStyleSheet("color:#c8efe6; font-size:14px;")

        cart_layout.addWidget(lbl_x,0,0); cart_layout.addWidget(self.val_x,0,1)
        cart_layout.addWidget(lbl_y,1,0); cart_layout.addWidget(self.val_y,1,1)
        cart_layout.addWidget(lbl_z,2,0); cart_layout.addWidget(self.val_z,2,1)
        cart_layout.addWidget(lbl_yaw,3,0); cart_layout.addWidget(self.val_yaw,3,1)
        mid_v.addWidget(cart_g)

        # Camera group
        cam_g = QGroupBox("Camera")
        cam_g.setStyleSheet(self._group_style())
        cam_layout = QVBoxLayout()
        cam_g.setLayout(cam_layout)

        # camera control row: combobox + scan + open/close
        cam_ctrl = QHBoxLayout()
        self.cam_device_combo = QComboBox()
        self.cam_device_combo.setFixedWidth(160)
        self.scan_cam_btn = QPushButton("Scan Cameras")
        self.scan_cam_btn.setFixedWidth(120)
        self.scan_cam_btn.clicked.connect(lambda: self._scan_cameras(max_devices=8))

        self.btn_cam_open = QPushButton("Open Camera")
        self.btn_cam_open.setFixedWidth(120)
        self.btn_cam_open.clicked.connect(self._on_cam_open)
        self.btn_cam_close = QPushButton("Close Camera")
        self.btn_cam_close.setFixedWidth(120)
        self.btn_cam_close.setEnabled(False)
        self.btn_cam_close.clicked.connect(self._on_cam_close)

        cam_ctrl.addWidget(self.cam_device_combo)
        cam_ctrl.addWidget(self.scan_cam_btn)
        cam_ctrl.addWidget(self.btn_cam_open)
        cam_ctrl.addWidget(self.btn_cam_close)
        cam_ctrl.addStretch()
        cam_layout.addLayout(cam_ctrl)

        # camera sliders
        cam_sl = QHBoxLayout()
        self.label_bright = QLabel("Brightness: 0"); self.label_bright.setFixedWidth(120); self.label_bright.setStyleSheet("font-size:13px; color:#c8efe6;")
        self.slider_bright = QSlider(Qt.Orientation.Horizontal); self.slider_bright.setRange(-100,100); self.slider_bright.setValue(0); self.slider_bright.setEnabled(False); self.slider_bright.setFixedWidth(200)
        self.label_sharp = QLabel("Sharpness: 0.0"); self.label_sharp.setFixedWidth(120); self.label_sharp.setStyleSheet("font-size:13px; color:#c8efe6;")
        self.slider_sharp = QSlider(Qt.Orientation.Horizontal); self.slider_sharp.setRange(0,50); self.slider_sharp.setValue(0); self.slider_sharp.setEnabled(False); self.slider_sharp.setFixedWidth(180)
        cam_sl.addWidget(self.label_bright); cam_sl.addWidget(self.slider_bright); cam_sl.addWidget(self.label_sharp); cam_sl.addWidget(self.slider_sharp)
        cam_layout.addLayout(cam_sl)

        # image display (reduced)
        self.image_label = QLabel("Camera Off")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(420, 260)
        self.image_label.setStyleSheet("background:#1f2933;border:2px solid #323b42;border-radius:6px;color:#cfe8e5; font-size:13px;")
        cam_layout.addWidget(self.image_label)

        mid_v.addWidget(cam_g)
        mid_v.addStretch(1)
        main_h.addLayout(mid_v, 1)   # middle weight

        self.setCentralWidget(root)
        self._apply_styles()

        # connect signals
        for k in self.sliders.keys():
            self.sliders[k].valueChanged.connect(self._make_slider_changed(k))
        for k in self.spinboxes.keys():
            self.spinboxes[k].valueChanged.connect(self._make_spin_changed(k))

        self.slider_bright.valueChanged.connect(self._on_brightness_changed)
        self.slider_sharp.valueChanged.connect(self._on_sharp_changed)

    def _group_style(self, bold=False):
        # make group borders more contrasted/sharp
        if bold:
            return ('QGroupBox { font-weight:700; '
                    'border:2px solid rgba(140,220,200,0.42); '
                    'border-radius:8px; margin-top:6px; padding:10px; font-size:14px; color:#c8efe6; }')
        return ('QGroupBox { font-weight:700; '
                'border:2px solid rgba(60,100,95,0.18); '
                'border-radius:8px; margin-top:6px; padding:10px; font-size:14px; color:#c8efe6; }')

    def _apply_styles(self):
        self.setStyleSheet(r"""
            QMainWindow { background: #17222a; color: #e6f2f1; font-family: Arial, Helvetica; font-size:14px; }
            QLabel { color: #e6f2f1; font-size:14px; }

            /* Buttons: clearer border and subtle highlight */
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #4b9df8, stop:1 #2b6fdc);
                color: white;
                padding:10px 12px;
                border-radius:6px;
                font-size:14px;
                border:2px solid rgba(45,105,210,0.22);
                font-weight:600;
            }
            QPushButton:hover { background: #3a86f0; }
            QPushButton:disabled { background:#2b3b3b; color:#7b9b98; border:1px solid #213232; }

            /* Slightly slimmer slider groove + handle (keeps contrast) */
            QSlider::groove:horizontal {
                height:14px;
                background:#27363b;
                border-radius:8px;
                margin: 2px 0;
                border:1px solid rgba(40,80,76,0.25);
            }
            QSlider::sub-page:horizontal { background: #4b9df8; border-radius:8px; }
            QSlider::add-page:horizontal { background: #213239; border-radius:8px; }
            QSlider::handle:horizontal {
                background: #f1f7f7;
                border: 2px solid #cbdfe3;
                width:16px;
                height:16px;
                margin: -3px 0;
                border-radius:8px;
            }

            /* crisp input widgets */
            QDoubleSpinBox {
                background:#0f1b1f;
                color:#e6f2f1;
                border:2px solid #274344;
                padding:6px;
                min-width:88px;
                font-size:14px;
                border-radius:6px;
            }
            QComboBox {
                background:#ffffff; color:#111; border:2px solid #274344; padding:4px; border-radius:6px;
            }
            QTextEdit {
                background:#071214;
                color:#dff0ee;
                border:2px solid #213c3a;
                font-size:13px;
                border-radius:6px;
            }

            /* Group box title color and position */
            QGroupBox::title {
                subcontrol-origin: margin;
                left:8px;
                padding: 0 4px;
                color: #9feadf;    /* changed: not black */
                font-weight:800;
            }
        """)

    # ---------------- widget sync ----------------
    def _sync_widgets(self):
        for k in self.sliders.keys():
            minv, maxv = JOINT_LIMITS[k]
            val = self.joints[k]
            pos = int((val - minv) / (maxv - minv) * 1000)
            self.sliders[k].blockSignals(True); self.sliders[k].setValue(pos); self.sliders[k].blockSignals(False)
            self.spinboxes[k].blockSignals(True); self.spinboxes[k].setValue(val); self.spinboxes[k].blockSignals(False)
        self._update_fk_display()

    def _make_slider_changed(self, key):
        def on_change(pos):
            minv, maxv = JOINT_LIMITS[key]
            value = minv + (pos / 1000.0) * (maxv - minv)
            self.spinboxes[key].blockSignals(True)
            self.spinboxes[key].setValue(round(value,2))
            self.spinboxes[key].blockSignals(False)
            self.joints[key] = float(value)
            self._update_fk_display()
        return on_change

    def _make_spin_changed(self, key):
        def on_change(val):
            minv, maxv = JOINT_LIMITS[key]
            if val < minv: val = minv
            if val > maxv: val = maxv
            pos = int((val - minv) / (maxv - minv) * 1000)
            self.sliders[key].blockSignals(True)
            self.sliders[key].setValue(pos)
            self.sliders[key].blockSignals(False)
            self.joints[key] = float(val)
            self._update_fk_display()
        return on_change

    def _update_fk_display(self):
        x,y,z,yaw = coordinate.fk_scara(self.joints['theta1'], self.joints['theta2'], self.joints['d4'])
        self.val_x.setText(f"{x:.2f}"); self.val_y.setText(f"{y:.2f}")
        self.val_z.setText(f"{z:.2f}"); self.val_yaw.setText(f"{yaw:.2f}")

    # ---------------- actions ----------------
    def _on_send(self):
        # build command string for log (existing format)
        cmd = f"CMD,TH1,{self.joints['theta1']:.2f},TH2,{self.joints['theta2']:.2f},D4,{self.joints['d4']:.2f}"
        self._log(f"[SEND] {cmd}")
        # --- HERE: call main.set_joints to actually drive hardware ---
        try:
            setj = getattr(main, "set_joints", None)
            if callable(setj):
                main.set_joints(self.joints['theta1'], self.joints['theta2'], self.joints['d4'])
                self._log("[SEND] dispatched to motor driver (main.set_joints).")
            else:
                print("[final2] main.set_joints() not found; running in SIMULATION mode.")
                self._log("[WARN] main.set_joints() not found; SIMULATION only.")
        except Exception as e:
            self._log(f"[ERR] sending to motor driver failed: {e}")

    def _on_home(self):
        for k in self.joints.keys():
            self.joints[k] = 0.0
            self.spinboxes[k].blockSignals(True); self.spinboxes[k].setValue(0.0); self.spinboxes[k].blockSignals(False)
            minv,maxv = JOINT_LIMITS[k]; pos = int((0.0 - minv) / (maxv - minv) * 1000)
            self.sliders[k].blockSignals(True); self.sliders[k].setValue(pos); self.sliders[k].blockSignals(False)
        self._log("[ACTION] Home (all zeros)")
        self._update_fk_display()
        # optionally also home hardware
        try:
            setj = getattr(main, "set_joints", None)
            if callable(setj):
                main.set_joints(0.0, 0.0, 0.0)
                self._log("[ACTION] Home sent to motor driver.")
            else:
                self._log("[WARN] main.set_joints() not found; Home only simulated.")
        except Exception as e:
            self._log(f"[ERR] home send failed: {e}")

    def _on_stop(self):
        self._log("[ACTION] EMERGENCY STOP (simulated)")
        # optional: immediately stop motors (if supported)
        try:
            setj = getattr(main, "set_joints", None)
            if callable(setj):
                main.set_joints(0.0, 0.0, 0.0)
                self._log("[ACTION] Stop sent to motor driver.")
            else:
                self._log("[WARN] main.set_joints() not found; Stop only simulated.")
        except Exception as e:
            self._log(f"[ERR] stop failed: {e}")

    def _log(self, text):
        ts = time.strftime("%H:%M:%S")
        try:
            self.log.append(f"{ts}  {text}")
        except Exception:
            # fallback to printing if log widget not ready
            print(f"{ts} {text}")

    # ---------------- camera: scan / open / close ----------------
    def _scan_cameras(self, max_devices=8):
        self.cam_device_combo.clear()
        found = []
        if not HAS_OPENCV:
            self.cam_device_combo.addItem("OpenCV not installed", -1)
            return
        for i in range(max_devices):
            cap = None
            try:
                if sys.platform.startswith("win"):
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(i)
                if not cap:
                    continue
                ok, _ = cap.read()
                cap.release()
                if ok:
                    found.append(i)
            except Exception:
                try:
                    if cap and cap.isOpened():
                        cap.release()
                except:
                    pass
        if not found:
            self.cam_device_combo.addItem("No camera", -1)
        else:
            for idx in found:
                self.cam_device_combo.addItem(f"Camera {idx}", idx)
            self.cam_device_combo.setCurrentIndex(0)
        self._log(f"[CAM] Scan finished: found {len(found)} device(s)")

    def _on_cam_open(self):
        if not HAS_OPENCV:
            QMessageBox.warning(self, "Camera", "OpenCV not installed. Install opencv-python to use camera.")
            return
        idx = 0
        if self.cam_device_combo.count() == 0:
            idx = 0
        else:
            data = self.cam_device_combo.currentData()
            if data is None:
                txt = self.cam_device_combo.currentText()
                try:
                    idx = int(''.join(ch for ch in txt if ch.isdigit()))
                except:
                    idx = 0
            else:
                idx = int(data)
                if idx < 0:
                    QMessageBox.warning(self, "Camera", "No valid camera selected.")
                    return

        if self.camera_thread and self.camera_thread.isRunning():
            return
        self.camera_thread = CameraThread(device_index=idx)
        self.camera_thread.frame_ready.connect(self._on_camera_frame)
        self.camera_thread.error_occurred.connect(self._on_camera_error)
        self.camera_thread.set_brightness(self.slider_bright.value())
        self.camera_thread.set_sharpness(self.slider_sharp.value() / 10.0)
        self.camera_thread.start()
        self.btn_cam_open.setEnabled(False); self.btn_cam_close.setEnabled(True)
        self.slider_bright.setEnabled(True); self.slider_sharp.setEnabled(True)
        self._log(f"[CAM] Opened device {idx}")

    def _on_cam_close(self):
        if self.camera_thread:
            try:
                self.camera_thread.stop()
                self.camera_thread.wait(timeout=1000)
            except Exception:
                pass
            self.camera_thread = None
        self.image_label.setText("Camera Off")
        self.btn_cam_open.setEnabled(True); self.btn_cam_close.setEnabled(False)
        self.slider_bright.setEnabled(False); self.slider_sharp.setEnabled(False)
        self._log("[CAM] Closed")

    def _on_brightness_changed(self, v):
        self.label_bright.setText(f"Brightness: {v}")
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.set_brightness(v)

    def _on_sharp_changed(self, v):
        s = v / 10.0
        self.label_sharp.setText(f"Sharpness: {s:.1f}")
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.set_sharpness(s)

    def _on_camera_frame(self, frame):
        try:
            h,w,ch = frame.shape
            bytes_per_line = ch * w
            qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            scaled = pix.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(scaled)
        except Exception as e:
            self._log(f"[CAM ERR] {e}")

    def _on_camera_error(self, msg):
        QMessageBox.critical(self, "Camera error", msg)
        self._on_cam_close()

    # ---------------- periodic ----------------
    def _on_tick(self):
        # nothing else needed; FK updates happen on widget changes
        pass

    def closeEvent(self, event):
        try:
            if self.camera_thread:
                self.camera_thread.stop()
                self.camera_thread.wait(timeout=1000)
        except Exception:
            pass
        # cleanup motor driver
        try:
            cleanup_func = getattr(main, "cleanup", None)
            if callable(cleanup_func):
                main.cleanup()
            else:
                print("[final2] main.cleanup() not found; skipping.")
        except Exception:
            pass
        event.accept()


# ---------------- run ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ScaraCameraApp()
    win.show()
    sys.exit(app.exec())
