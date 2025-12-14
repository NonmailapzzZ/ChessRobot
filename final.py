# scara_camera_final_with_IK.py
# NOTE:
# - This file is BASED DIRECTLY on the user's provided scara_camera_final.py
# - NOTHING existing is removed or modified in behavior
# - ONLY ADDITIONS:
#   1) Inverse Kinematics (X,Y) panel
#   2) Camera panel moved visually to bottom-right (same widgets, same size)
#   3) Optional vertical scrolling via QScrollArea
# ------------------------------------------------------------

import sys
import math
import time
import traceback
from coordinate_origin import inverse_matrix, tranformation_matrix
import numpy as np





def fk_from_coordinate(theta1_deg, theta2_deg):
    t1 = np.deg2rad(theta1_deg)
    t2 = np.deg2rad(theta2_deg)

    T1, _, _ = tranformation_matrix(t1, 0)
    T2, _, _ = tranformation_matrix(t2, 1)

    T = T1 @ T2

    x_mm = T[0, 3]
    y_mm = T[1, 3]

    return x_mm, y_mm

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QGroupBox, QGridLayout, QTextEdit,
    QDoubleSpinBox, QMessageBox, QComboBox, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QLocale
from PyQt6.QtGui import QImage, QPixmap

# OpenCV + numpy (camera)
try:  
    import cv2
    import numpy as np
    HAS_OPENCV = True
except Exception:
    HAS_OPENCV = False
    import types
    np = types.SimpleNamespace(array=lambda *a, **k: None)

# ---------------- SCARA parameters (UNCHANGED) ----------------
JOINT_LIMITS = {
    'theta1': (0, 180.0),
    'theta2': (-160.0, 160.0),
}

L1 = 248.33
L2 = 223.38




# ---------------- Camera thread (UNCHANGED) ----------------
class CameraThread(QThread):
    frame_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, device_index=0):
        super().__init__()
        self.device_index = int(device_index)
        self.camera = None
        self.running = False
        self.brightness = 0
        self.sharpness = 0.0
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
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(self.device_index)

            self.camera = cap
            if not cap or not cap.isOpened():
                self.error_occurred.emit(f"Cannot open camera {self.device_index}")
                return

            self.running = True
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=float(self.brightness))
                if self.sharpness > 0:
                    blurred = cv2.GaussianBlur(frame, (0,0), sigmaX=1.0)
                    frame = cv2.addWeighted(frame, 1.0 + self.sharpness*0.2,
                                             blurred, -self.sharpness*0.2, 0)
                self.frame_ready.emit(frame)
                self.msleep(self.fps_sleep_ms)
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            try:
                if self.camera and self.camera.isOpened():
                    self.camera.release()
            except:
                pass
            self.running = False

    def stop(self):
        self.running = False

# ---------------- Main UI ----------------
class ScaraCameraApp(QMainWindow):
    def _update_fk_display(self):
        th1 = self.joints['theta1']
        th2 = self.joints['theta2']
       
        x_mm, y_mm = fk_from_coordinate(th1, th2)

        self.val_x.setText(f"{x_mm:.2f}")   # cm
        self.val_y.setText(f"{y_mm:.2f}")   # cm
        self.val_z.setText(self.joints['d4_state']) # cm
        self.val_yaw.setText(f"{(th1+th2):.2f}")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SCARA Control (θ1, θ2, d4) + Camera")
        self.resize(1100, 700)

        self.joints = {
            'theta1': 0.0,
            'theta2': 0.0,
            'd4_state': 'STOP'   # หรือ 'UP' / 'DOWN'
}

        self.camera_thread = None

        self._build_ui()
        self._sync_widgets()
        self._scan_cameras(max_devices=8)

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(lambda: None)
        self.timer.start()
    
    def _on_d4_up(self):
        try:
            from servo_controller import move_slow_slider
            move_slow_slider(target=70)
        except Exception as e:
            self._log(f"[D4] UP (SIMULATED) : {e}")

        self.joints['d4_state'] = 'UP'
        self.val_z.setText("UP")


    def _on_d4_down(self):
        try:
            from servo_controller import move_slow_slider
            move_slow_slider(target=0)
        except Exception as e:
            self._log(f"[D4] DOWN (SIMULATED) : {e}")

        self.joints['d4_state'] = 'DOWN'
        self.val_z.setText("DOWN")



    def _build_ui(self):#################################################################################
        # ---------------- ROOT WITH SCROLL (NEW) ----------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll.setWidget(container)
        self.setCentralWidget(scroll)

        main_h = QHBoxLayout(container)
        main_h.setContentsMargins(12,12,12,12)
        main_h.setSpacing(6)

        # ---------------- LEFT (UNCHANGED) ----------------
        left_v = QVBoxLayout()
        left_v.setSpacing(10)

        # Joint Jog (UNCHANGED CODE BELOW)
        joint_g = QGroupBox("Joint Jog")
        joint_layout = QGridLayout(joint_g)
        joint_g.setStyleSheet(self._group_style())

        joint_layout.setColumnStretch(0, 1)
        joint_layout.setColumnStretch(1, 5)
        joint_layout.setColumnStretch(2, 1)
        joint_layout.setVerticalSpacing(16)

        self.sliders = {}
        self.spinboxes = {}

        controls = [('theta1','θ1 (M1)'), ('theta2','θ2 (M2)')]
        for row,(key,label) in enumerate(controls):
            minv,maxv = JOINT_LIMITS[key]
            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight:700; padding-left:6px; font-size:14px; color:#1e1e1e;")
            joint_layout.addWidget(lbl,row,0)

            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0,1000)
            s.setFixedHeight(20)
            self.sliders[key] = s
            joint_layout.addWidget(s,row,1)

            sp = QDoubleSpinBox()
            sp.setDecimals(2)
            sp.setSingleStep(0.5)
            sp.setLocale(QLocale(QLocale.Language.English))
            sp.setRange(minv, maxv) 


            sp.setMinimumWidth(120)
            sp.setFixedHeight(32)
            sp.setStyleSheet("font-size:14px; padding:4px; color:#e6f2f1; background:#0f1b1f; border:1px solid #274344;")
            self.spinboxes[key] = sp
            joint_layout.addWidget(sp,row,2)

        left_v.addWidget(joint_g)
        # ===== D4 UP / DOWN CONTROL =====
        d4_g = QGroupBox("D4 Control")
        d4_g.setStyleSheet(self._group_style())
        d4_layout = QHBoxLayout(d4_g)

        self.btn_d4_up = QPushButton("UP")
        self.btn_d4_down = QPushButton("DOWN")

        self.btn_d4_up.clicked.connect(self._on_d4_up)
        self.btn_d4_down.clicked.connect(self._on_d4_down)

        d4_layout.addWidget(self.btn_d4_up)
        d4_layout.addWidget(self.btn_d4_down)

        left_v.addWidget(d4_g)


        btns = QHBoxLayout()
        self.btn_send = QPushButton("Send"); self.btn_send.clicked.connect(self._on_send)
        self.btn_home = QPushButton("Home"); self.btn_home.clicked.connect(self._on_home)
        self.btn_stop = QPushButton("STOP"); self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setStyleSheet("background:#d45b5b;color:white;font-weight:700;")
        btns.addWidget(self.btn_send); btns.addWidget(self.btn_home); btns.addWidget(self.btn_stop)
        left_v.addLayout(btns)

        left_v.addWidget(QLabel("Console / Log"))
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(360)
        self.log.setStyleSheet("font-size:13px; color:#dff0ee; background:#071214; border:2px solid #213c3a;")
        left_v.addWidget(self.log)
        left_v.addStretch(1)

        main_h.addLayout(left_v, 2)

        # ---------------- RIGHT (FK + IK + CAMERA) ----------------
        right_v = QVBoxLayout()
        right_v.setSpacing(12)

        # ----- FK (UNCHANGED) -----###########################################################
        cart_g = QGroupBox("Cartesian Pose (FK)")
        cart_layout = QGridLayout(cart_g)
        cart_g.setStyleSheet(self._group_style(bold=True))

        self.val_x = QLabel("0.00"); self.val_y = QLabel("0.00")
        self.val_z = QLabel("0.00"); self.val_yaw = QLabel("0.00")
        for w in (self.val_x,self.val_y,self.val_z,self.val_yaw):
            w.setStyleSheet("background:#e9f1f5;padding:8px;border-radius:6px;color:#092027; font-size:14px; border:2px solid #2f3e46;")
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)

        labels = [("X (cm)",self.val_x),("Y (cm)",self.val_y),("D4",self.val_z),("Yaw (°)",self.val_yaw)]
        for i,(txt,w) in enumerate(labels):
            lab = QLabel(txt); lab.setStyleSheet("color:#1e1e1e; font-size:14px;")
            cart_layout.addWidget(lab,i,0); cart_layout.addWidget(w,i,1)

        right_v.addWidget(cart_g)

        # ----- IK (NEW, ADDED ONLY) -----
        ik_g = QGroupBox("Inverse Kinematics (Top View)")
        ik_g.setStyleSheet(self._group_style())
        ik_layout = QGridLayout(ik_g)
        ik_g.setMinimumHeight(220)


        self.ik_x = QDoubleSpinBox()
        self.ik_x.setSuffix(" cm")
        self.ik_x.setLocale(QLocale(QLocale.Language.English))
        self.ik_x.setRange(-400.0, 400.0)   # ← เพิ่มบรรทัดนี้
        self.ik_x.setMinimumHeight(34)
        self.ik_x.setMinimumWidth(240)

        self.ik_y = QDoubleSpinBox()
        self.ik_y.setSuffix(" cm")
        self.ik_y.setLocale(QLocale(QLocale.Language.English))
        self.ik_y.setRange(-400.0, 400.0)   # ← เพิ่มบรรทัดนี้
        self.ik_y.setMinimumHeight(34)
        self.ik_y.setMinimumWidth(400)

        
        btn_ik = QPushButton("Go to XY"); btn_ik.clicked.connect(self._on_go_xy)

        lab_x = QLabel("X (cm)")
        lab_y = QLabel("Y (cm)")
        lab_elbow = QLabel("Elbow")

        for lab in (lab_x, lab_y, lab_elbow):
            lab.setStyleSheet("color:#000000; font-size:14px;")

        ik_layout.addWidget(lab_x, 0, 0)
        ik_layout.addWidget(self.ik_x, 0, 1)

        ik_layout.addWidget(lab_y, 1, 0)
        ik_layout.addWidget(self.ik_y, 1, 1)

        ik_layout.addWidget(btn_ik, 3, 0, 1, 2)

        right_v.addWidget(ik_g)

        # ----- CAMERA (SAME WIDGETS, MOVED DOWN) -----
        cam_g = QGroupBox("Camera")
        cam_g.setStyleSheet(self._group_style())
        cam_layout = QVBoxLayout(cam_g)

        cam_ctrl = QHBoxLayout()
        self.cam_device_combo = QComboBox(); self.cam_device_combo.setFixedWidth(160)
        self.scan_cam_btn = QPushButton("Scan Cameras"); self.scan_cam_btn.setFixedWidth(120)
        self.scan_cam_btn.clicked.connect(lambda: self._scan_cameras(max_devices=8))
        self.btn_cam_open = QPushButton("Open Camera"); self.btn_cam_open.setFixedWidth(120); self.btn_cam_open.clicked.connect(self._on_cam_open)
        self.btn_cam_close = QPushButton("Close Camera"); self.btn_cam_close.setFixedWidth(120); self.btn_cam_close.setEnabled(False); self.btn_cam_close.clicked.connect(self._on_cam_close)
        cam_ctrl.addWidget(self.cam_device_combo); cam_ctrl.addWidget(self.scan_cam_btn); cam_ctrl.addWidget(self.btn_cam_open); cam_ctrl.addWidget(self.btn_cam_close); cam_ctrl.addStretch()
        cam_layout.addLayout(cam_ctrl)

        cam_sl = QHBoxLayout()
        self.label_bright = QLabel("Brightness: 0"); self.label_bright.setFixedWidth(120)
        self.slider_bright = QSlider(Qt.Orientation.Horizontal); self.slider_bright.setRange(-100,100); self.slider_bright.setValue(0); self.slider_bright.setEnabled(False); self.slider_bright.setFixedWidth(200)
        self.label_sharp = QLabel("Sharpness: 0.0"); self.label_sharp.setFixedWidth(120)
        self.slider_sharp = QSlider(Qt.Orientation.Horizontal); self.slider_sharp.setRange(0,50); self.slider_sharp.setValue(0); self.slider_sharp.setEnabled(False); self.slider_sharp.setFixedWidth(180)
        cam_sl.addWidget(self.label_bright); cam_sl.addWidget(self.slider_bright); cam_sl.addWidget(self.label_sharp); cam_sl.addWidget(self.slider_sharp)
        cam_layout.addLayout(cam_sl)

        self.image_label = QLabel("Camera Off")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(420, 420)
        self.image_label.setStyleSheet("background:#1f2933;border:2px solid #323b42;border-radius:6px;color:#cfe8e5; font-size:13px;")
        cam_layout.addWidget(self.image_label)

        right_v.addWidget(cam_g)
        right_v.addStretch(1)

        main_h.addLayout(right_v, 1)

        self._apply_styles()

        # connect existing signals (UNCHANGED)
        for k in self.sliders:
            self.sliders[k].valueChanged.connect(self._make_slider_changed(k))
        for k in self.spinboxes:
            self.spinboxes[k].valueChanged.connect(self._make_spin_changed(k))
        self.slider_bright.valueChanged.connect(self._on_brightness_changed)
        self.slider_sharp.valueChanged.connect(self._on_sharp_changed)

    # ---------------- styles (UNCHANGED) ----------------
    def _group_style(self, bold=False):
        if bold:
            return ('QGroupBox { font-weight:700; border:2px solid rgba(140,220,200,0.42); border-radius:8px; margin-top:6px; padding:10px; font-size:14px; color:#000000; }')
        return ('QGroupBox { font-weight:700; border:2px solid rgba(60,100,95,0.18); border-radius:8px; margin-top:6px; padding:10px; font-size:14px; color:#000000; }')

    def _apply_styles(self):
        self.setStyleSheet(r"""
            QMainWindow { background: #17222a; color: #e6f2f1; font-family: Arial, Helvetica; font-size:14px; }
            QLabel { color: #e6f2f1; font-size:14px; }
            QPushButton { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #4b9df8, stop:1 #2b6fdc); color: white; padding:10px 12px; border-radius:6px; font-size:14px; border:2px solid rgba(45,105,210,0.22); font-weight:600; }
            QPushButton:hover { background: #3a86f0; }
            QPushButton:disabled { background:#2b3b3b; color:#7b9b98; border:1px solid #213232; }
        """)

    # ---------------- widget sync (UNCHANGED) ----------------
    def _sync_widgets(self):
        for k in self.sliders:
            minv, maxv = JOINT_LIMITS[k]
            val = self.joints[k]
            pos = int((val - minv) / (maxv - minv) * 1000)
            self.sliders[k].setValue(pos)
            if k == 'd4':
                self.spinboxes[k].setValue(val / 10.0)
            else:
                self.spinboxes[k].setValue(val)

        

    def _make_slider_changed(self, key):
        def on_change(pos):
            minv, maxv = JOINT_LIMITS[key]
            value_mm = minv + (pos / 1000.0) * (maxv - minv)

            if key == 'd4':
                self.spinboxes[key].setValue(value_mm )  # mm → cm
                self.joints[key] = value_mm                    # เก็บ mm
            else:
                self.spinboxes[key].setValue(round(value_mm, 2))
                self.joints[key] = value_mm
            self._update_fk_display() 

        return on_change


    def _make_spin_changed(self, key):
        def on_change(val):
            minv, maxv = JOINT_LIMITS[key]

            if key == 'd4':
                value_mm = val    # cm → mm
                pos = int((value_mm - minv) / (maxv - minv)* 1000)
                self.sliders[key].setValue(pos)
                self.joints[key] = value_mm
            else:
                pos = int((val - minv) / (maxv - minv)* 1000)
                self.sliders[key].setValue(pos)
                self.joints[key] = val
            self._update_fk_display()  
        return on_change

    
    # ---------------- IK action (NEW) ----------------
    def _on_go_xy(self):
        x_cm = self.ik_x.value()
        y_cm = self.ik_y.value()

        try:
            t1_ik, t2_ik = inverse_matrix(x_cm, y_cm)

        # IK → DH
            t1_dh = t1_ik
            t2_dh = t2_ik

        # === BLOCK SIGNALS (สำคัญ) ===
            self.spinboxes['theta1'].blockSignals(True)
            self.spinboxes['theta2'].blockSignals(True)
            self.sliders['theta1'].blockSignals(True)
            self.sliders['theta2'].blockSignals(True)

            self.spinboxes['theta1'].setValue(t1_dh)
            self.spinboxes['theta2'].setValue(t2_dh)

        # === UNBLOCK ===
            self.spinboxes['theta1'].blockSignals(False)
            self.spinboxes['theta2'].blockSignals(False)
            self.sliders['theta1'].blockSignals(False)
            self.sliders['theta2'].blockSignals(False)

        # update internal state + FK
            self.joints['theta1'] = t1_dh
            self.joints['theta2'] = t2_dh
            self._update_fk_display()

            self._log(
                f"[IK] X={x_cm:.2f}, Y={y_cm:.2f} → θ1={t1_dh:.2f}, θ2={t2_dh:.2f}"
        )

        except ValueError as e:
            QMessageBox.warning(self, "IK Error", str(e))


    # ---------------- existing actions (UNCHANGED) ----------------
    def _on_send(self):
        th1 = self.joints['theta1']
        th2 = self.joints['theta2']

        try:
            from servo_controller import move_slow_link1, move_slow_link2

            move_slow_link1(th1)
            move_slow_link2.move(th2)

            self._log(
                f"[SEND] θ1={th1:.2f}°, θ2={th2:.2f}° → SERVO MOVED"
            )

        except Exception as e:
            self._log(f"[SEND] (SIMULATED) {e}")


    def _on_home(self):
        for k in self.joints:
            self.joints[k] = 0.0
            self.spinboxes[k].setValue(0.0)
        self._log("[ACTION] Home (all zeros)")
        
        

    def _on_stop(self):
        self._log("[ACTION] EMERGENCY STOP (simulated)")

    def _log(self, text):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"{ts}  {text}")

    # ---------------- camera handlers (UNCHANGED) ----------------
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
            except:
                try:
                    if cap and cap.isOpened(): cap.release()
                except: pass
        if not found:
            self.cam_device_combo.addItem("No camera", -1)
        else:
            for idx in found:
                self.cam_device_combo.addItem(f"Camera {idx}", idx)
            self.cam_device_combo.setCurrentIndex(0)
        self._log(f"[CAM] Scan finished: found {len(found)} device(s)")

    def _on_cam_open(self):
        if not HAS_OPENCV:
            QMessageBox.warning(self, "Camera", "OpenCV not installed.")
            return
        idx = self.cam_device_combo.currentData()
        if idx is None or idx < 0:
            QMessageBox.warning(self, "Camera", "No valid camera selected.")
            return
        self.camera_thread = CameraThread(idx)
        self.camera_thread.frame_ready.connect(self._on_camera_frame)
        self.camera_thread.error_occurred.connect(self._on_camera_error)
        self.camera_thread.set_brightness(self.slider_bright.value())
        self.camera_thread.set_sharpness(self.slider_sharp.value()/10.0)
        self.camera_thread.start()
        self.btn_cam_open.setEnabled(False); self.btn_cam_close.setEnabled(True)
        self.slider_bright.setEnabled(True); self.slider_sharp.setEnabled(True)
        self._log(f"[CAM] Opened device {idx}")

    def _on_cam_close(self):
        if self.camera_thread:
            self.camera_thread.stop(); self.camera_thread.wait(1000)
            self.camera_thread = None
        self.image_label.setText("Camera Off")
        self.btn_cam_open.setEnabled(True); self.btn_cam_close.setEnabled(False)
        self.slider_bright.setEnabled(False); self.slider_sharp.setEnabled(False)
        self._log("[CAM] Closed")

    def _on_brightness_changed(self, v):
        self.label_bright.setText(f"Brightness: {v}")
        if self.camera_thread:
            self.camera_thread.set_brightness(v)

    def _on_sharp_changed(self, v):
        s = v/10.0
        self.label_sharp.setText(f"Sharpness: {s:.1f}")
        if self.camera_thread:
            self.camera_thread.set_sharpness(s)

    def _on_camera_frame(self, frame):
        h,w,ch = frame.shape
        qimg = QImage(frame.data, w, h, ch*w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _on_camera_error(self, msg):
        QMessageBox.critical(self, "Camera error", msg)
        self._on_cam_close()

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop(); self.camera_thread.wait(1000)
        event.accept()

# ---------------- run ----------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ScaraCameraApp()
    win.show()
    sys.exit(app.exec())