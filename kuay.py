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
from Origin.coordinate_origin import inverse_matrix, tranformation_matrix
import numpy as np
from checker_ai import predict


def fk_from_coordinate(theta1_deg, theta2_deg):
    t1 = np.deg2rad(theta1_deg)
    t2 = np.deg2rad(theta2_deg)

    T1, _, _ = tranformation_matrix(t1, 0)
    T2, _, _ = tranformation_matrix(t2, 1)

    T = T1 @ T2

    x_mm = T[0, 3]
    y_mm = T[1, 3]

    return x_mm, y_mm
def fk_to_board_frame(x_fk, y_fk, x_ref=None):
    # robot frame → board world frame (top-left origin)
    
    board_x = x_fk + 20.0
    board_y = 40.5 - y_fk

    return board_x, board_y





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
# ===== CHECKERBOARD REAL SIZE =====
BOARD_SIZE_CM = 40.0
BOARD_CELLS = 8
CELL_SIZE_CM = BOARD_SIZE_CM / BOARD_CELLS  # = 5.0 cm
BOARD_PX = 400
CELL_PX = BOARD_PX // 8   # = 50 px ต่อช่อง


# ตำแหน่งมุมซ้ายล่างของกระดาน (ปรับหน้างานจริง)
BOARD_ORIGIN_X =  20
BOARD_ORIGIN_Y = 0.5





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
class ClickableCell(QLabel):
    clicked = pyqtSignal(int, int)

    def __init__(self, r, c, parent=None):
        super().__init__("", parent)
        self.r = r
        self.c = c

    def mousePressEvent(self, event):
        self.clicked.emit(self.r, self.c)

class ScaraCameraApp(QMainWindow):

    def _apply_offset(self):
        self.offset_apply_x = self.off_x_box.value()
        self.offset_apply_y = self.off_y_box.value()

        self._log(f"[OFFSET] Updated → X={self.offset_apply_x:.2f}, Y={self.offset_apply_y:.2f}")

    def ai_move(self):
        try:
            board_list = self.board_state_to_list()
            move = predict.predict_move(board_list)

            if move is None:
                self._log("[AI] No valid move")
                return

            fr = move["from"]
            to = move["to"]

            fx, fy = self.board_cell_to_xy(fr[0], fr[1])
            tx, ty = self.board_cell_to_xy(to[0], to[1])

            self._log(f"[AI] MOVE from r={fr} -> r={to}")
            self._log(f"[AI] XY: ({fx:.2f},{fy:.2f}) → ({tx:.2f},{ty:.2f})")

            # แล้วสั่ง IK
            self.ik_x.setValue(fx)
            self.ik_y.setValue(fy)
            self._on_go_xy()

            # ทำหยิบชิ้น
            self._on_gripper_close()

            self.ik_x.setValue(tx)
            self.ik_y.setValue(ty)
            self._on_go_xy()

            self._on_gripper_open()

        except Exception as e:
            self._log(f"[AI ERROR] {e}")


    def _draw_board_overlay(self, img):
        h, w, _ = img.shape

        cell_h = h // 8
        cell_w = w // 8

        for r in range(8):
            for c in range(8):

                x1 = c * cell_w
                y1 = r * cell_h
                x2 = (c + 1) * cell_w
                y2 = (r + 1) * cell_h

                # ช่องดำ / ช่องขาว
                if (r + c) % 2 == 1:
                    color = (0, 0, 0)      # ดำ
                    alpha = 0.40          # ทึบกว่า
                else:
                    color = (255, 255, 255)
                    alpha = 0.18          # โปร่งกว่า

                overlay = img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)



    def _paint_at(self, pos):
        if self.ignore_mask is None or self.image_label.pixmap() is None:
            return

        # ----- ขนาด QLabel -----
        label_w = self.image_label.width()
        label_h = self.image_label.height()

        # ----- ขนาดภาพจริง -----
        img_h, img_w = self.ignore_mask.shape

        # ----- คำนวณขนาด pixmap ที่ถูก scale -----
        pixmap = self.image_label.pixmap()
        pm_w = pixmap.width()
        pm_h = pixmap.height()

        # offset (ตรงนี้คือจุดสำคัญ)
        offset_x = (label_w - pm_w) // 2
        offset_y = (label_h - pm_h) // 2

        x = int(pos.x() - offset_x)
        y = int(pos.y() - offset_y)

        # อยู่นอกภาพ → ไม่ต้องทำ
        if x < 0 or y < 0 or x >= pm_w or y >= pm_h:
            return

        # ----- map ไปยังพิกัดภาพจริง -----
        fx = int(x * img_w / pm_w)
        fy = int(y * img_h / pm_h)

        cv2.circle(
            self.ignore_mask,
            (fx, fy),
            self.brush_radius,
            0,      # 0 = ignore
            -1
        )

            



    def detect_black_corner_markers(self, frame):
        """
        ตรวจจับสี่เหลี่ยมดำ (corner markers)
        return: list ของ (cx, cy, approx)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # ช่วงสีดำ (ปรับได้หน้างาน)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 70])

        mask = cv2.inRange(hsv, lower_black, upper_black)

        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        markers = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:   # กัน noise
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4,2)
                cx = pts[:,0].mean()
                cy = pts[:,1].mean()
                markers.append((cx, cy, pts))

        return markers
    
    def get_board_corners_from_markers(self, markers):
        """
        markers: [(cx, cy, pts), ...]
        return: 4x2 array (TL, TR, BR, BL) หรือ None
        """
        if len(markers) < 4:
            return None

        # เอา 4 อันที่พื้นที่ใหญ่สุด
        markers = sorted(
            markers,
            key=lambda m: cv2.contourArea(m[2]),
            reverse=True
        )[:4]

        centers = [(m[0], m[1]) for m in markers]
        pts_all = [m[2] for m in markers]

        centers = np.array(centers)

        s = centers.sum(axis=1)
        diff = np.diff(centers, axis=1).reshape(-1)

        tl = pts_all[np.argmin(s)]
        br = pts_all[np.argmax(s)]
        tr = pts_all[np.argmin(diff)]
        bl = pts_all[np.argmax(diff)]

        return np.array([
            tl.mean(axis=0),
            tr.mean(axis=0),
            br.mean(axis=0),
            bl.mean(axis=0)
        ], dtype=np.float32)



    def _label_to_frame(self, x, y, frame):
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        h, w, _ = frame.shape

        fx = int(x * w / label_w)
        fy = int(y * h / label_h)
        return fx, fy

    

    def _draw_corner_marks(self, img, x1, y1, x2, y2, color=(255, 0, 0), size=30, thickness=3):
        """
        วาดติ๊ก 4 มุมแทนกรอบสี่เหลี่ยม
        """
        # มุมซ้ายบน
        cv2.line(img, (x1, y1), (x1 + size, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + size), color, thickness)

        # มุมขวาบน
        cv2.line(img, (x2, y1), (x2 - size, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + size), color, thickness)

        # มุมซ้ายล่าง
        cv2.line(img, (x1, y2), (x1 + size, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - size), color, thickness)

        # มุมขวาล่าง
        cv2.line(img, (x2, y2), (x2 - size, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - size), color, thickness)



    def _draw_green_grid(self, img, rows=8, cols=8):
        """
        วาดกริดสีเขียวทับบนภาพ (ใช้ช่วยจัดแนวกระดาน)
        """
        h, w, _ = img.shape

        for i in range(1, rows):
            y = int(i * h / rows)
            cv2.line(img, (0, y), (w, y), (0, 255, 0), 1)

        for j in range(1, cols):
            x = int(j * w / cols)
            cv2.line(img, (x, 0), (x, h), (0, 255, 0), 1)


    def _on_cam_reset(self):
        # ===== RESET ZOOM =====
        self.zoom_factor = 1.0
        self.slider_zoom.blockSignals(True)
        self.slider_zoom.setValue(10)   # 1.0x
        self.slider_zoom.blockSignals(False)
        self.label_zoom.setText("Zoom: 1.0x")

        
        self.full_detect = True      # 🔥 บังคับ detect เต็มภาพ
        self.zoom_factor = 1.0       # 🔥 กัน error ตอน zoom
        self.selecting_roi = False
        self.roi_locked = False
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None


    # ===== เปิดโหมด detect เต็มภาพ =====
        self.full_detect = True

        self._log("[CAM] Reset view → Full frame detect")


    def _choose_color(self, color):
        self.my_color = color
        self._log(f"[BOARD] My color = {color}")
    
    def board_state_to_list(self):
        """
        return: 8x8 list
        1  = หมากเรา
        -1 = หมากฝั่งตรงข้าม
        0  = ว่าง / ช่องขาว
        """
        if self.last_board_state is None:
            raise ValueError("Board not detected yet")

        if self.my_color is None:
            raise ValueError("Please choose your color first")

        board = []

        for r in range(8):
            row = []
            for c in range(8):
                if not self.is_dark_square(r, c):
                    row.append(0)
                    continue

                piece = self.last_board_state[r][c]

                if piece is None:
                    row.append(0)
                elif piece == self.my_color:
                    row.append(1)
                else:
                    row.append(-1)

            board.append(row)

        return board
    
    def _send_board_state(self):
        try:
            board_list = self.board_state_to_list()

            self._log("[BOARD] Send board state:")
            for row in board_list:
                self._log(str(row))

            # 🔴 ตรงนี้เอาไปส่งต่อ AI / logic / file / socket ได้
            # example:
            # self.ai_engine.update_board(board_list)

        except Exception as e:
            QMessageBox.warning(self, "Board Error", str(e))




    def _reset_roi(self):
        """
        รีเซต ROI กลับไปภาพเต็ม
        """
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.roi_locked = False
        self.selecting_roi = False
        self.full_detect = True   # ✅ detect ทั้งภาพโดยไม่ต้อง ROI

        self.zoom_factor = 1.0

        # กลับไปใช้ภาพจากกล้องตรง ๆ
        

        self._log("[ROI] Reset to full frame")



    def _roi_to_frame_coords(self, frame):
        """
        แปลง ROI จากพิกัด QLabel → พิกัด frame (numpy)
        """
        label_w = self.image_label.width()
        label_h = self.image_label.height()

        frame_h, frame_w, _ = frame.shape

        x1, y1, x2, y2 = self.roi_rect

        scale_x = frame_w / label_w
        scale_y = frame_h / label_h

        fx1 = int(min(x1, x2) * scale_x)
        fy1 = int(min(y1, y2) * scale_y)
        fx2 = int(max(x1, x2) * scale_x)
        fy2 = int(max(y1, y2) * scale_y)

        # clamp กันหลุด
        fx1 = max(0, min(frame_w - 1, fx1))
        fx2 = max(0, min(frame_w, fx2))
        fy1 = max(0, min(frame_h - 1, fy1))
        fy2 = max(0, min(frame_h, fy2))

        return fx1, fy1, fx2, fy2



    def eventFilter(self, obj, event):
        if obj is self.image_label and self.ignore_mask is not None:

            if event.type() == event.Type.MouseButtonPress and event.buttons() & Qt.MouseButton.LeftButton:
                self.painting = True
                self._paint_at(event.position())
                return True

            elif event.type() == event.Type.MouseMove and self.painting:
                self._paint_at(event.position())
                return True

            elif event.type() == event.Type.MouseButtonRelease:
                self.painting = False
                return True

        return super().eventFilter(obj, event)





    def detect_board_corners(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                return self._order_points(approx.reshape(4, 2))

        return None
    
    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect
    
    def warp_board_auto(self, frame, corners):
        BOARD_PX = 400

        dst = np.array([
            [0, 0],
            [BOARD_PX, 0],
            [BOARD_PX, BOARD_PX],
            [0, BOARD_PX]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(frame, M, (BOARD_PX, BOARD_PX))
        return warped







    def is_dark_square(self, r, c):
    # ช่องดำของกระดานหมากฮอส
        return (r + c) % 2 == 1
    def detect_cell_piece_color(self, cell_img):
        hsv = cv2.cvtColor(cell_img, cv2.COLOR_RGB2HSV)

    # --- สีฟ้า ---
        blue_lower = np.array([90, 80, 50])
        blue_upper = np.array([130, 255, 255])

    # --- สีแดง (2 ช่วง) ---
        red_lower1 = np.array([0, 100, 80])
        red_upper1 = np.array([10, 255, 255])

        red_lower2 = np.array([170, 100, 80])
        red_upper2 = np.array([180, 255, 255])


        mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
        mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        blue_pixels = cv2.countNonZero(mask_blue)
        red_pixels = cv2.countNonZero(mask_red)

        total = cell_img.shape[0] * cell_img.shape[1]

        blue_ratio = blue_pixels / total
        red_ratio  = red_pixels / total

        if blue_ratio > 0.15:
            return "BLUE"
        elif red_ratio > 0.15:
            return "RED"
        else:
            return None

    def detect_board_state(self, board_img):
        state = [[None]*8 for _ in range(8)]

        for r in range(8):
            for c in range(8):

            # 🔴 ตรวจเฉพาะช่องดำ
                if not self.is_dark_square(r, c):
                    state[r][c] = None
                    continue

                y0 = r * CELL_PX
                y1 = (r + 1) * CELL_PX
                x0 = c * CELL_PX
                x1 = (c + 1) * CELL_PX

                margin = 10   # 🔥 ตัดขอบช่องทิ้ง
                cell = board_img[
                    y0 + margin : y1 - margin,
                    x0 + margin : x1 - margin
                ]
                state[r][c] = self.detect_cell_piece_color(cell)


        return state
    
    def update_checkerboard_ui(self, state):
        for r in range(8):
            for c in range(8):
                cell = self.board_cells[r][c]

            # ช่องขาว → ไม่ต้องแสดงหมาก
                if not self.is_dark_square(r, c):
                    cell.setText("")
                    cell.setStyleSheet("background:#eeeeee;")
                    continue

                piece = state[r][c]

                if piece == "RED":
                    cell.setText("●")
                    cell.setStyleSheet(
                        "background:#444444;"
                        "color:red;"
                        "font-size:24px;"
                        "border:1px solid #999;"
                    )
                elif piece == "BLUE":
                    cell.setText("●")
                    cell.setStyleSheet(
                        "background:#444444;"
                        "color:cyan;"
                        "font-size:24px;"
                        "border:1px solid #999;"
                    )
                else:
                    # ช่องดำแต่ไม่มีหมาก
                    cell.setText("")
                    cell.setStyleSheet(
                        "background:#444444;"
                        "border:1px solid #999;"
                    )





    
    def crop_board_center(self, frame):
        h, w, _ = frame.shape

        # ใช้สี่เหลี่ยมจัตุรัสตรงกลางภาพ
        size = min(h, w)
        cx = w // 2
        cy = h // 2
        half = size // 2

        board = frame[
            cy-half:cy+half,
            cx-half:cx+half
        ]
        return board
    
    
    




    def _update_fk_display(self):
        th1 = self.joints['theta1']
        th2 = self.joints['theta2']

    # 1) FK จาก robot frame (DH)
        x_fk, y_fk = fk_from_coordinate(th1, th2)

        x_ref = self.ik_x.value()

        x_board, y_board = fk_to_board_frame(x_fk, y_fk, x_ref)


    

    # 3) แสดงค่า "เดียวกับ IK / กระดาน"
        self.val_x.setText(f"{x_board:.2f}")
        self.val_y.setText(f"{y_board:.2f}")
        self.val_z.setText(self.joints['d4_state'])
        self.val_yaw.setText(f"{(th1 + th2):.2f}")


    def __init__(self):
        super().__init__()
                # ===== IGNORE MASK (MUST EXIST BEFORE eventFilter) =====
        self.ignore_mask = None
        self.painting = False
        self.brush_radius = 18

        
        self.setWindowTitle("SCARA Control (θ1, θ2, d4) + Camera")
        self.resize(1100, 700)

        self.joints = {
            'theta1': 0.0,
            'theta2': 0.0,
            'd4_state': 'STOP'   # หรือ 'UP' / 'DOWN'
        }
        self.gripper_state = "OPEN"

        self.camera_thread = False
        self.last_frame = None
        self.board_locked = False      # ✅ เพิ่ม
        self.board_corners = None
        self.my_color = None      # "RED" หรือ "BLUE"
        self.last_board_state = None
        self.board_history = []   # 🔥 เก็บผล detect หลายเฟรม

        self.zoom_factor = 1.0
        self.offset_x_val = 20.0     # default ค่าเดิมที่ใช้
        self.offset_y_val = 40.5

        self.full_detect = True       # 🔥 ต้องมี
  
      

        # ===== ROI STATE (เพิ่มใหม่) =====
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.roi_locked = False

        self.selecting_roi = False


        
   


        self._build_ui()
        self._sync_widgets()
        self._scan_cameras(max_devices=8)

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(lambda: None)
        self.timer.start()
    
    def _on_d4_up(self):
        try:
            from Origin.servo_controller import move_slider
            move_slider(300)   # up
        except Exception as e:
            self._log(f"[D4] UP (SIMULATED) : {e}")

        self.joints['d4_state'] = 'UP'
        self.val_z.setText("UP")

    


    def _on_d4_down(self):
        try:
            from Origin.servo_controller import move_slider
            move_slider(90)   # down
        except Exception as e:
            self._log(f"[D4] DOWN (SIMULATED) : {e}")

        self.joints['d4_state'] = 'DOWN'
        self.val_z.setText("DOWN")


    def _on_gripper_open(self):
        try:
            from Origin.servo_controller import move_gripper
            move_gripper(300)   # 🔥 แก้ตรงนี้
        except Exception as e:
            self._log(f"[GRIPPER] OPEN (SIMULATED) : {e}")

        self.gripper_state = "OPEN"
        self._log("[GRIPPER] OPEN")
        self.btn_grip_open.setEnabled(False)
        self.btn_grip_close.setEnabled(True)
  



    def _on_gripper_close(self):
        try:
            from Origin.servo_controller import move_gripper
            move_gripper(35)

        except Exception as e:
            self._log(f"[GRIPPER] CLOSE (SIMULATED) : {e}")

        self.gripper_state = "CLOSE"
        self._log("[GRIPPER] CLOSE")
        self.btn_grip_open.setEnabled(True)    # 🟢 เปิดปุ่ม OPEN
        self.btn_grip_close.setEnabled(False)  # 🔴 ปิดปุ่ม CLOSE





    def _build_ui(self):#################################################################################
        # ---------------- ROOT WITH SCROLL (NEW) ----------------
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll.setWidget(container)
        self.setCentralWidget(scroll)

        main_v = QVBoxLayout(container)
        main_v.setContentsMargins(12,12,12,12)
        main_v.setSpacing(6)

        main_h = QHBoxLayout()
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
        # ================= CHECKERBOARD UI =================
        board_g = QGroupBox()
        board_g.setStyleSheet(self._group_style())
        board_layout = QGridLayout(board_g)
        board_layout.setSpacing(2)                 # ระยะห่างระหว่างช่อง (px)
        board_layout.setContentsMargins(0, 0, 0, 0)  # ขอบรอบกระดาน


        self.board_cells = [[None]*8 for _ in range(8)]

        for r in range(8):
            for c in range(8):
                cell = ClickableCell(r, c)
                cell.clicked.connect(self._on_board_clicked)
                cell.setFixedSize(90, 90)
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setContentsMargins(0, 0, 0, 0)

                if (r + c) % 2 == 0:
                    cell.setStyleSheet("background:#eeeeee; border:1px solid #999;")
                else:
                    cell.setStyleSheet("background:#444444; color:white;")

                board_layout.addWidget(cell, r, c)
                self.board_cells[r][c] = cell
        # ===== BOARD BUTTONS =====
        board_btn_layout = QHBoxLayout()

        self.btn_choose_red = QPushButton("Choose RED")
        self.btn_choose_blue = QPushButton("Choose BLUE")
        self.btn_board_send = QPushButton("Send Board")

        self.btn_choose_red.clicked.connect(lambda: self._choose_color("RED"))
        self.btn_choose_blue.clicked.connect(lambda: self._choose_color("BLUE"))
        self.btn_board_send.clicked.connect(self._send_board_state)

        board_btn_layout.addWidget(self.btn_choose_red)
        board_btn_layout.addWidget(self.btn_choose_blue)
        board_btn_layout.addWidget(self.btn_board_send)
        # ===== WRAP BOARD + BUTTONS =====
        board_outer = QVBoxLayout()
        board_outer.addWidget(board_g)
        board_outer.addLayout(board_btn_layout)


        


    
        # left_v.addWidget(board_g)

  # ✅ สำคัญมาก
        left_v.addStretch(1)


        left_v.addStretch(1)
        # ===== BOTTOM ROW : CHECKERBOARD + CAMERA =====
        

        main_h.addLayout(left_v, 2)
        main_v.addLayout(main_h)
        #main_v.addLayout(bottom_h)


        # ---------------- RIGHT (FK + IK + CAMERA) ----------------
        right_v = QVBoxLayout()
        right_v.setSpacing(12)
        top_h = QHBoxLayout()
        top_h.setSpacing(12)

        pose_v = QVBoxLayout()
        pose_v.setSpacing(12)


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

        # right_v.addWidget(cart_g)
        


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
        lab_offx = QLabel("Offset X")
        lab_offy = QLabel("Offset Y")


        for lab in (lab_x, lab_y, lab_elbow):
            lab.setStyleSheet("color:#000000; font-size:14px;")

        ik_layout.addWidget(lab_x, 0, 0)
        ik_layout.addWidget(self.ik_x, 0, 1)

        ik_layout.addWidget(lab_y, 1, 0)
        ik_layout.addWidget(self.ik_y, 1, 1)

    
        # ===============================

        ik_layout.addWidget(btn_ik, 4, 0, 1, 2)
        # ===== OFFSET ADJUST =====
        self.off_x_box = QDoubleSpinBox()
        self.off_y_box = QDoubleSpinBox()
        self.off_x_box.setRange(-100.0, 100.0)
        self.off_y_box.setRange(-100.0, 100.0)
        self.off_x_box.setSuffix(" cm")
        self.off_y_box.setSuffix(" cm")

        btn_apply_off = QPushButton("Apply Offset")
        btn_apply_off.clicked.connect(self._apply_offset)

        ik_layout.addWidget(QLabel("Offset X"), 5, 0)
        ik_layout.addWidget(self.off_x_box, 5, 1)

        ik_layout.addWidget(QLabel("Offset Y"), 6, 0)
        ik_layout.addWidget(self.off_y_box, 6, 1)

        ik_layout.addWidget(btn_apply_off, 7, 0, 1, 2)




        #right_v.addWidget(ik_g)
        pose_v.addWidget(cart_g)
        pose_v.addWidget(ik_g)
        right_v.addLayout(pose_v)

        # ===== GRIPPER CONTROL (NEW) =====
        grip_g = QGroupBox("Gripper Control")
        grip_g.setStyleSheet(self._group_style())
        grip_layout = QHBoxLayout(grip_g)

        self.btn_grip_open = QPushButton("OPEN")
        self.btn_grip_close = QPushButton("CLOSE")

        self.btn_grip_open.clicked.connect(self._on_gripper_open)
        self.btn_grip_close.clicked.connect(self._on_gripper_close)
        self.btn_grip_close.setEnabled(False)  # เริ่มต้นเป็น OPEN


        self.btn_grip_open.setStyleSheet("background:#4caf50; font-weight:700;")
        self.btn_grip_close.setStyleSheet("background:#e53935; font-weight:700;")

        grip_layout.addWidget(self.btn_grip_open)
        grip_layout.addWidget(self.btn_grip_close)

        pose_v.addWidget(grip_g)


        # ----- CAMERA (SAME WIDGETS, MOVED DOWN) -----
        cam_g = QGroupBox("Camera")
        cam_g.setStyleSheet(self._group_style())
        cam_layout = QVBoxLayout(cam_g)

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
        self.btn_cam_reset = QPushButton("Reset View")
        self.btn_cam_reset.setFixedWidth(120)
        self.btn_cam_reset.clicked.connect(self._on_cam_reset)


        
        
        

        cam_ctrl.addWidget(self.cam_device_combo)
        cam_ctrl.addWidget(self.scan_cam_btn)
        cam_ctrl.addWidget(self.btn_cam_open)
        cam_ctrl.addWidget(self.btn_cam_close)
        cam_ctrl.addWidget(self.btn_cam_reset)

        
        cam_ctrl.addStretch()

        cam_layout.addLayout(cam_ctrl)


        cam_sl = QHBoxLayout()
        self.label_bright = QLabel("Brightness: 0"); self.label_bright.setFixedWidth(120)
        self.slider_bright = QSlider(Qt.Orientation.Horizontal); self.slider_bright.setRange(-100,100); self.slider_bright.setValue(0); self.slider_bright.setEnabled(False); self.slider_bright.setFixedWidth(200)
        self.label_sharp = QLabel("Sharpness: 0.0"); self.label_sharp.setFixedWidth(120)
        self.slider_sharp = QSlider(Qt.Orientation.Horizontal); self.slider_sharp.setRange(0,50); self.slider_sharp.setValue(0); self.slider_sharp.setEnabled(False); self.slider_sharp.setFixedWidth(180)
        # ===== ZOOM SLIDER =====
        zoom_layout = QHBoxLayout()

        self.label_zoom = QLabel("Zoom: 1.0x")
        self.label_zoom.setFixedWidth(120)

        self.slider_zoom = QSlider(Qt.Orientation.Horizontal)
        self.slider_zoom.setRange(10, 40)   # 1.0x – 4.0x
        self.slider_zoom.setValue(10)
        self.slider_zoom.setFixedWidth(200)

        zoom_layout.addWidget(self.label_zoom)
        zoom_layout.addWidget(self.slider_zoom)

        cam_layout.addLayout(zoom_layout)

        cam_sl.addWidget(self.label_bright); cam_sl.addWidget(self.slider_bright); cam_sl.addWidget(self.label_sharp); cam_sl.addWidget(self.slider_sharp)
        cam_layout.addLayout(cam_sl)

        self.image_label = QLabel("Camera Off")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(420, 420)
        self.image_label.setStyleSheet("background:#1f2933;border:2px solid #323b42;border-radius:6px;color:#cfe8e5; font-size:13px;")
        cam_layout.addWidget(self.image_label)
        self.image_label.setMouseTracking(True)
        self.image_label.installEventFilter(self)
        self.image_label.setMouseTracking(True)
        
        
        # ===== BOTTOM ROW : CHECKERBOARD + CAMERA =====
        bottom_h = QHBoxLayout()
        bottom_h.setSpacing(12)
        bottom_h.setContentsMargins(0, 0, 0, 0)

        bottom_h.addLayout(board_outer, 1)
        bottom_h.addWidget(cam_g, 1)
        main_v.addLayout(bottom_h)


        #right_v.addWidget(cam_g)
        right_v.addStretch(1)


        main_h.addLayout(right_v, 1)

        self._apply_styles()
        
       

                # ============ END CHECKERBOARD ============
        
        #top_h.addWidget(board_g, 1)

        # สีตารางหมากฮอส
                
# ===================================================


        # connect existing signals (UNCHANGED)
        for k in self.sliders:
            self.sliders[k].valueChanged.connect(self._make_slider_changed(k))
        for k in self.spinboxes:
            self.spinboxes[k].valueChanged.connect(self._make_spin_changed(k))
        self.slider_bright.valueChanged.connect(self._on_brightness_changed)
        self.slider_sharp.valueChanged.connect(self._on_sharp_changed)
        self.slider_zoom.valueChanged.connect(self._on_zoom_changed)


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

        # 1) ค่าจริงให้มาจาก spinbox เท่านั้น
            self.joints[key] = round(val, 2)

        # 2) slider เป็นแค่ตัวแสดงตำแหน่ง
            self.sliders[key].blockSignals(True)
            pos = round((val - minv) / (maxv - minv) * 1000)
            self.sliders[key].setValue(pos)
            self.sliders[key].blockSignals(False)

        # 3) update FK
            self._update_fk_display()

        return on_change

    
    # ---------------- IK action (NEW) ----------------
    def _on_go_xy(self):
        x_cm = self.ik_x.value()
        y_cm = self.ik_y.value()

        try:
            # 1) เรียก inverse kinematics
            t1_ik, t2_ik = inverse_matrix(x_cm, y_cm)

            # 2) ถ้า θ1 ติดลบ -> แปลงเป็น elbow-up
            if t1_ik < 0:
                # 2.1 ปรับ θ1 ให้เข้า zone 0–180
                t1_ik = 180 + t1_ik

                # 2.2 สลับสัญญาณ θ2
                t2_ik = -t2_ik



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
            
        # ================= CHECKERBOARD ACTION =================
    def board_cell_to_xy(self, r, c):
        cell = 40.0 / 8.0   # 5 cm

        world_x = (c + 0.5) * cell
        world_y = (r + 0.5) * cell

        robot_x = world_x - 20.0 + self.offset_apply_x
        robot_y = 40.5 - world_y + self.offset_apply_y

        return robot_x, robot_y



    def _on_board_clicked(self, r, c):
        x, y = self.board_cell_to_xy(r, c)

        self._log(f"[BOARD] r={r}, c={c} → X={x:.2f}, Y={y:.2f}")

        # ใส่ค่าเข้า IK
        self.ik_x.setValue(x)
        self.ik_y.setValue(y)

        # สั่งให้หุ่นไปตำแหน่งนั้น
        self._on_go_xy()



    # ---------------- existing actions (UNCHANGED) ----------------
    def _on_send(self):
        
        
        th1 = self.joints['theta1']
        th2 = self.joints['theta2']

        try:
            from Origin.servo_controller import move_slow_link1, move_slow_link2
            move_link2 = move_slow_link2()

            move_slow_link1(th1)
            move_link2.move(th2)

            self._log(
                f"[SEND] θ1={th1:.2f}°, θ2={th2:.2f}° → SERVO MOVED"
            )

        except Exception as e:
            self._log(f"[SEND] (SIMULATED) {e}")


    def _on_home(self):
    # reset เฉพาะ joint ที่มี slider / spinbox
        for k in ('theta1', 'theta2'):
            self.joints[k] = 0.0

        # block signal กัน loop
            self.spinboxes[k].blockSignals(True)
            self.sliders[k].blockSignals(True)

            self.spinboxes[k].setValue(0.0)
            self.sliders[k].setValue(0)

            self.spinboxes[k].blockSignals(False)
            self.sliders[k].blockSignals(False)

    # reset d4 แยก
        self.joints['d4_state'] = 'STOP'
        self.val_z.setText("STOP")

        self._update_fk_display()
        self._log("[ACTION] Home (theta1, theta2 = 0, D4 = STOP)")

        
        

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
        #self.camera_thread.error_occurred.connect(self._on_camera_error)
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

    def _on_zoom_changed(self, v):
        """
        รับค่าจาก slider zoom
        """
        self.zoom_factor = v / 10.0
        self.label_zoom.setText(f"Zoom: {self.zoom_factor:.1f}x")


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
        try:
            # ===== 1) crop to board =====
            display_frame = self.crop_board_center(frame)

            # ===== 2) zoom =====
            if self.zoom_factor > 1.0:
                h, w, _ = display_frame.shape
                new_w = int(w / self.zoom_factor)
                new_h = int(h / self.zoom_factor)

                cx = w // 2
                cy = h // 2

                x1 = max(0, cx - new_w // 2)
                y1 = max(0, cy - new_h // 2)
                x2 = min(w, cx + new_w // 2)
                y2 = min(h, cy + new_h // 2)

                display_frame = display_frame[y1:y2, x1:x2]
                display_frame = cv2.resize(display_frame, (w, h))

            # ===== 3) detect board using zoomed frame =====
            if self.full_detect:
                board_img = cv2.resize(display_frame, (BOARD_PX, BOARD_PX))
                state = self.detect_board_state(board_img)

                self.board_history.append(state)
                if len(self.board_history) > 5:
                    self.board_history.pop(0)

                stable = [[None]*8 for _ in range(8)]
                for r in range(8):
                    for c in range(8):
                        vals = [b[r][c] for b in self.board_history if b[r][c] is not None]
                        if vals:
                            stable[r][c] = max(set(vals), key=vals.count)

                self.last_board_state = stable
                self.update_checkerboard_ui(stable)

            # ===== 4) draw green grid AFTER zoom =====
            self._draw_green_grid(display_frame, 8, 8)

            # ===== 5) show p on QLabel =====
            h, w, ch = display_frame.shape
            qimg = QImage(display_frame.tobytes(), w, h, ch*w, QImage.Format.Format_RGB888)

            self.image_label.setPixmap(
                QPixmap.fromImage(qimg).scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )

        except Exception as e:
            self._log(f"[CAM ERROR] {e}")
            traceback.print_exc()









if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ScaraCameraApp()
    win.show()
    sys.exit(app.exec())  

