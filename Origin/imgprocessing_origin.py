import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("best.pt")

# ลำดับ: ซ้ายบน, ขวาบน, ขวาล่าง, ซ้ายล่าง
BOARD_CORNERS = np.float32([
    [120, 80],
    [520, 90],
    [530, 480],
    [110, 470]
])

BOARD_SIZE = 400  # pixel (กระดานจะถูก warp เป็น 400x400)

def warp_board(frame):
    dst = np.float32([
        [0, 0],
        [BOARD_SIZE, 0],
        [BOARD_SIZE, BOARD_SIZE],
        [0, BOARD_SIZE],
    ])
    M = cv2.getPerspectiveTransform(BOARD_CORNERS, dst)
    warped = cv2.warpPerspective(frame, M, (BOARD_SIZE, BOARD_SIZE))
    
    return warped

def detect_pieces(board_img):
    result = model(board_img, conf=.5, verbose=False)
    detection = []

    for r in result:
        for box in r.boxes:
            cls = int(box.cls[])