import serial
import time
from coordinate_origin import inverse_matrix

# เชื่อมต่อ Arduino
ser = serial.Serial('COM11', 9600, timeout=1)
time.sleep(2)  # ให้ Arduino reset เสร็จ

def move_slow_link1(deg):
    deg = (180 - deg) * (300/180)
    ser.write(f"L1:{deg}\n".encode())

def move_slider(deg):
    ser.write(f"SL:{deg}\n".encode())

def move_gripper(deg):
    ser.write(f"GR:{deg}\n".encode())

class move_slow_link2:
    def __init__(self):
        self.pos = 0

    def move(self, deg):
        ser.write(f"L2:{deg}\n".encode())
        self.pos = deg

def play_chess(pos):
    # ----- const. -----
    grab = 55
    ungrab = 300
    up = 300
    down = 90
    home_link1 = 300
    home_link2 = 0
    # ------------------
    
    
    link2 = move_slow_link2()
    grab_pos = pos[0]
    place_pos = pos[1]
    theta1, theta2 = inverse_matrix(grab_pos)
    move_slow_link1(theta1)
    time.sleep(10)
    link2.move(theta2)
    time.sleep(10)
    move_slider(down)
    time.sleep(10)
    move_gripper(grab)
    time.sleep(10)
    move_slider(up)
    
    theta1, theta2 = inverse_matrix(place_pos)
    move_slow_link1(theta1)
    time.sleep(10)
    link2.move(theta2)
    time.sleep(10)
    move_slider(down)
    time.sleep(10)
    move_gripper(ungrab)
    time.sleep(10)
    move_slider(up)
    
    move_slow_link1(home_link1)
    link2.move(home_link2)

# if __name__ == "__main__":
    # --------- test link2 ---------
    # link2 = Link2()

    # link2.move(90)
    # time.sleep(4)
    # link2.move(0)

    # --------- test link1 --------
    # move_slow_link1(180)
    # time.sleep(5)
    # move_slow_link1(0)
    # time.sleep(8)
    # move_slow_link1(300)
    # time.sleep(5)
    # move_slow_link1(180)
    # time.sleep(10)
    # move_slow_link1(0)
    
    #  --------- test gripper ---------
    # move_gripper(300) # fully open
    # time.sleep(3)
    # move_gripper(55)   # fully close
    # time.sleep(3)
    # move_gripper(300) # fully open
    
    # --------- test slider ---------
    # move_slider(90)   # down
    # time.sleep(1)
    # move_slider(300) # up
   
   



    