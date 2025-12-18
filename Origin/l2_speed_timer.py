import serial
import time

ser = serial.Serial("COM11", 9600, timeout=1)
time.sleep(2)

print("c = CW")
print("a = CCW")
dir_cmd = input("direction: ").strip()

ser.write(dir_cmd.encode())
time.sleep(0.1)

print("Press ENTER to START")
input()

ser.write(b's')
t0 = time.time()
print("RUNNING... Press Ctrl+C to STOP")

try:
    while True:
        time.sleep(0.1)

except KeyboardInterrupt:
    ser.write(b'e')
    t1 = time.time()
    print("\nSTOP")
    print(f"Run time = {t1 - t0:.3f} sec")
