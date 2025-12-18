#include <Servo.h>

/* =================================================
   SERVO OBJECTS
   ================================================= */
Servo servo1;   // Link 1 (300 deg)
Servo servo2;   // Link 2 (360 deg continuous)
Servo slider;   // Slider / Z axis (300 deg)
Servo gripper;  // Gripper (300 deg)

/* =================================================
   PWM calibration (300 deg servos)
   ================================================= */
#define L1_MIN 1050
#define L1_MAX 2300
#define SL_MIN 600
#define SL_MAX 1500
#define GR_MIN 600
#define GR_MAX 1500

/* =================================================
   Current position state
   ================================================= */
int   l1_pos      = 180;
int   slider_pos  = 180;
int   gripper_pos = 180;

/* =================================================
   Link2 (360 deg continuous servo) CONFIG
   ================================================= */
int   L2_STOP     = 1500;   // tune
int   L2_PWM_CW   = 1500 - 200;   // tune
int   L2_PWM_CCW  = 1500 + 200;   // tune

float L2_deg_per_sec_CW  = 360/8.769;  // <-- ใส่ค่าที่วัดมา
float L2_deg_per_sec_CCW = 360/7.54;  // <-- ใส่ค่าที่วัดมา

float l2_deg = 0.0;   // ตำแหน่งสะสมของ link2 (deg)

/* =================================================
   Speed base (smooth motion for 300 deg servos)
   ================================================= */
int BASE_DELAY  = 40;
int EXTRA_DELAY = 20;

/* =================================================
   Angle -> PWM (300 deg servos)
   ================================================= */
int angleToPWM(int deg, int pwm_min, int pwm_max) {
  deg = constrain(deg, 0, 180);
  return map(deg, 0, 180, pwm_min, pwm_max);
}

/* =================================================
   Smooth motion (300 deg servos only)
   ================================================= */
void moveServoSmooth(Servo &sv,
                     int &currentPos,
                     int targetPos,
                     int pwm_min,
                     int pwm_max)
{
  targetPos = constrain(targetPos, 0, 300);   // สำหรับ servo 300°

  int distance = abs(targetPos - currentPos);
  if (distance == 0) return;

  int startPos = currentPos;
  int direction = (targetPos > currentPos) ? 1 : -1;

  for (int i = 0; i <= distance; i++)
  {
    float phase = (float)i / distance;

    // ease-in-out (smoothstep)
    float ease = phase * phase * (3 - 2 * phase);

    // easing อยู่ที่ "ตำแหน่ง"
    float easedStep = ease * distance;
    int pos = startPos + direction * easedStep;

    int pwm = angleToPWM(pos, pwm_min, pwm_max);
    sv.writeMicroseconds(pwm);

    delay(BASE_DELAY);   // คงที่
  }

  // 🔥 สำคัญมาก
  currentPos = targetPos;
}


/* =================================================
   Link2 low-level control
   ================================================= */
void l2Stop() {
  servo2.writeMicroseconds(L2_STOP);
}

void l2CW() {
  servo2.writeMicroseconds(L2_PWM_CW);
}

void l2CCW() {
  servo2.writeMicroseconds(L2_PWM_CCW);
}

/* =================================================
   Link2 motion (time-based)
   ================================================= */
void moveLink2To(float target_deg) {
  float delta = target_deg - l2_deg;
  if (abs(delta) < 0.3) return;

  bool cw = (delta < 0);
  float speed = cw ? L2_deg_per_sec_CW : L2_deg_per_sec_CCW;

  unsigned long move_time_ms =
    (unsigned long)(abs(delta) / speed * 1000.0);

  if (cw) l2CW();
  else    l2CCW();

  delay(move_time_ms);
  l2Stop();

  l2_deg = target_deg;
}

/* =================================================
   SETUP
   ================================================= */
void setup() {
  Serial.begin(9600);

  servo1.attach(3);
  servo2.attach(5);
  slider.attach(6);
  gripper.attach(9);

  // Home
  moveServoSmooth(servo1, l1_pos, 300, L1_MIN, L1_MAX);
  l2Stop();
  l2_deg = 0.0;
  moveServoSmooth(slider, slider_pos, 300, SL_MIN, SL_MAX);
  moveServoSmooth(gripper, gripper_pos, 300, GR_MIN, GR_MAX);

  Serial.println("servo.ino READY");
}

/* =================================================
   LOOP
   ================================================= */
void loop() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();

  if (cmd.startsWith("L1:")) {
    int deg = cmd.substring(3).toInt();
    moveServoSmooth(servo1, l1_pos, deg, L1_MIN, L1_MAX);
  }
  else if (cmd.startsWith("L2:")) {
    float deg = cmd.substring(3).toFloat();
    moveLink2To(deg);
  }
  else if (cmd.startsWith("SL:")) {
    int deg = cmd.substring(3).toInt();
    moveServoSmooth(slider, slider_pos, deg, SL_MIN, SL_MAX);
  }
  else if (cmd.startsWith("GR:")) {
    int deg = cmd.substring(3).toInt();
    moveServoSmooth(gripper, gripper_pos, deg, GR_MIN, GR_MAX);
  }
}
