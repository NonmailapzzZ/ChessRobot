#include <Servo.h>

Servo servo2;

// =============================
// TUNE ตรงนี้
// =============================
#define PWM_STOP 1500      // neutral (ต้อง tune จริง)
#define PWM_RANGE 200      // ช่วงความเร็ว (±)

// =============================
int pwm_cmd = PWM_STOP;
bool running = false;
int direction = 1;   // 1 = CW, -1 = CCW

void setup()
{
  Serial.begin(9600);
  servo2.attach(5);

  servo2.writeMicroseconds(PWM_STOP); // STOP แน่นอน
  Serial.println("READY");
}

void loop()
{
  if (Serial.available())
  {
    char c = Serial.read();

    if (c == 'c') direction = 1;
    if (c == 'a') direction = -1;

    if (c == 's') {
      running = true;
      pwm_cmd = PWM_STOP + direction * PWM_RANGE;
      servo2.writeMicroseconds(pwm_cmd);   // หมุนทันที
      Serial.println("RUN");
    }

    if (c == 'e') {
      running = false;
      servo2.writeMicroseconds(PWM_STOP);  // <<< STOP จริง
      Serial.println("STOP");
    }
  }
}
