

// this file is meant to be uploaded on to the arduino Uno that is connected to the robotic arm

#include <Braccio.h>
#include <Servo.h>

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_ver;
Servo wrist_rot;
Servo gripper;
String joint_angle_string;

void setup() {
  Braccio.begin();
  Serial.begin(9600);
  
}

void loop() {
  // contantly check serial port for inputs
  if(Serial.available() > 0){
    joint_angle_string = Serial.readStringUntil('\n');

    int values[6];
    int valueIndex = 0;
    char* ptr = strtok(const_cast<char*>(joint_angle_string.c_str()), ", ");
    // while loop for removing the commas in our string
    while (ptr != NULL && valueIndex < 6) {
      values[valueIndex] = atoi(ptr);
      ptr = strtok(NULL, ", ");
      valueIndex++;
    }
    // store each value in its own variable
    int base_angle, shoulder_angle, elbow_angle, vertical_wrist_angle, rotatory_wrist_angle,  gripper_angle;
    if (valueIndex >= 1) base_angle = values[0];
    if (valueIndex >= 2) shoulder_angle = values[1];
    if (valueIndex >= 3) elbow_angle = values[2];
    if (valueIndex >= 4) vertical_wrist_angle = values[3];
    if (valueIndex >= 5) rotatory_wrist_angle = values[4];
    if (valueIndex >= 6) gripper_angle = values[5];
    // move arm using the values we got
    Braccio.ServoMovement(20, base_angle, shoulder_angle, elbow_angle, vertical_wrist_angle, rotatory_wrist_angle,  gripper_angle );
    // adding 2 second delay and then we chnage the position of the robotic arm to an upright position to make sure that it doesnt sudenly fall down
    //delay(2000);
    //Braccio.ServoMovement(20, 90,90,90,90,90,73);
    


  }
}
