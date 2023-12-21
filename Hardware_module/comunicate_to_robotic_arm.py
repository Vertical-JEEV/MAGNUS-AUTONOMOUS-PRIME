import serial
import serial.tools.list_ports
import math 
import time


class ArduinoComunicator:
    SHOULDER_LINK_LENGTH = 10 # length of the shoulder link
    ELBOW_LINK_LENGTH = 10 # length of the elbow link
    WRIST_LINK_LENGTH = 10 # length of the wrist link
    BAUDRATE = 9600 # baudrate of the arduino


    def __init__(self):
        self.COM_PORT = self.get_com_port()
        

    @staticmethod
    def get_com_port():
        TARGET_HARDWARE = "Arduino Uno"
        # we are looping through the com ports and checking if its an arduino uno
        for port in serial.tools.list_ports.comports():
            if TARGET_HARDWARE in port.description:
                return port.device
            
            
    def inverse_kinematics(self, chesspiece_3d_coordinate):
        # code for inverse kinematics
        x,y,z = chesspiece_3d_coordinate

        base_angle = math.atan2(y, x) # base_angle is the angle between the x-axis and the projection of the vector onto the x-y plane
        base_angle = math.degrees(base_angle) 
        base_angle = max(0, min(180, base_angle))  # Ensure base_angle is within [0, 180]

        r1 = math.sqrt(x**2 + y**2) # r1 is the distance between the base and the projection of the vector onto the x-y plane
        r2 = z - self.SHOULDER_LINK_LENGTH # r2 is the distance between the shoulder and the projection of the vector onto the x-y plane
        r3 = math.sqrt(r1**2 + r2**2) # r3 is the distance between the base and the wrist

        cosine_shoulder_angle = (self.ELBOW_LINK_LENGTH**2 - self.WRIST_LINK_LENGTH**2 + r3**2)/(2*self.ELBOW_LINK_LENGTH*r3) # cosine of the shoulder angle
        shoulder_angle = math.acos(cosine_shoulder_angle) # shoulder_angle is the angle between the shoulder and the wrist
        shoulder_angle = math.degrees(shoulder_angle) 
        shoulder_angle = max(15, min(165, shoulder_angle))  # Ensure shoulder_angle is within [15, 165]

        cosine_elbow_angle = (self.ELBOW_LINK_LENGTH**2 + self.WRIST_LINK_LENGTH**2 - r3**2)/(2*self.ELBOW_LINK_LENGTH*self.WRIST_LINK_LENGTH) # cosine of the elbow angle
        elbow_angle = math.acos(cosine_elbow_angle) # elbow_angle is the angle between the elbow and the wrist
        elbow_angle = math.degrees(elbow_angle)
        elbow_angle = max(0, min(180, elbow_angle))  # Ensure elbow_angle is within [0, 180]

        wrist_roll_angle = math.atan2(r2, r1) # wrist_roll_angle is the angle between the x-axis and the projection of the vector onto the x-y plane
        wrist_roll_angle = math.degrees(wrist_roll_angle)
        wrist_roll_angle = max(0, min(180, wrist_roll_angle))  # Ensure wrist_roll_angle is within [0, 180]

        wrist_pitch_angle = math.pi - shoulder_angle - elbow_angle # wrist_pitch_angle is the angle between the shoulder and the wrist
        wrist_pitch_angle = math.degrees(wrist_pitch_angle)
        wrist_pitch_angle = max(0, min(180, wrist_pitch_angle))  # Ensure wrist_pitch_angle is within [0, 180]

        wrist_yaw_angle = 0 # wrist_yaw_angle is the angle between the x-axis and the projection of the vector onto the x-y plane
        wrist_yaw_angle = math.degrees(wrist_yaw_angle) 
        wrist_yaw_angle = max(10, min(73, wrist_yaw_angle))  # Ensure wrist_yaw_angle is within [10, 73]

        joint_angle_string = f"{base_angle},{shoulder_angle},{elbow_angle},{wrist_roll_angle},{wrist_pitch_angle},{wrist_yaw_angle}"
        return joint_angle_string


    def move_arm(self, chesspiece_3d_coordinate):
        joint_angle_string = self.inverse_kinematics(chesspiece_3d_coordinate)
        arduino_connection = serial.Serial(self.COM_PORT, self.BAUDRATE)
        # need to add delay or else arduino or else arduino wont register it
        time.sleep(12)
        joint_angle_string = joint_angle_string.strip() # remove any whitespace
        joint_angle_string = joint_angle_string.encode() #  encode the string to bytes for arduino to understand
        arduino_connection.write(joint_angle_string)
        arduino_connection.close()


        


def testing():
    
    arduino = ArduinoComunicator()
    print(arduino.COM_PORT)
    chesspiece_3d_coordinate = (10, 10, 10)
    arduino.move_arm(chesspiece_3d_coordinate)


#testing()


        
    

