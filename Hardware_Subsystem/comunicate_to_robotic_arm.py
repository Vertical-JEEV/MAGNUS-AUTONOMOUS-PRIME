import serial
import serial.tools.list_ports
import math 
import time


class ArduinoComunicator:
    SHOULDER_LINK_LENGTH = 10
    ELBOW_LINK_LENGTH = 10
    WRIST_LINK_LENGTH = 10


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
        # 6 DOF arm
        # x,y,z are the coordinates of the end effector
        # SHOUL,self.ELBOW_LINK_LENGTH,self.WRIST_LINK_LENGTH are the lengths of the links in cm
        x,y,z = chesspiece_3d_coordinate
        
        # we are using the inverse kinematics equations to calculate the joint angles
        # base_angle
        base_angle = math.atan2(y, x)
        # theta2
        r1 = math.sqrt(x**2 + y**2)
        r2 = z - self.SHOULDER_LINK_LENGTH
        r3 = math.sqrt(r1**2 + r2**2)

        cosine_shoulder_angle = (self.ELBOW_LINK_LENGTH**2 - self.WRIST_LINK_LENGTH**2 + r3**2)/(2*self.ELBOW_LINK_LENGTH*r3)
        shoulder_angle = math.acos(cosine_shoulder_angle)


        # theta3
        cosine_elbow_angle = (self.ELBOW_LINK_LENGTH**2 + self.WRIST_LINK_LENGTH**2 - r3**2)/(2*self.ELBOW_LINK_LENGTH*self.WRIST_LINK_LENGTH)
        elbow_angle = math.acos(cosine_elbow_angle)

        # theta4
        wrist_roll_angle = math.atan2(r2, r1)

        # theta5
        wrist_pitch_angle = math.pi - shoulder_angle - elbow_angle

        # theta6
        wrist_yaw_angle = 0
        # converting to degrees

        base_angle = math.degrees(base_angle)
        theta2 = math.degrees(theta2)
        theta3 = math.degrees(theta3)
        theta4 = math.degrees(theta4)
        theta5 = math.degrees(theta5)
        theta6 = math.degrees(theta6)
        # converting to string
        joint_angle_string = str(base_angle) + "," + str(theta2) + "," + str(theta3) + "," + str(theta4) + "," + str(theta5) + "," + str(theta6)
        return joint_angle_string



    def pass_joint_angles_to_arm(self, joint_angle_string):
        # we assume the baudrate to be 9600 for the arduino so we have to match it here
       
        # need to add delay or else arduino wont register it 
      
        BAUDRATE = 9600
        
        arduino_connection = serial.Serial(self.COM_PORT, BAUDRATE)
        # need to add delay or else arduino or else arduino wont register it
        time.sleep(12)
        
        joint_angle_string = joint_angle_string.strip()
        joint_angle_string = joint_angle_string.encode()
        arduino_connection.write(joint_angle_string)
        arduino_connection.close()





        
    

