import serial
import serial.tools.list_ports

import time


class ArduinoComunicator:

    @staticmethod
    def get_com_port():
        TARGET_HARDWARE = "Arduino Uno"
        # we are looping through the com ports and checking if its an arduino uno
        for port in serial.tools.list_ports.comports():
            if TARGET_HARDWARE in port.description:
                return port.device
            
    
    def pass_joint_angles_to_arm(self, joint_angle_string):
        # we assume the baudrate to be 9600 for the arduino so we have to match it here
       
        # need to add delay or else arduino wont register it 
      
        BAUDRATE = 9600
        COM_PORT = self.get_com_port()
        arduino_connection = serial.Serial(COM_PORT, BAUDRATE)
        # need to add delay or else arduino or else arduino wont register it
        time.sleep(12)
        
        joint_angle_string = joint_angle_string.strip()
        joint_angle_string = joint_angle_string.encode()
        arduino_connection.write(joint_angle_string)
        arduino_connection.close()




        





        
    

