from Hardware_Subsystem.comunicate_to_robotic_arm import ArduinoComunicator
from Image_processing_module.chessboard_recognition import ChessboardRecognition
import cv2


class StartGame:
    def __init__(self, chessboard_dimensions, elo, colour):
        self.chessboard_dimensions = chessboard_dimensions
        self.ELO = elo
        self.chessboard_recogniser = ChessboardRecognition()
        self.arduino_comunicator = ArduinoComunicator()
        self.chesspiece_recogniser = ChesspieceRecognition()
        self.PLAYER_COLOR = colour
        self.ROBOT_COLOR = "white" if self.PLAYER_COLOR == "black" else "black"
        self.turn = True if self.PLAYER_COLOR == "white" else False

       
        
    





        














def test_chessboard_recognition(chessboard_recogniser, frame):
    chessboard_recogniser.og_img = frame
    #chessboard_recogniser._pre_process_img()
    #lines = chessboard_recogniser.canny_and_hough_detection()
    #chessboard_recogniser.draw_lines(lines)

def access_web_cam():
    chessboard_recogniser = ChessboardRecognition()
    cap = cv2.VideoCapture(1)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        
        test_chessboard_recognition(chessboard_recogniser, frame)

        cv2.imshow('frame', chessboard_recogniser.og_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main():

    arduino_comunicator = ArduinoComunicator()
    
    cam_joint_angle_string = "130,45,90,90,90,10"
    arduino_comunicator.pass_joint_angles_to_arm(cam_joint_angle_string)
    #access_web_cam()

    
    # joint_angle_string = "120,45,45,130,60,73"
    # arduino_comunicator.pass_joint_angles_to_arm(joint_angle_string)

    # joint_angle_string = "60,90,90,80,150,10"
    # arduino_comunicator.pass_joint_angles_to_arm(joint_angle_string)

    


if __name__ == "__main__":
    main()     