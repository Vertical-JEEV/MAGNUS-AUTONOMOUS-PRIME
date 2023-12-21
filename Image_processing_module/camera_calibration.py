import cv2
import numpy as np
from .corner_detection import ChessboardCornerDetection # our own class for corner detection



class CameraCalibration:

    def __init__(self, CHESSBOARD_DIMENSIONS):
        self.IMAGE_HEIGHT = 1000
        self.IMAGE_WIDTH = 1000
        self.WIDTH = float(CHESSBOARD_DIMENSIONS.strip().split("x")[0])/8
        self.HEIGHT = float(CHESSBOARD_DIMENSIONS.strip().split("x")[1])/8
        self.NUM_X_DIRECTION_CORNERS = self.NUM_Y_DIRECTION_CORNERS = 9
        self.frame = None


    def __generate_3d_corners_for_calibration(self):
        corner_coordinates = np.mgrid[0:self.NUM_X_DIRECTION_CORNERS, 0: self.NUM_Y_DIRECTION_CORNERS].T.reshape(-1,2)
        corner_coordinates = corner_coordinates.astype(float)
        corner_coordinates[:, 0] *= self.WIDTH
        corner_coordinates[:, 1] *= self.HEIGHT

        chessboard_3d_corners = np.hstack((corner_coordinates, np.zeros((self.NUM_X_DIRECTION_CORNERS * self.NUM_Y_DIRECTION_CORNERS, 1))))
        return chessboard_3d_corners
    

    def __calibrate_camera(self, pixel_coordinates_for_calibration, chessboard_3d_corners_for_calibration):
        # reshape pixel_coordinates_for_calibration to 10,49,1,2
        pixel_coordinates_for_calibration = np.reshape(pixel_coordinates_for_calibration,((len(pixel_coordinates_for_calibration)),81,1,2))
        # reshape chessboard_3d_corners_for_calibration to 10,49,1,3
        chessboard_3d_corners_for_calibration = np.reshape(chessboard_3d_corners_for_calibration,(len(chessboard_3d_corners_for_calibration),81,1,3))
        # calibrate the camera using object and image points
        ret, intrinsic_matrix, distortion_coefficients, rotation_vector, translation_vector = cv2.calibrateCamera(chessboard_3d_corners_for_calibration, pixel_coordinates_for_calibration, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), None, None)
        print("camera has been calibrated")
       
        # calulate the extrinsic parametres with solve pnp
        ret, rotation_vector, translation_vector = cv2.solvePnP(chessboard_3d_corners_for_calibration[0], pixel_coordinates_for_calibration[0], intrinsic_matrix, distortion_coefficients)

        # convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        # get the extrinsic matrix
        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector))

        print("calculated the extrinsic matrix")

        # calculate the mean error for the intrinsic and extrinsic matrix
        # mean_error = 0
        # for i in range(len(chessboard_3d_corners_for_calibration)):
        #     imgpoints2, _ = cv2.projectPoints(chessboard_3d_corners_for_calibration[i], rotation_vector, translation_vector, intrinsic_matrix, distortion_coefficients)
        #     error = cv2.norm(pixel_coordinates_for_calibration[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        #     mean_error += error

        # mean_error /= len(chessboard_3d_corners_for_calibration)
        # print("mean error: ", mean_error)
        return intrinsic_matrix, extrinsic_matrix, distortion_coefficients    
        
    

    def start_calibration(self):
        corner_detection = ChessboardCornerDetection()
        NUM_IMAGES_USED = 0
        pixel_coordinates_for_calibration = []
        chessboard_3d_corners_for_calibration = []
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.IMAGE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.IMAGE_HEIGHT)

        #get the frames from the camera until we have 20 images for calibration
        while NUM_IMAGES_USED < 10:
            # Capture frame-by-frame
            ret, frame = cap.read()
            corner_detection.recognise_corners(frame)
            self.frame = corner_detection.drawing_original_img
        
            # if the user presses s then we save the image and object points for calibration. we will change this we start combining this with the robotic arm to move.
            if cv2.waitKey(1) & 0xFF == ord('s'):
                
                if corner_detection.pixel_coordinates is not None:
                    pixel_coordinates_for_calibration.append(corner_detection.pixel_coordinates)
                    chessboard_3d_corners_for_calibration.append(self.__generate_3d_corners_for_calibration())
                    NUM_IMAGES_USED += 1
                    print(f"Number of images used for calibration is {NUM_IMAGES_USED}")
                    print("saved the img for calibration\n")
     
        # when we have saved the points for 20 images we will then close the cap and destroy all windows      
        cap.release()
        cv2.destroyAllWindows()
        print(f"Number of images used for calibration is {NUM_IMAGES_USED}")

        # make sure the image and object points are a numpy array
        pixel_coordinates_for_calibration = np.array(pixel_coordinates_for_calibration,dtype='float32')
        chessboard_3d_corners_for_calibration = np.array(chessboard_3d_corners_for_calibration, dtype = 'float32')

        # calibrate the camera
        return self.__calibrate_camera(pixel_coordinates_for_calibration, chessboard_3d_corners_for_calibration)



    
        

        

