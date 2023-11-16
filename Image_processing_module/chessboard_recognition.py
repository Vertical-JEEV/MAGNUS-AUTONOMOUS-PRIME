import cv2
import numpy as np
import os
from time import sleep


class ChessboardRecognition:
    def __init__(self, CHESSBOARD_DIMENSIONS):
       
        self.og_img = None # original image
        self.processing_img = None # image used for processing
        self.warped_img = None # image after perspective transform for top down view

        self.transformation_matrix = None # matrix to apply perspective transform for top down view
        self.inverse_tranformation_matrix = None # matrix to get from top down view to original view

        self.pixel_coordinates = None # the coordinate of each corner in each square in the original image
        self.WIDTH =  int(CHESSBOARD_DIMENSIONS.strip().split("x")[0]) # width of chessboard in cm
        self.HEIGHT = int(CHESSBOARD_DIMENSIONS.strip().split("x")[1]) # height of chessboard in cm

        self.camera_matrix = None # camera matrix will be used for pixel to 3d conversion
        self.distortion_coeffcients = None # will be used to undistort the cameras
        self.translation_vector = None # gives a relative postiion of the camera to the 3d coordinate system
        self.rotation_vector = None # gives a rotation relative to 3d coordinates system
       
        self.chessboard_3d_coordinates = None # stores the 3d coordinates once its been converted 

    

    def get_3d_chessboard_corners_for_calibration(self):
        # store the 3d corners for the calibration
        chessboard_3d_corners_for_calibration = []
        # 8x8 chessboard so 81 corners
        for i in range(9):
            for j in range(9):
                chessboard_3d_corners_for_calibration.append([i*self.WIDTH, j*self.HEIGHT, 0])
        # convert the list to a numpy array
        chessboard_3d_corners_for_calibration = np.array(chessboard_3d_corners_for_calibration)
        return chessboard_3d_corners_for_calibration
    


    def get_imgs_and_their_corners_for_calibration(self):
        # get the frames fromt the camera
        cap = cv2.VideoCapture(0)
        pixel_coordinates_for_calibration = []
        chessboard_3d_corners_for_calibration = []
        
        # get the frames from the camera
        while(True):
            
            ret, frame = cap.read()

            
            test_chessboard_recognition(self, frame)
            cv2.imshow('detected corners', self.og_img) # shows what corners are detected
            cv2.imshow('processing image', self.processing_img) # the user can see if the chessboard is the largest contour
           
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # if the user presses s then run corner detecion on the image
            elif cv2.waitKey(1) & 0xFF == ord('s'):
                
                # get the pixel coordinates of each corner of each square on the chessboard and add to our list
                if self.pixel_coordinates != None:

                    
                    pixel_coordinates_for_calibration.append(self.pixel_coordinates)
                    chessboard_3d_corners_for_calibration.append(self.get_3d_chessboard_corners_for_calibration())
                    print("saved img coords for calibration")

        
   
        # convert the list to a numpy array
        pixel_coordinates_for_calibration = np.array(pixel_coordinates_for_calibration)
        chessboard_3d_corners_for_calibration = np.array(chessboard_3d_corners_for_calibration)
       
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        return pixel_coordinates_for_calibration,  chessboard_3d_corners_for_calibration
                

        

    def calibrate_camera(self):
        chessboard_3d_corners_for_calibration = self.get_3d_chessboard_corners_for_calibration()
        chessboard_pixel_corners_for_calibration = self.find_chessboard_pixel_corners_for_calibration()

        # get the camera matrix and distortion coefficients
        ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(chessboard_3d_corners_for_calibration, chessboard_pixel_corners_for_calibration, (self.og_img.shape[1], self.og_img.shape[0]), None, None)
        return camera_matrix, distortion_coefficients, rvecs, tvecs



    def _resize_img(self):
        # constants for resizing image
        SCALE_PERCENT = 50
        # Resize image
        new_width = int(self.og_img.shape[1] * SCALE_PERCENT / 100)
        new_height = int(self.og_img.shape[0] * SCALE_PERCENT / 100)
        new_dimension = (new_width, new_height)
        self.processing_img = cv2.resize(self.og_img, new_dimension, interpolation = cv2.INTER_AREA)
    
    
    def _pre_process_img(self):
        # Convert to gray scale image
        THRESHOLD = 100
        self.processing_img = cv2.cvtColor(self.processing_img, cv2.COLOR_BGR2GRAY)
        # apply gaussian blur to remove noise
        self.processing_img = cv2.GaussianBlur(self.processing_img, (7, 7), 0)
        # Apply threshold to get binary image, helps with localising chessboard
        ret, self.processing_img = cv2.threshold(self.processing_img, THRESHOLD, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # uses otsu thresholding for optimal threshold value


    def locate_chessboard_and_get_4_external_corners(self):
        # find the all the contours of the image,ie the white parts of the binary image
        contours, _ = cv2.findContours(self.processing_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the largest  enclosed contour area which should be the chessboard
        largest_contour = max(contours, key = cv2.contourArea)
        # Find the corners of the largest contour, ie the 4 outer corners of the chessboard
        corners = cv2.approxPolyDP(largest_contour, 0.05 * cv2.arcLength(largest_contour, True), True)
        # convert corners to numpy array
        corners = corners.reshape(-1, 2)

        # find the specific corners based on x + y, x - y
        top_left = corners[np.argmin(corners.sum(axis=1))]
        bottom_right = corners[np.argmax(corners.sum(axis=1))]
        bottom_left = corners[np.argmin(np.diff(corners, axis=1))]
        top_right = corners[np.argmax(np.diff(corners, axis=1))]
        # make sure the corners are in the right order
        corners = np.array([top_right, bottom_right, bottom_left, top_left], dtype='float32') # corners will be used for perspective transform 
        return corners


    def crop_and_get_top_down_view_of_chessboard(self, corners):
        # the required width and height of the warped image
        WIDTH = 400
        HEIGHT = 400
        # # convert corners to numpy array
        # corners = np.array(corners, dtype='float32')


        # this is the destination points, where we want to transform the image to
        destination_points = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], dtype ='float32')

        # get the transformation matrix for a top down view of the original image
     

        self.transformation_matrix = cv2.getPerspectiveTransform(corners, destination_points)
        self.inverse_tranformation_matrix = cv2.getPerspectiveTransform(destination_points, corners)
        #apply the transformation matrix to the image
        self.warped_img = cv2.warpPerspective(self.og_img, self.transformation_matrix, (WIDTH, HEIGHT))
        #rotate image 180 degrees
        # self.warped_img = cv2.rotate(self.warped_img, cv2.ROTATE_180)
        # # mirror the image
        # self.warped_img = cv2.flip(self.warped_img, 1)

        # show the top down view of the chessboard
        cv2.imshow('Warped image', self.warped_img)
        


    def get_pixel_coordinates_using_shi_tomasi_on_warped_img(self):

        img_copy = self.warped_img.copy()
        # convert warped image to grayscale
        self.warped_img = cv2.cvtColor(self.warped_img, cv2.COLOR_BGR2GRAY)
        # apply a strong gaussian blur to prevent numbers and letters of the chessboard from being detected as corners
        self.warped_img = cv2.GaussianBlur(self.warped_img, (9,9), 0)
        # using the shi- tomasi algorithm to find each corner of each square on the chessboard
        try:

            corners = cv2.goodFeaturesToTrack(self.warped_img, 85, 0.02, 40)
            # improve accuracy of corners, using subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)
            corners = cv2.cornerSubPix(self.warped_img, corners, (11,11), (-1,-1), criteria)
            # this guarantees that we have 81 corners, if not then we cant use the image
            if len(corners) != 81:
                return False
            
            
        



            # draw the corners on the warped image
            # for corner in corners:
                # cv2.circle(img_copy, (int(corner[0][0]), int(corner[0][1])), 5, (0,0,255), -1)

            # convert warped plane corners to original plane corners using the inverse transformation matrix we got from the reverse perspective transform
            self.pixel_coordinates = np.dot(self.inverse_tranformation_matrix, np.array([corners[:,0,0], corners[:,0,1], np.ones(corners.shape[0])]))
            

            self.pixel_coordinates = self.pixel_coordinates[:2,:] / self.pixel_coordinates[2,:]

            self.pixel_coordinates = np.transpose(self.pixel_coordinates)
            # #draw the corners on the original image
            for corner in self.pixel_coordinates:
                cv2.circle(self.og_img, (int(corner[0]), int(corner[1])), 5, (0,0,255), -1)
                # add text to the image
                    # cv2.putText(self.og_img, f"({int(corner[0])}, {int(corner[1])})", (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            #cv2.imshow("detected corners",self.og_img )
        except cv2.error:
            print("No corners found")

    



    # def pixel_coordinates_using_chessboard_corners(self):
    #     img_copy = self.warped_img.copy()
    #     # convert warped image to grayscale
    #     self.warped_img = cv2.cvtColor(self.warped_img, cv2.COLOR_BGR2GRAY)
    #     # apply a very strong gaussian blur to prevent numbers and letters of the chessboard from being detected as corners
    #     self.warped_img = cv2.GaussianBlur(self.warped_img, (9,9), 0)

    #     ret, corners = cv2.findChessboardCorners(self.warped_img, (6,6), None)
    #     if ret:
    #         # improve accuracy of corners, using subpixel accuracy
    #         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.0001)
    #         corners = cv2.cornerSubPix(self.warped_img, corners, (11,11), (-1,-1), criteria)

    #         # draw the corners on the warped image
    #         for corner in corners:
    #             cv2.circle(img_copy, (int(corner[0][0]), int(corner[0][1])), 5, (0,0,255), -1)



    #         self.pixel_coordinates = np.dot(self.inverse_tranformation_matrix, np.array([corners[:,0,0], corners[:,0,1], np.ones(corners.shape[0])]))
            

    #         self.pixel_coordinates = self.pixel_coordinates[:2,:] / self.pixel_coordinates[2,:]

    #         self.pixel_coordinates = np.transpose(self.pixel_coordinates)
    #         # #draw the corners on the original image
    #         for corner in self.pixel_coordinates:
    #             cv2.circle(self.og_img, (int(corner[0]), int(corner[1])), 5, (0,0,255), -1)
        

    def convert_pixel_coordinates_to_3d(self):
        inverse_camera_matrix = np.linalg.inv(self.camera_matrix)
        
        self.chessboard_3d_coordinates = np.dot(inverse_camera_matrix, self.pixel_coordinates)


        
        

    def class_reset(self):
        self.og_img = None
        self.processing_img = None
        self.warped_img = None
        self.transformation_matrix = None
        self.inverse_tranformation_matrix = None
        self.pixel_coordinates = None
        
      
        #print("This is not working")
            


    
def test_camera_calibration():
    chessboard_recogniser = ChessboardRecognition("32x32")
    pixel_corners_for_calibration = chessboard_recogniser.get_imgs_and_their_corners_for_calibration()
    
    



def test_chessboard_recognition(chessboard_recogniser, frame):

    # use our class as process the images

    # set the original image and processing image to the frame
    chessboard_recogniser.og_img = frame
    chessboard_recogniser.processing_img = frame
    # pre process the image for getting the chessboard
    chessboard_recogniser._pre_process_img()
    # locate the chessboard by finding largest contour
    corners = chessboard_recogniser.locate_chessboard_and_get_4_external_corners()
    # crop and transform the image to a top down view
    chessboard_recogniser.crop_and_get_top_down_view_of_chessboard(corners)
    # get the pixel coordinates of each corner of each square on the chessboard
    if chessboard_recogniser.get_pixel_coordinates_using_shi_tomasi_on_warped_img() == False: #  if you cant 81 corners of the img 
        pass
    


    # show the images
    cv2.imshow('Original image', chessboard_recogniser.og_img)
    cv2.imshow('Processing image', chessboard_recogniser.processing_img)
    cv2.imshow('Warped image', chessboard_recogniser.warped_img)
    #chessboard_recogniser.pixel_coordinates_using_chessboard_corners()
   
      



        
            
   
def access_web_cam():
    chessboard_recogniser = ChessboardRecognition("32x32")
    #test_camera_calibration(chessboard_recogniser)
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        test_chessboard_recognition(chessboard_recogniser, frame)
    
        #chessboard_recogniser.class_reset()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows() 


access_web_cam()

#test_camera_calibration()
