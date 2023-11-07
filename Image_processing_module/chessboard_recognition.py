import cv2
import numpy as np
import math

from time import sleep

class ChessboardRecognition:
    def __init__(self):
        self.og_img = None
        self.processing_img = None
        self.warped_img = None
        self.tranformation_matrix = None
        self.pixel_coordinates = None
        
    


    def calculate_intrinisc_camera_matrix(self):
        pass





    def _resize_img(self):
        SCALE_PERCENT = 50
        # Resize image
        width = int(self.og_img.shape[1] * SCALE_PERCENT / 100)
        height = int(self.og_img.shape[0] * SCALE_PERCENT / 100)
        dim = (width, height)
        self.processing_img = cv2.resize(self.og_img, dim, interpolation = cv2.INTER_AREA)
    
    
    def _pre_process_img(self):
        THRESHOLD = 100
        # Convert to gray scale
        self.processing_img = cv2.cvtColor(self.processing_img, cv2.COLOR_BGR2GRAY)

        # apply gaussian blur
        self.processing_img = cv2.GaussianBlur(self.processing_img, (7, 7), 0)

        # Apply threshold
        ret, self.processing_img = cv2.threshold(self.processing_img, THRESHOLD, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    def locate_chessboard(self):
        contours, _ = cv2.findContours(self.processing_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest  enclosed contour area
        largest_contour = max(contours, key = cv2.contourArea)

        # cv2.drawContours(self.og_img, [largest_contour], -1, (0, 255, 0), 2)  # Draw the largest contour in green

        # Find the corners of the largest contour
        corners = cv2.approxPolyDP(largest_contour, 0.05 * cv2.arcLength(largest_contour, True), True)

        # circle each corner with a new colour
        # for i in range(len(corners)):
        #     cv2.circle(self.og_img, tuple(corners[i][0]), 7, (0, 0, 255), -1)
        #     cv2.putText(self.og_img, str(i), tuple(corners[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        return corners


    def crop_and_perspective_transform(self, corners):
        # write corners into a numpy array

        # crop image to where corners are
        
        WIDTH = 500
        HEIGHT = 500

        # convert corners to numpy array
        corners = np.array(corners, dtype='float32')
        # this is the destination points, where we want to transform the image to
        destination_points = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]], dtype ='float32')
        # get the transformation matrix
        try:
            transformation_matrix = cv2.getPerspectiveTransform(corners, destination_points)
            self.inverse_tranformation_matrix = np.linalg.inv(transformation_matrix)
            #apply the transformation matrix to the image
            self.warped_img = cv2.warpPerspective(self.og_img, transformation_matrix, (WIDTH, HEIGHT))
            # rotate image 90 degrees
            # self.self.warped_img = cv2.rotatePerspective(self.warped_img, cv2.ROTATE_90_CLOCKWISE)
            # mirror the image
            self.warped_img = cv2.flip(self.warped_img, 1)
            #cv2.imshow('Warped image', self.warped_img)
            return True
        except:
            return False

    

    def get_pixel_coordinates(self):
        # convert warped image to grayscale
        warped_img_copy = self.warped_img
        self.warped_img = cv2.cvtColor(self.warped_img, cv2.COLOR_BGR2GRAY)
        self.warped_img = cv2.GaussianBlur(self.warped_img, (17,17), 0)

        # using the shi- tomasi algorithm to find the corners
        corners = cv2.goodFeaturesToTrack(self.warped_img, 81, 0.5, 20)
        corners = np.int0(corners)
        #print(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(warped_img_copy, (x, y), 5, (0, 0, 255), 2)

        #showing the image output of corners
        cv2.imshow('gray warped img', self.warped_img)
        cv2.imshow('detected corners', warped_img_copy)


        # convert warped plane corners to original plane corners
        actual_corners = np.dot(self.inverse_tranformation_matrix, np.array([corners[:,0,0], corners[:,0,1], np.ones(corners.shape[0])]))

        # draw corners on original image
        for i in range(actual_corners.shape[1]):
            cv2.circle(self.og_img, (int(actual_corners[0,i]/actual_corners[2,i]), int(actual_corners[1,i]/actual_corners[2,i])), 5, (0, 0, 255), 2)




        
    




def test_chessboard_recognition(chessboard_recogniser, frame):

    # use our class as process the images
    chessboard_recogniser.og_img = frame
    chessboard_recogniser.processing_img = frame
    chessboard_recogniser._pre_process_img()
    corners = chessboard_recogniser.locate_chessboard()
    if chessboard_recogniser.crop_and_perspective_transform(corners) == True:
        chessboard_recogniser.get_pixel_coordinates()
        pass
    
    



def access_web_cam():
    chessboard_recogniser = ChessboardRecognition()
    cap = cv2.VideoCapture(1)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
      

        test_chessboard_recognition(chessboard_recogniser, frame)

        cv2.imshow('Original image', chessboard_recogniser.og_img)
        cv2.imshow('Processing image', chessboard_recogniser.processing_img)

        #cv2.imshow('Warped image', chessboard_recogniser.warped_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows() 

access_web_cam()