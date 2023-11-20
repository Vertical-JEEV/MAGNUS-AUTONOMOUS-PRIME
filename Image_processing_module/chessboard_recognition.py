import cv2
import numpy as np
import os
from time import sleep

class ChessboardRecognition:


    def __init__(self, CHESSBOARD_DIMENSIONS):

        self.og_img = None # original image
        self.processing_img = None # mage used for processing
        self.warped_img = None # image after perspective transform for top down view

        self.transformation_matrix = None # matrix to apply perspective transform for top down view
        
        self.inverse_tranformation_matrix = None # matrix to get from top down view to original view

        
        self.pixel_coordinates = None # the coordinate of each corner in each square in the original image
        self.WIDTH = int(CHESSBOARD_DIMENSIONS.strip().split("x")[0])/8 # width of chessboard in cm
        self.HEIGHT = int(CHESSBOARD_DIMENSIONS.strip().split("x")[1])/8 # height of chessboard in cm

       
        self.intrinsic_camera_matrix = None # intrinsic matrix will be used for pixel to 3d conversion
        self.extrinsic_camera_matrix = None # will be used for pixel to 3d conversion
        self.distortion_coeffcients = None # will be used to undistort the cameras

       
        self.chessboard_3d_coordinates = None # stores the 3d coordinates once its been converted from pixel coordinates


    def get_3d_chessboard_corners_for_calibration(self):
        # store the 3d corners for the calibration
        chessboard_3d_corners_for_calibration = []

        # 8x8 chessboard so 81 corners
        for i in range(9):
            for j in range(9):
                chessboard_3d_corners_for_calibration.append([j*self.WIDTH, i*self.HEIGHT, 0])

        # convert the list to a numpy array
        return np.array(chessboard_3d_corners_for_calibration, dtype='float32')
    


    def get_pixel_corners_for_calibration(self):
        # use findchessboardcorners to get internal 7,7 corners
          # convert warped image to grayscale
        self.warped_img = cv2.cvtColor(self.warped_img, cv2.COLOR_BGR2GRAY)
        # apply a strong gaussian blur to prevent numbers and letters of the chessboard from being detected as corners
        self.warped_img = cv2.GaussianBlur(self.warped_img, (15, 15), 0)

        ret, corners = cv2.findChessboardCorners(self.warped_img,(7,7), None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)
            corners = cv2.cornerSubPix(self.warped_img, corners, (11, 11), (-1, -1), criteria)
            # use corners and convert back to original image
            corners = np.dot(self.inverse_tranformation_matrix, np.array([corners[:, 0, 0], corners[:, 0, 1], np.ones(corners.shape[0])]))
            corners = corners[:2,:] / corners[2, :]
            corners = np.transpose(corners)
            corners = np.reshape(corners, (49, 1, 2)) # gives it back in a format usable by calibratecamera method
           # draw the corners onto the original image using drawchessboardcorners
            cv2.drawChessboardCorners(self.og_img, (7, 7), corners, ret)
            cv2.imshow("found corners for calibration", self.og_img)
            return corners



    def get_imgs_and_their_corners_for_calibration(self):
        desired_width = 800
        desired_height = 800

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
        pixel_coordinates_for_calibration = []
        chessboard_3d_corners_for_calibration = []

        # get the frames from the camera
        while (True):

            ret, frame = cap.read()

            test_chessboard_recognition(self, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # if the user presses s then run corner detecion on the image
            elif cv2.waitKey(1) & 0xFF == ord('s'):

                # get the pixel coordinates of each corner of each square on the chessboard and add to our list
                if self.pixel_coordinates is None:
                    print("no corners found, skipping image")

                else:
                    pixel_coordinates_for_calibration.append(self.pixel_coordinates)
                    chessboard_3d_corners_for_calibration.append(self.get_3d_chessboard_corners_for_calibration())
                    print("saved img coords for calibration")
                continue

        # convert the list to a numpy array

        pixel_coordinates_for_calibration = np.array(pixel_coordinates_for_calibration,dtype='float32')

        print(pixel_coordinates_for_calibration)
        pixel_coordinates_for_calibration = np.reshape(pixel_coordinates_for_calibration, (4, 1, 2))
        chessboard_3d_corners_for_calibration = np.array(chessboard_3d_corners_for_calibration)

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        return pixel_coordinates_for_calibration,  chessboard_3d_corners_for_calibration


    def calibrate_camera(self):
        pixel_coordinates_for_calibration, chessboard_3d_corners_for_calibration = self.get_imgs_and_their_corners_for_calibration()

        # calibrae the camera using the pixel coordinates and 3d coordinates
        ret, self.intrinsic_camera_matrix, self.distortion_coeffcients, rotation_vector, translation_vector = cv2.calibrateCamera(chessboard_3d_corners_for_calibration, pixel_coordinates_for_calibration, (self.og_img.shape[1], self.og_img.shape[0]), None, None)

        # get the error of the calibration
        # error = 0
        # for i in range(len(chessboard_3d_corners_for_calibration)):
        #     img_points, _ = cv2.projectPoints(chessboard_3d_corners_for_calibration[i], rotation_vector[i], translation_vector[i], self.intrinsic_camera_matrix, self.distortion_coeffcients)
        #     error += cv2.norm(pixel_coordinates_for_calibration[i], img_points, cv2.NORM_L2) / len(img_points)

        # print("Total error: ", error / len(chessboard_3d_corners_for_calibration))


    def _resize_img(self):
        # constants for resizing image
        SCALE_PERCENT = 50
        # Resize image
        new_width = int(self.og_img.shape[1] * SCALE_PERCENT / 100)
        new_height = int(self.og_img.shape[0] * SCALE_PERCENT / 100)
        new_dimension = (new_width, new_height)
        self.processing_img = cv2.resize(self.og_img, new_dimension, interpolation=cv2.INTER_AREA)


    def _pre_process_img(self):
        # Convert to gray scale image
        THRESHOLD = 100
        self.processing_img = cv2.cvtColor(self.processing_img, cv2.COLOR_BGR2GRAY)
        # apply gaussian blur to remove noise
        self.processing_img = cv2.GaussianBlur(self.processing_img, (7, 7), 0)
        # Apply threshold to get binary image, helps with localising chessboard
        # uses otsu thresholding for optimal threshold value
        ret, self.processing_img = cv2.threshold(self.processing_img, THRESHOLD, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    def locate_chessboard_and_get_4_external_corners(self):
        # find the all the contours of the image,ie the white parts of the binary image
        contours, _ = cv2.findContours(self.processing_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find the largest  enclosed contour area which should be the chessboard
        largest_contour = max(contours, key=cv2.contourArea)
        # Find the corners of the largest contour, ie the 4 outer corners of the chessboard
        corners = cv2.approxPolyDP(largest_contour, 0.05 * cv2.arcLength(largest_contour, True), True)
        # convert corners to numpy array
        try:
            corners = corners.reshape(-1, 2)
            # find the distance between the bottom 2 corners
            corners = corners[corners[:,0].argsort()] # sort by x
            left = corners[:2, :]
            right = corners[2:, :]
            left = left[left[:,1].argsort()] # sort by y
            right = right[right[:,1].argsort()] # sort by y
            corners = np.array([right[0], right[1], left[1], left[0]], dtype='float32')

            if len(corners) == 4:
                # # draw the corners on the original image
                # for corner in corners:
                #     cv2.circle(self.og_img, (int(corner[0]), int(corner[1])), 5, (255,0,0), -1)

                
                return corners
        except IndexError:
            pass


    def crop_and_get_top_down_view_of_chessboard(self, corners_for_perspective_transform):
        # the required width and height of the warped image
        WIDTH = 500
        HEIGHT = 500
        
        # crop the left and right sides of the image
        
        # this is the destination points, where we want to transform the image to
        #destination_points = np.array([[WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT], [0, 0]], dtype='float32')
        # Assume `crop_amount` is the amount you want to crop from both sides
        crop_amount_sides = -20  # Change this to the amount you want to crop
        
        crop_amount_top = -10
        # Adjust the destination points
        destination_points = destination_points = np.array([[WIDTH - crop_amount_sides, crop_amount_top], [WIDTH - crop_amount_sides, HEIGHT], [crop_amount_sides, HEIGHT],[crop_amount_sides, crop_amount_top]], dtype='float32')

        # get the transformation matrix for a top down view of the original image
        self.transformation_matrix = cv2.getPerspectiveTransform(corners_for_perspective_transform, destination_points)
        self.inverse_tranformation_matrix = cv2.getPerspectiveTransform(destination_points, corners_for_perspective_transform)
        
        # apply the transformation matrix to the cropped image
        self.warped_img = cv2.warpPerspective(self.og_img, self.transformation_matrix, (WIDTH, HEIGHT))
        # crop_amount = 20
        # self.warped_img = self.warped_img[:, crop_amount:-crop_amount]
        
        # rotate image 180 degrees
        # self.warped_img = cv2.rotate(self.warped_img, cv2.ROTATE_180)
        # mirror the image
        # self.warped_img = cv2.flip(self.warped_img, 1)

        # show the top down view of the chessboard
        # cv2.imshow('Warped image', self.warped_img)


    def get_pixel_coordinates_using_shi_tomasi_on_warped_img(self):
        temp_coords = []

        img_copy = self.warped_img.copy()
        # convert warped image to grayscale
        self.warped_img = cv2.cvtColor(self.warped_img, cv2.COLOR_BGR2GRAY)
        # apply a strong gaussian blur to prevent numbers and letters of the chessboard from being detected as corners
        self.warped_img = cv2.GaussianBlur(self.warped_img, (15, 15), 0)
        # using the shi- tomasi algorithm to find each corner of each square on the chessboard
        # try:
        corners = cv2.goodFeaturesToTrack(self.warped_img, 81, 0.02, 40)
        # improve accuracy of corners, using subpixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.001)
        corners = cv2.cornerSubPix(self.warped_img, corners, (11, 11), (-1, -1), criteria)
        # this guarantees that we have 81 corners, if not then we cant use the image
        if len(corners) != 81:
            if len(corners) > 81:
                print("Too many corners found")
            else:
                print("Not enough corners found")
        else:
            
            # draw the corners on the warped image
            for corner in corners:
                cv2.circle(img_copy, (int(corner[0][0]), int(corner[0][1])), 5, (0, 0, 255), -1)

            # show the warped image with the corners
            cv2.imshow('deteted corners on warped image', img_copy)

            # convert warped plane corners to original plane corners using the inverse transformation matrix we got from the reverse perspective transform
            self.pixel_coordinates = np.dot(self.inverse_tranformation_matrix, np.array([corners[:, 0, 0], corners[:, 0, 1], np.ones(corners.shape[0])]))

            self.pixel_coordinates = self.pixel_coordinates[:2,:] / self.pixel_coordinates[2, :]

            self.pixel_coordinates = np.transpose(self.pixel_coordinates)

            # sort corners from accending x
           # self.pixel_coordinates = self.pixel_coordinates[self.pixel_coordinates[:,0].argsort()]
            # sort corners from accending y
           # self.pixel_coordinates = self.pixel_coordinates[self.pixel_coordinates[:,1].argsort(kind='mergesort')]

          
            corners = np.float32(self.pixel_coordinates)

            # Flatten the array to 2D
            corners = corners.reshape(-1, 2)

            # Sort the corners based on their y-coordinate
            corners = corners[np.argsort(corners[:, 1])]

            # Initialize an array to hold the sorted corners
            sorted_corners = []

            # For each row
            for i in range(0, len(corners), 9):  # Change 9 to the number of corners per row
                # Sort the row based on the x-coordinate and append it to the sorted corners
                sorted_corners.append(sorted(corners[i:i+9], key=lambda x: x[0]))

            self.pixel_coordinates = sorted_corners
            self.pixel_coordinates = np.reshape(self.pixel_coordinates, (81, 1, 2))
            
            # # undistort the corners if we do have the intrinsic and distortion matrix
            # if self.intrinsic_camera_matrix is not None and self.distortion_coeffcients is not None:
            #     self.pixel_coordinates = cv2.undistortPoints(self.pixel_coordinates, self.intrinsic_camera_matrix, self.distortion_coeffcients, P=self.intrinsic_camera_matrix)

            #draw the corners on the original image
            #for corner in self.pixel_coordinates:
            #     cv2.circle(self.og_img, (int(corner[0][0]), int(corner[0][1])), 5, (0, 0, 255), -1)

            # draw the corners using draw chessboard corners to see if they are in th correct order
            cv2.drawChessboardCorners(self.og_img, (9, 9), self.pixel_coordinates, True)
                    
   
        # except cv2.error:
        #     print("No corners found")


    def convert_pixel_coordinates_to_3d(self):
        inverse_camera_matrix = np.linalg.inv(self.camera_matrix)
        self.chessboard_3d_coordinates = np.dot(inverse_camera_matrix, self.pixel_coordinates)





def test_chessboard_recognition(chessboard_recogniser, frame):

    # use our class as process the images

    # set the original image and processing image to the frame
    chessboard_recogniser.og_img = frame
    chessboard_recogniser.processing_img = frame
    # pre process the image for getting the chessboard
    chessboard_recogniser._pre_process_img()
    # locate the chessboard by finding largest contour
    corners_for_perspective_transform = chessboard_recogniser.locate_chessboard_and_get_4_external_corners()

    if corners_for_perspective_transform is not None:
        # crop and transform the image to a top down view, passing in false so that we dont crop the image
        chessboard_recogniser.crop_and_get_top_down_view_of_chessboard(corners_for_perspective_transform)
        # get the pixel coordinates of each corner of each square on the chessboard, passing in false so that we draw the corners
        chessboard_recogniser.get_pixel_coordinates_using_shi_tomasi_on_warped_img()
        cv2.imshow('Warped image', chessboard_recogniser.warped_img)



    # show the images
    cv2.imshow('Original image', chessboard_recogniser.og_img)
    cv2.imshow('Processing image', chessboard_recogniser.processing_img)
    


def access_web_cam():
    chessboard_recogniser = ChessboardRecognition("32x32")
    #points = chessboard_recogniser.get_3d_chessboard_corners_for_calibration()
    #print(points)
    #chessboard_recogniser.calibrate_camera()
    desired_width = 1000
    desired_height = 1000

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        test_chessboard_recognition(chessboard_recogniser, frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            print(f"the shape of the array is {np.shape(chessboard_recogniser.pixel_coordinates)}")
            print(chessboard_recogniser.pixel_coordinates)
            # for corner in chessboard_recogniser.pixel_coordinates:
            #     print(corner)
            #     print("-"*100)

        # chessboard_recogniser.class_reset()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



access_web_cam()


