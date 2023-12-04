import cv2
import numpy as np
import pickle as pkl


class ChessboardRecognition:

    def __init__(self, CHESSBOARD_DIMENSIONS):
        self.original_img = None
        self.drawing_original_img = None # original image
        self.processing_img = None # mage used for processing
        self.warped_img = None # image after perspective transform for top down view

        self.transformation_matrix = None # matrix to apply perspective transform for top down view
        
        self.inverse_tranformation_matrix = None # matrix to get from top down view to original view

        self.pixel_coordinates = None # the coordinate of each corner in each square in the original image
        self.WIDTH = float(CHESSBOARD_DIMENSIONS.strip().split("x")[0])/8 # width of chessboard in cm
        self.HEIGHT = float(CHESSBOARD_DIMENSIONS.strip().split("x")[1])/8 # height of chessboard in cm

        self.INTRINSIC_CAMERA_MATRIX = None # intrinsic matrix will be used for pixel to 3d conversion
        self.EXTRINSIC_CAMERA_MATRIX = None # will be used for pixel to 3d conversion
        self.DISTORTION_COEFFICIENT = None # will be used to undistort the cameras

        self.chessboard_3d_coordinates = None # stores the 3d coordinates once its been converted from pixel coordinates


    def get_3d_chessboard_corners_for_calibration(self):
        # store the 3d corners for the calibration
        NUM_CORNERS_X = 9
        NUM_CORNERS_Y = 9

        corner_coordinates = np.mgrid[0:NUM_CORNERS_X, 0:NUM_CORNERS_Y].T.reshape(-1,2)
        corner_coordinates = corner_coordinates.astype(float)
        corner_coordinates[:, 0] *= self.WIDTH
        corner_coordinates[:, 1] *= self.HEIGHT

        chessboard_3d_corners = np.hstack((corner_coordinates, np.zeros((NUM_CORNERS_X * NUM_CORNERS_Y, 1))))
        return chessboard_3d_corners
    


   

    def get_imgs_and_their_corners_for_calibration(self):
        DESIRED_WIDTH = 1000
        DESIRED_HEIGHT = 1000
        NUM_IMAGES_USED = 0
        pixel_coordinates_for_calibration = []
        chessboard_3d_corners_for_calibration = []
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)

        #get the frames from the camera until we have 20 images for calibration
        while NUM_IMAGES_USED < 10:
            # Capture frame-by-frame
            ret, frame = cap.read()
            self.recognise_corners(frame)
        
            # if the user presses s then we save the image and object points for calibration. we will change this we start combining this with the robotic arm to move.
            if cv2.waitKey(1) & 0xFF == ord('s'):
                
                if self.pixel_coordinates is not None:
                    pixel_coordinates_for_calibration.append(self.pixel_coordinates)
                    chessboard_3d_corners_for_calibration.append(self.get_3d_chessboard_corners_for_calibration())
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

        return pixel_coordinates_for_calibration,  chessboard_3d_corners_for_calibration


    def calibrate_camera(self):
        pixel_coordinates_for_calibration, chessboard_3d_corners_for_calibration = self.get_imgs_and_their_corners_for_calibration()
        # reshape pixel_coordinates_for_calibration to 10,49,1,2
        pixel_coordinates_for_calibration = np.reshape(pixel_coordinates_for_calibration,((len(pixel_coordinates_for_calibration)),81,1,2))
        # reshape chessboard_3d_corners_for_calibration to 10,49,1,3
        chessboard_3d_corners_for_calibration = np.reshape(chessboard_3d_corners_for_calibration,(len(chessboard_3d_corners_for_calibration),81,1,3))
        # calibrate the camera using object and image points
        ret, intrinsic_matrix, distortion_coefficients, rotation_vector, translation_vector = cv2.calibrateCamera(chessboard_3d_corners_for_calibration, pixel_coordinates_for_calibration, (self.original_img.shape[1], self.original_img.shape[0]), None, None)

        print("camera has been calibrated")
       
        # calulate the extrinsic parametres with solve pnp
        ret, rotation_vector, translation_vector = cv2.solvePnP(chessboard_3d_corners_for_calibration[0], pixel_coordinates_for_calibration[0], intrinsic_matrix, distortion_coefficients)

        # convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        # get the extrinsic matrix
        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector))

        print("calculated the extrinsic matrix")

        # calculate the mean error for the intrinsic and extrinsic matrix
        mean_error = 0
        for i in range(len(chessboard_3d_corners_for_calibration)):
            imgpoints2, _ = cv2.projectPoints(chessboard_3d_corners_for_calibration[i], rotation_vector, translation_vector, intrinsic_matrix, distortion_coefficients)
            error = cv2.norm(pixel_coordinates_for_calibration[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        mean_error /= len(chessboard_3d_corners_for_calibration)
        print("mean error: ", mean_error)
       

        
        return intrinsic_matrix, extrinsic_matrix, distortion_coefficients    

    def _resize_img(self):
        # constants for resizing image
        SCALE_PERCENT = 50
        # Resize image
        new_width = int(self.drawing_original_img.shape[1] * SCALE_PERCENT / 100)
        new_height = int(self.drawing_original_img.shape[0] * SCALE_PERCENT / 100)
        new_dimension = (new_width, new_height)
        self.processing_img = cv2.resize(self.drawing_original_img, new_dimension, interpolation=cv2.INTER_AREA)


    def _pre_process_img(self):
        
        # undistort the image if we have the camera matrix and distortion coefficients
        if self.INTRINSIC_CAMERA_MATRIX is not None and self.DISTORTION_COEFFICIENT is not None:
            self.drawing_original_img = cv2.undistort(self.drawing_original_img, self.INTRINSIC_CAMERA_MATRIX, self.DISTORTION_COEFFICIENT)

        THRESHOLD = 100
        # Convert to gray scale image
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
                # draw the corners on the original image with the index of the list of corners
                # for i, corner in enumerate(corners):
                #     # convert corner to integer
                #     corner = corner.astype(int)

                #     cv2.circle(self.drawing_original_img, tuple(corner), 5, (0, 0, 255), -1)
                #     cv2.putText(self.drawing_original_img, str(i), tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                return corners
        except IndexError:
            pass


    def crop_and_get_top_down_view_of_chessboard(self, corners_for_perspective_transform):
        # the required width and height of the warped image
        WIDTH = 500
        HEIGHT = 500
        
        # crop the left and right sides of the image
        
       
        # Assume `crop_amount` is the amount you want to crop from both sides
        crop_amount_sides = -20  # Change this to the amount you want to crop
        
        crop_amount_top = -10
        # Adjust the destination points
        destination_points = destination_points = np.array([[WIDTH - crop_amount_sides, crop_amount_top], [WIDTH - crop_amount_sides, HEIGHT], [crop_amount_sides, HEIGHT],[crop_amount_sides, crop_amount_top]], dtype='float32')

        # get the transformation matrix for a top down view of the original image
        self.transformation_matrix = cv2.getPerspectiveTransform(corners_for_perspective_transform, destination_points)
        self.inverse_tranformation_matrix = cv2.getPerspectiveTransform(destination_points, corners_for_perspective_transform)
        
        # apply the transformation matrix to the cropped image
        self.warped_img = cv2.warpPerspective(self.original_img, self.transformation_matrix, (WIDTH, HEIGHT))
        

        # show the top down view of the chessboard
        # cv2.imshow('Warped image', self.warped_img)

    
  
    def get_pixel_coordinates_using_shi_tomasi_on_warped_img(self):
    
        img_copy = self.warped_img.copy()
        # convert warped image to grayscale
        self.warped_img = cv2.cvtColor(self.warped_img, cv2.COLOR_BGR2GRAY)
        # apply a strong gaussian blur to prevent numbers and letters of the chessboard from being detected as corners
        self.warped_img = cv2.GaussianBlur(self.warped_img, (15, 15), 0)
        # using the shi- tomasi algorithm to find each corner of each square on the chessboard
        try:
            corners = cv2.goodFeaturesToTrack(self.warped_img, 81, 0.02, 40)
            # improve accuracy of corners, using subpixel accuracy
            if corners is None:
                print("no corners found")
            elif len(corners) > 81:
                print("Too many corners found")
            elif len(corners) < 81:
                print("Not enough corners found")
            else:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.00001)
                corners = cv2.cornerSubPix(self.warped_img, corners, (11, 11), (-1, -1), criteria)
                # this guarantees that we have 81 corners, if not then we cant use the image
                corners = corners.reshape(-1, 2)
                # Sort the corners based on their y-coordinate
                corners = corners[np.argsort(corners[:, 1])[::-1]]
                # Initialize an array to hold the sorted corners
                sorted_corners = []
                # For each row
                for i in range(0, len(corners), 9):  # Change 9 to the number of corners per row
                    # Sort the row based on the x-coordinate and append it to the sorted corners
                    sorted_corners.append(sorted(corners[i:i+9], key=lambda x: x[0]))

                corners = np.reshape(sorted_corners, (81, 1, 2))
                # for i,corner in enumerate(corners):
                #     cv2.circle(img_copy, (int(corner[0][0]), int(corner[0][1])), 2, (0, 0, 255), -1)
                #     cv2.putText(img_copy, str(i+1), (int(corner[0][0]), int(corner[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # # show the warped image with the corners
                # cv2.imshow('deteted corners on warped image', img_copy)

                # convert warped plane corners to original plane corners using the inverse transformation matrix we got from the reverse perspective transform
                self.pixel_coordinates = np.dot(self.inverse_tranformation_matrix, np.array([corners[:, 0, 0], corners[:, 0, 1], np.ones(corners.shape[0])]))
                self.pixel_coordinates = self.pixel_coordinates[:2,:] / self.pixel_coordinates[2, :]
                self.pixel_coordinates = np.transpose(self.pixel_coordinates)
                corners = np.float32(self.pixel_coordinates)
                self.pixel_coordinates = np.reshape(self.pixel_coordinates, (81, 1, 2))

                #draw the corners on the original image and add text of the current index to the
                # for i in range(len(corners)):
                #     cv2.circle(self.drawing_original_img, (int(corners[i][0]), int(corners[i][1])), 2, (0, 0, 255), -1)
                #     #add the order of the corners to the image
                #     cv2.putText(self.drawing_original_img, str(i), (int(corners[i][0]), int(corners[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # draw chessboard corners with drawchessboarcorners
                cv2.drawChessboardCorners(self.drawing_original_img, (9, 9), corners, True)
                # show the original image with the corners
                
        
        except ValueError:
            print("No corners found")


    def convert_pixel_coordinates_to_3d(self):
        
        self.pixel_coordinates = np.reshape(self.pixel_coordinates, (81,2))
        undistored_corners = cv2.undistortPoints(self.pixel_coordinates, self.INTRINSIC_CAMERA_MATRIX, self.DISTORTION_COEFFICIENT, P = self.INTRINSIC_CAMERA_MATRIX)

        self.chessboard_3d_coordinates = np.zeros((81, 3))

    # For each 2D corner, create a 3D point.
        for i in range(81):
            # The x and y coordinates are the same as the 2D point.
            self.chessboard_3d_coordinates[i, :2] = undistored_corners[i, 0, :] * self.WIDTH
            # The z coordinate is 0.
            self.chessboard_3d_coordinates[i, 2] = 0

        # Rotate the 3D points using the extrinsic matrix.
        self.chessboard_3d_coordinates = np.dot(self.chessboard_3d_coordinates, self.EXTRINSIC_CAMERA_MATRIX[:, :3].T) + self.EXTRINSIC_CAMERA_MATRIX[:, 3]

        
        

        # draw corners onto original image and each corner label the corresponding 3d coordinate
        for i in range(len(self.pixel_coordinates)):
        # Draw a circle at the corner's location.
            cv2.circle(self.original_img, (int(self.pixel_coordinates[i][0]), int(self.pixel_coordinates[i][1])), 2, (0, 0, 255), -1)
            
            # Get the corresponding 3D coordinate.
            coord_3d = self.chessboard_3d_coordinates[i]
            
            # Convert the 3D coordinate to a string.
            coord_3d_str = f'({coord_3d[0]:.2f}, {coord_3d[1]:.2f}, {coord_3d[2]:.2f})'
            
            # Draw the string next to the corner.
            cv2.putText(self.original_img, coord_3d_str, (int(self.pixel_coordinates[i][0]), int(self.pixel_coordinates[i][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
          







    def recognise_corners(self, frame):

        # use our class as process the images
        # set the original image and processing image to the frame
        self.original_img = frame
        self.drawing_original_img = frame
        self.processing_img = frame
        # pre process the image for getting the chessboard
        self._pre_process_img()
        # locate the chessboard by finding largest contour
        corners_for_perspective_transform = self.locate_chessboard_and_get_4_external_corners()

        if corners_for_perspective_transform is not None:
            #crop and transform the image to a top down view, passing in false so that we dont crop the image
            self.crop_and_get_top_down_view_of_chessboard(corners_for_perspective_transform)

            #get the pixel coordinates of each corner of each square on the chessboard, passing in false so that we draw the corners
            self.get_pixel_coordinates_using_shi_tomasi_on_warped_img()
            cv2.imshow('Warped image', self.warped_img)
            self.convert_pixel_coordinates_to_3d()


        # show the images
        cv2.imshow('original image', self.original_img)
        cv2.imshow('drawing original image', self.drawing_original_img)
        cv2.imshow('Processing image', self.processing_img)
    


def access_web_cam():
    chessboard_recogniser = ChessboardRecognition("32x32")
    # object_points = chessboard_recogniser.get_3d_chessboard_corners_for_calibration()
    # print(object_points)

    #intrinsic_matrix, extrinsic_matrix, distortion_coefficients = chessboard_recogniser.calibrate_camera()
    

    # save the camera matrix and distortion coefficients and extrinsic matrix with pickle
    # with open("camera_matrix.pkl", "wb") as f:
    #     pkl.dump(intrinsic_matrix, f)
    # with open("distortion_coefficients.pkl", "wb") as f:
    #     pkl.dump(distortion_coefficients, f)
    # with open("extrinsic_matrix.pkl", "wb") as f:
    #     pkl.dump(extrinsic_matrix, f)

    

    # load pickle files
    with open("camera_matrix.pkl", "rb") as f:
        intrinsic_matrix = pkl.load(f)
    with open("distortion_coefficients.pkl", "rb") as f:
        distortion_coefficients = pkl.load(f)
    with open("extrinsic_matrix.pkl", "rb") as f:
        extrinsic_matrix = pkl.load(f)

    print(f"intrinsic_matrix is {intrinsic_matrix}")
    print(f"distortion_coefficients is {distortion_coefficients}")
    print(f"extrinsic_matrix is {extrinsic_matrix}")


    chessboard_recogniser.INTRINSIC_CAMERA_MATRIX = intrinsic_matrix
    chessboard_recogniser.EXTRINSIC_CAMERA_MATRIX = extrinsic_matrix
    chessboard_recogniser.DISTORTION_COEFFICIENT = distortion_coefficients




    DESIRED_WIDTH = 1000
    DESIRED_HEIGHT = 1000

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    while True:#chessboard_recogniser.pixel_coordinates is None:
        # Capture frame-by-frame
        ret, frame = cap.read()
        chessboard_recogniser.recognise_corners(frame)
        # chessboard_recogniser.class_reset()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()



access_web_cam()
