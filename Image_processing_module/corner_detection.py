import cv2
import numpy as np



class ChessboardCornerDetection:

    def __init__(self):
        self.original_img = None
        self.drawing_original_img = None # original image
        self.processing_img = None # mage used for processing
        self.warped_img = None # image after perspective transform for top down view
        self.transformation_matrix = None # matrix to apply perspective transform for top down view
        self.inverse_tranformation_matrix = None # matrix to get from top down view to original view
        self.pixel_coordinates = None # the coordinate of each corner in each square in the original image
        self.uci_positions = None
        
        
      
    def __pre_process_img(self):
        THRESHOLD = 100
        # Convert to gray scale image
        self.processing_img = cv2.cvtColor(self.processing_img, cv2.COLOR_BGR2GRAY)
        # apply gaussian blur to remove noise
        self.processing_img = cv2.GaussianBlur(self.processing_img, (7, 7), 0)
        # Apply threshold to get binary image, helps with localising chessboard
        # uses otsu thresholding for optimal threshold value
        ret, self.processing_img = cv2.threshold(self.processing_img, THRESHOLD, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    def __locate_chessboard_and_get_4_external_corners(self):
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


    def __crop_and_get_top_down_view_of_chessboard(self, corners_for_perspective_transform):
        # the required width and height of the warped image
        WIDTH = 500
        HEIGHT = 500
        # Assume `crop_amount` is the amount you want to crop from both sides
        CROP_AMOUNT_SIDES = -20  # Change this to the amount you want to crop
        CROP_AMOUNT_TOP = -10
        # Adjust the destination points
        destination_points = destination_points = np.array([[WIDTH - CROP_AMOUNT_SIDES, CROP_AMOUNT_TOP], [WIDTH - CROP_AMOUNT_SIDES, HEIGHT], [CROP_AMOUNT_SIDES, HEIGHT],[CROP_AMOUNT_SIDES, CROP_AMOUNT_TOP]], dtype='float32')
        # get the transformation matrix for a top down view of the original image
        self.transformation_matrix = cv2.getPerspectiveTransform(corners_for_perspective_transform, destination_points)
        self.inverse_tranformation_matrix = cv2.getPerspectiveTransform(destination_points, corners_for_perspective_transform)
        # apply the transformation matrix to the cropped image
        self.warped_img = cv2.warpPerspective(self.drawing_original_img, self.transformation_matrix, (WIDTH, HEIGHT))
        # show the top down view of the chessboard
        #cv2.imshow('Warped image', self.warped_img)

    
    def __get_pixel_coordinates_using_shi_tomasi_on_warped_img(self):
        #img_copy = self.warped_img.copy()
        # convert warped image to grayscale
        self.warped_img = cv2.cvtColor(self.warped_img, cv2.COLOR_BGR2GRAY)
        # apply a strong gaussian blur to prevent numbers and letters of the chessboard from being detected as corners
        self.warped_img = cv2.GaussianBlur(self.warped_img, (13, 13), 0)
        # using the shi- tomasi algorithm to find each corner of each square on the chessboard
        # show the warped image 
        #cv2.imshow('Warped image', self.warped_img)

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
                for i in range(len(corners)):
                    cv2.circle(self.drawing_original_img, (int(corners[i][0]), int(corners[i][1])), 2, (0, 0, 255), -1)
                    #add the order of the corners to the image
                    cv2.putText(self.drawing_original_img, str(i), (int(corners[i][0]), int(corners[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                # draw chessboard corners with drawchessboarcorners
                #cv2.drawChessboardCorners(self.drawing_original_img, (9, 9), corners, True)
                # show the original image with the corners
        except ValueError:
            print("No corners found")


    def map_corners_to_uci_positions(self):
        uci_positions = {}
        corners = np.array(self.pixel_coordinates).squeeze()
        for i in range(8):
            for j in range(8):
                # Calculate the indices of the corners in the sorted corners array
                bottom_left = i * 9 + j
                bottom_right = i * 9 + j + 1
                top_left = (i + 1) * 9 + j
                top_right = (i + 1) * 9 + j + 1
                # Assign a letter and number to the square
                letter = chr(ord('a') + j)
                number = i+1
                # Use the letter and number as the key, and the corresponding corners as the value
                uci_positions[f'{letter}{number}'] = [
                    corners[bottom_left].tolist(),
                    corners[bottom_right].tolist(),
                    corners[top_left].tolist(),
                    corners[top_right].tolist()
                ]
        self.uci_positions = uci_positions

    
    def recognise_corners(self, frame):
        # use our class to process the images and get corners
        # set the original image and processing image to the frame
        self.original_img = frame
        self.drawing_original_img = frame.copy()
        self.processing_img = frame.copy()
        # pre process the image for getting the chessboard
        self.__pre_process_img()
        # locate the chessboard by finding largest contour
        corners_for_perspective_transform = self.__locate_chessboard_and_get_4_external_corners()

        if corners_for_perspective_transform is not None:
            #crop and transform the image to a top down view, passing in false so that we dont crop the image
            self.__crop_and_get_top_down_view_of_chessboard(corners_for_perspective_transform)
            #get the pixel coordinates of each corner of each square on the chessboard, passing in false so that we draw the corners
            self.__get_pixel_coordinates_using_shi_tomasi_on_warped_img()
            #map the corners to uci positions
            if self.pixel_coordinates is not None:
                self.map_corners_to_uci_positions()

            #cv2.imshow('Warped image', self.warped_img)
        #show the images
        cv2.imshow('original image', self.original_img)
        cv2.imshow('drawing original image', self.drawing_original_img)
        cv2.imshow('Processing image', self.processing_img)




def test():
    # create a chessboard corner detection object
    chessboard_corner_detection = ChessboardCornerDetection()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        chessboard_corner_detection.recognise_corners(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(chessboard_corner_detection.pixel_coordinates)


#test()




    
