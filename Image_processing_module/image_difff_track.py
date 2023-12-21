import cv2
import numpy as np
from corner_detection import ChessboardCornerDetection



class ChessboardChangeDetector:
    def __init__(self):
        self.old_img = None
        self.new_img = None
        self.uci_positions = None
        self.last_detected_uci = None
        

    def update_image(self, new_img):
        self.old_img = self.new_img
        self.new_img = new_img

    def detect_changes(self):
        if self.old_img is not None and self.new_img is not None:
            # Convert the images to grayscale
            old_gray = cv2.cvtColor(self.old_img, cv2.COLOR_BGR2GRAY)
            new_gray = cv2.cvtColor(self.new_img, cv2.COLOR_BGR2GRAY)

            # apply histogram equalization to improve the contrast of the images and then apply Gaussian blurring
            old_gray = cv2.equalizeHist(old_gray)
            new_gray = cv2.equalizeHist(new_gray)


            old_gray = cv2.GaussianBlur(old_gray, (5, 5), 0)
            new_gray = cv2.GaussianBlur(new_gray, (5, 5), 0)

            # Compute the absolute difference between the old image and the new image
            diff = cv2.absdiff(old_gray, new_gray)
            cv2.imshow("Difference", diff)
            thresh = diff

        
            threshold_value = 50
            # Apply Otsu's thresholding
            _, thresh = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)

            #show the thresholded image 
            cv2.imshow("Thresholded", thresh)
            changed_positions = []
            # Iterate over the UCI positions
            
            for uci, corners in self.uci_positions.items():
                # Calculate the minimum and maximum x and y coordinates
                min_y = int(min(corners[0][1], corners[1][1], corners[2][1], corners[3][1])) 
                max_y = int(max(corners[0][1], corners[1][1], corners[2][1], corners[3][1])) 
                min_x = int(min(corners[0][0], corners[1][0], corners[2][0], corners[3][0])) 
                max_x = int(max(corners[0][0], corners[1][0], corners[2][0], corners[3][0])) 
              

                # Extract the region of interest from the thresholded image
                roi = thresh[min_y:max_y, min_x:max_x]
               
                # show the new image
                


                # If there are any changes in the region of interest, return the UCI position
                if np.any(roi):
                     # draw the bounding rectangle
                    new_img_copy = self.new_img.copy()
                    cv2.rectangle(new_img_copy, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
                    cv2.imshow("Chess Piece Tracker", new_img_copy)
                    cv2.waitKey(0)
                        
                    return uci
                    

        return None
    



def test():
    corners_detector = ChessboardCornerDetection()
    change_detector = ChessboardChangeDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)

    # First loop for corner detection
    while True:
        ret, frame = cap.read()
        if ret:
            corners_detector.recognise_corners(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):  # The user pressed the "c" key
            break
    cv2.destroyAllWindows()
    # Update the UCI positions
    change_detector.uci_positions = corners_detector.uci_positions
    ret, img = cap.read()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    color_index = 0

    # Loop over each UCI position
    for uci, corners in change_detector.uci_positions.items():
        # Calculate the minimum and maximum x and y coordinates
        min_y = int(min(corners[0][1], corners[1][1], corners[2][1], corners[3][1]))
        max_y = int(max(corners[0][1], corners[1][1], corners[2][1], corners[3][1]))
        min_x = int(min(corners[0][0], corners[1][0], corners[2][0], corners[3][0]))
        max_x = int(max(corners[0][0], corners[1][0], corners[2][0], corners[3][0]))

        # Draw a filled rectangle at the UCI position
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), colors[color_index], -1)

        # Update the color index
        color_index = (color_index + 1) % len(colors)


    cv2.imshow('Chess Piece Tracker', img)
    cv2.waitKey(0)

    # Second loop for change detection
    cv2.namedWindow('Chess Piece Tracker')
    while True:
        ret, img = cap.read()
        if ret:
            cv2.imshow('Chess Piece Tracker', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):  # The user pressed the "s" key
                print("s pressed")
                change_detector.update_image(img)
                changed_positions = change_detector.detect_changes()
                if changed_positions is not None:
                    print(f"Change detected at UCI positions {changed_positions}")
            elif key == ord("q"):  # The user pressed the "q" key
                break

    cv2.destroyAllWindows()

#test()