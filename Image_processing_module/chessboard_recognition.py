import cv2
import numpy as np
import math

from time import sleep

class ChessboardRecognition:
    def __init__(self):
        self.og_img = None
        self.processing_img = None
        
    

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

       

        

        cv2.drawContours(self.og_img, [largest_contour], -1, (0, 255, 0), 2)  # Draw the largest contour in green


      
        
        
       
   
       

   


    






def test_chessboard_recognition(chessboard_recogniser, frame):

    # use our class as process the images
    chessboard_recogniser.og_img = frame
    chessboard_recogniser.processing_img = frame
    chessboard_recogniser._pre_process_img()
    chessboard_recogniser.locate_chessboard()
    



def access_web_cam():
    chessboard_recogniser = ChessboardRecognition()
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
      

        test_chessboard_recognition(chessboard_recogniser, frame)

        cv2.imshow('Original image', chessboard_recogniser.og_img)
        cv2.imshow('Processing image', chessboard_recogniser.processing_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows() 

access_web_cam()


