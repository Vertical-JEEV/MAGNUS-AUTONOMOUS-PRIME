import cv2
import numpy as np

def test_chessboard_recognition(frame):
    # find chessboard corners in the image
    ret, corners = cv2.findChessboardCorners(frame, (7, 7), None)
    # if found, add object points, image points (after refining them)
    if ret == True:
        # draw the corners
        
        
        cv2.drawChessboardCorners(frame, (7, 7), corners, ret)
        print(np.shape(corners))
        print(corners)
        print(1000*"-")
        return corners
    
def chessboard_3d_coords(sq_width, sq_height):
    objp = []
    for i in range(7):
        for j in range(7):
            objp.append([[i*sq_width, j*sq_height, 0]])
    
    return np.array(objp, dtype=np.float32)



def convert_image_point_to_3d_point(self):


    pass
        


objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.




desired_width = 800
desired_height = 800

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = test_chessboard_recognition(gray)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        imgpoints.append(corners)
        objpoints.append(chessboard_3d_coords(4, 4))
        print("saved")
        
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break
    # chessboard_recogniser.class_reset()
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error

mean_error = total_error/len(objpoints)
far_off = 0- mean_error

print("mean error: ", mean_error)
print("far off: ", far_off)





print("calibrated camera")

desired_width = 800
desired_height = 800

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
   
    # use the camera matrix and distortion coefficients to undistort
    
    dst1 = cv2.undistort(frame, mtx, dist, None, mtx)
   
   
    cv2.imshow('original frame', frame)
    cv2.imshow('undistorted with old camera matrix', dst1)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break