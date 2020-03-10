import cv2
import math
import numpy as np
import sys
import os
import dlib
import glob

def face_orientation(frame, landmarks):
    size = frame.shape #(height, width, color_channel)

    image_points = np.array([
                            landmarks["30"], # Nose
                            landmarks["8"],  # Chin
                            landmarks["45"], # Left eye left corner
                            landmarks["36"], # Right eye right cornee
                            landmarks["54"], # Left Mouth corner
                            landmarks["48"]  # Right mouth corner
                        ], dtype="double")

    #  ---------------------------------
    # (441,649),   # Nose tip
    # (415,924),   # Chin
    # (574,534),   # Left eye left corner
    # (278,513),   # Right eye right corne
    # (528,729),   # Left Mouth corner
    # (305,714)    # Right mouth corner
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])
 
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    return imgpts, modelpts, ((int(roll)), (int(pitch)), (int(yaw))), (landmarks["4"], landmarks["5"])


def process_input(predictor_path, image__path):
    """
        1).  Inputs Image + Predictor for face identification.
        2).  Finds out Roll, Yaw & Pitch Angles. 
    """
    predictor_path = predictor_path#sys.argv[1]
    image_path = image__path

    image = cv2.imread(image_path)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    win = dlib.image_window()
    img = dlib.load_rgb_image(image__path)

    win.clear_overlay()
    win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        # Draw the face landmarks on the screen.
        win.add_overlay(shape)
    r = 0
    g = 100
    b = 200
    index = 0
    landmark_list = {}
    for i in shape.parts():
        cords = (int(str(i).split(",")[0][1:]),int(str(i).split(",")[1][:-1]))
        image = cv2.circle(image, cords, 2, (b,g,r), -1)
        cv2.putText(image, str(index),cords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
        print(f"Index {index} :  {cords}")
        landmark_list[str(index)] = cords
        index +=1
        r +=5
        g +=5
        b +=5
    h,w,_ = image.shape
    win.add_overlay(dets)
    # dlib.hit_enter_to_continue()

    img_path = image__path #'pexels-photo-1090393.jpg' #img_info[0]
    frame = cv2.imread(img_path)
    # landmarks =  [int(float(i)) for i in img_info.split(',')]

    imgpts, modelpts, rotate_degree, nose = face_orientation(frame, landmark_list)
    cv2.line(frame, landmark_list["30"], tuple(imgpts[1].ravel()), (0,255,0), 3) #GREEN
    cv2.line(frame, landmark_list["30"], tuple(imgpts[0].ravel()), (255,0,), 3) #BLUE
    cv2.line(frame, landmark_list["30"], tuple(imgpts[2].ravel()), (0,0,255), 3) #RED

    remapping = [2,3,0,4,5,1]
    for j in range(len(rotate_degree)):
        cv2.putText(frame, ('{:05.2f}').format(float(rotate_degree[j])), (10, 30 + (50 * j)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
    h,w,_ = frame.shape
    cv2.imshow('my webcam', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('some.jpg', frame)





pred = "shape_predictor_68_face_landmarks.dat"
image = r'C:\Users\xyz\Desktop\MouseWithoutBorders\Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV-master\pexels-photo-1090393.jpg'
process_input(pred,image)