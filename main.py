import cv2
import math
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from pykalman import KalmanFilter

######################################################################
##############################FUNCTIONS###############################
######################################################################

# input: image and bounds for sphere hsv threshold
# output: scaled image and binary image where candidate pixels for sphere are 255; otherwise 0
def preprocess_image(frame, lowerBound, upperBound):
    # resize the frame to width of 600
    width = 600
    (h, w) = frame.shape[:2]
    r = width / float(w)
    dim = (width, int(h * r))
    scaled = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    # blur image and convert it to the HSV color space
    blurred = cv2.GaussianBlur(scaled, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # mask of blue
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    # perform a series of dilations and erosions to remove any small blobs left in the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return scaled, mask

# input: preprocessed image
# output: center and radius of sphere in image space
def find_sphere(mask):
    # find contours in the mask largest contour in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        # find minimum enclosing circle and center of contour
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # # alternatively we could use center of contour
        # M = cv2.moments(c)
        # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # return the center and radius
        return (x,y), radius
    else:
        return (False, False), False

# input: preprocessed image
# output: bounding rectangle in image space
def find_rectangle(mask):
    # find contours in the mask largest contour in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        # find minimum bounding rectangle of contour
        (x, y, w, h) = cv2.boundingRect(c)
        return (x, y, w, h)
    else:
        return (False, False, False, False)

# input: image, center point and radius of circle
# output: image with circle
def draw_circle(im, x, y, r):
    res = im.copy()
    thickness = r / 7
    cv2.circle(res, (int(x), int(y)), int(r), (107, 9, 143), int(thickness))
    cv2.circle(res, (int(x), int(y)), int(thickness * 1.5), (10, 231, 255), -1)
    return res

# input: true radius, focal length of camera, and perceived radius
# output: center and radius of sphere in image space
def distance_to_camera(trueRadius, focalLength, perceivedRadius):
    return (trueRadius * focalLength) / perceivedRadius

# input: distance from camera to sphere, true radius and perceived radius of sphere
# output: center and radius of sphere in image space
def get_focal_length(trueDistance, trueRadius, perceivedRadius):
    return (perceivedRadius * trueDistance) / trueRadius

# input: image space coordinates, distance from camera to sphere, focalLength, and width and height of image
# output: global coordinate of point
def get_world_coordinates(imX, imY, depth, focalLength, w, h):
    # change image space coordinates to be centered at center of image
    cx = imX - w/2
    cy = imY - h/2
    # calculate horizontal and vertical angles from z
    theta = math.atan(cx/focalLength)
    phi = math.atan(cy/focalLength)
    # get global coordinates
    x = depth * math.tan(theta)
    y = depth * math.atan(phi)
    z = math.sqrt(depth ** 2 - x ** 2 - y ** 2)
    return x, -1*y, z

# input: 2D data (array of x values array of y values, and 2D array of x and y values)
# output: smoothed data after applying a kalaman filter
def kalaman(a, b, ab):
    Transition_Matrix = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
    Observation_Matrix = [[1, 0, 0, 0], [0, 1, 0, 0]]
    xinit = a[0]
    yinit = b[0]
    vxinit = a[0] - a[1]
    vyinit = b[0] - b[1]
    initstate = [xinit, yinit, vxinit, vyinit]
    initcovariance = 1.0e-3 * np.eye(4)
    transistionCov = 1.0e-4 * np.eye(4)
    observationCov = 1.0e-1 * np.eye(2)
    kf = KalmanFilter(transition_matrices=Transition_Matrix,
                      observation_matrices=Observation_Matrix,
                      initial_state_mean=initstate,
                      initial_state_covariance=initcovariance,
                      transition_covariance=transistionCov,
                      observation_covariance=observationCov)
    return kf.filter(ab)

# input: array of 3D points
# output: smoothed data after applying kalaman filters
def smooth_points(points):
    points = np.array(points)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    xy = points[:,0:2]
    yz = points[:,1:3]
    xz = points[:,0:3:2]
    newxy, trash = kalaman(x,y,xy)
    newxz, trash = kalaman(x,z,xz)
    newyz, trash = kalaman(y,z,yz)
    newx = []
    newy = []
    newz = []
    for i in range(len(xy)):
        newx.append((newxy[i,0]+newxz[i,0])/2)
        newy.append((newxy[i, 1] + newyz[i, 0]) / 2)
        newz.append((newxz[i, 1] + newyz[i, 1]) / 2)
    return newx, newy, newz

######################################################################
#################################MAIN#################################
######################################################################

def show_webcam():
    # parameters
    lowerBound = (95, 65, 95)       # hsv threshold values
    upperBound = (120, 260, 265)
    TRUE_RADIUS = 0.6875            # radius of sphere (inches)
    CALIBRATION_DISTANCE = 24       # distance from camera to sphere center during calibration (inches)
    cx, cy = (300, 168.5)           # center pixel of scaled frame
    delay = 35                      # event delay = 1/FPS *1000
    pt = []                         # buffer for storing point in previous frame
    pts = []                        # buffer for storing tracked lines of points
    l = []                          # buffer for storing current line of points
    spaceDown = False               # space key state

    cam = cv2.VideoCapture(0)

    # calibration
    success = 0                     # number of consecutive successful calibration frames
    radii = []                      # list of calibration radii
    while True:
        # get current frame
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        # find sphere and display circle
        img, mask = preprocess_image(img, lowerBound, upperBound)
        (x, y), radius = find_sphere(mask)
        if x == False:
            cv2.imshow("my cam", img)
            if cv2.waitKey(delay) == 27:
                break  # esc to quit
            continue
        img = draw_circle(img, x, y, radius)
        # if observed center is close to image center store radius
        if math.sqrt((cx - x) ** 2 + (cy - y) ** 2) < 3:
            success += 1
            radii.append(radius)
            # display circle at center of screen
            if success > 24:
                cv2.circle(img, (int(cx), int(cy)), 12, (77, 77, 255), -1)
            if success > 12:
                cv2.circle(img, (int(cx), int(cy)), 8, (79, 152, 255), -1)
            cv2.circle(img, (int(cx), int(cy)), 5, (105, 255, 252), -1)
            # display text asking the user to hold
            font = cv2.FONT_HERSHEY_PLAIN
            if success % 12 > 7:
                cv2.putText(img, 'HOLD', (265, 30), font, 2, (129, 133, 93), 10, cv2.LINE_AA)
                cv2.putText(img, 'HOLD', (265, 30), font, 2, (86, 120, 54), 5, cv2.LINE_AA)
            elif success % 12 > 3:
                cv2.putText(img, 'HOLD', (265, 30), font, 2, (86, 120, 54), 5, cv2.LINE_AA)
            cv2.putText(img, 'HOLD', (265, 30), font, 2, (140, 184, 99), 2, cv2.LINE_AA)
        elif success > 0:
            success = 0
            radii = []
        else:
            # display red circle at center of screen
            cv2.circle(img, (int(cx), int(cy)), 8, (49 , 33, 194), -1)
            cv2.circle(img, (int(cx), int(cy)), 5, (99, 87, 207), -1)
            # display text asking the user to center sphere
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(img, 'CENTER SPHERE ON DOT', (105, 30), font, 2, (49 , 33, 194), 6, cv2.LINE_AA)
            cv2.putText(img, 'CENTER SPHERE ON DOT', (105, 30), font, 2, (99, 87, 207), 2, cv2.LINE_AA)
        # display image
        cv2.imshow("my cam", img)
        # check for key event
        key = cv2.waitKey(delay)
        if key == 27:
            break  # esc to quit
        # if we have calibrated a sufficient number of consecutive frames, calculate the focalLength
        if success > 36:
            r = np.average(radii)
            focalLength = get_focal_length(CALIBRATION_DISTANCE, TRUE_RADIUS, r)
            break

    # calculate and display 3D points
    # initialize figure for displaying points
    fig = pyplot.figure()
    ax = Axes3D(fig, azim=-90, elev=90)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    while True:
        # get current frame
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        # find sphere and display circle
        img, mask = preprocess_image(img, lowerBound, upperBound)
        (x, y), radius = find_sphere(mask)
        if x == False:
            cv2.imshow("my cam", img)
            # check for key event
            key = cv2.waitKey(delay)
            if key == 27:
                break  # esc to quit
            continue
        img = draw_circle(img, x, y, radius)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, 'HOLD SPACE TO DRAW', (105, 30), font, 2, (49, 33, 194), 6, cv2.LINE_AA)
        cv2.putText(img, 'HOLD SPACE TO DRAW', (105, 30), font, 2, (99, 87, 207), 2, cv2.LINE_AA)
        # display image
        cv2.imshow("my cam", img)
        # check for key event
        key = cv2.waitKey(delay)
        if key == 27: # esc key
            break  # quit
        elif key == 32: # space bar is pressed
            # find the depth
            d = distance_to_camera(TRUE_RADIUS, focalLength, radius)
            # calculate world coordinate
            x, y, z = get_world_coordinates(x, y, d, focalLength, img.shape[0], img.shape[1])
            if spaceDown:   # it was pressed in the last frame
                # draw line from last point to this point
                line = Line3D((pt[0],x),(pt[1],y),(pt[2],z))
                ax.add_line(line)
                ax.scatter(x,y,z, marker=".")
                # store point in buffer
                pt = [x, y, z]
                l.append(pt)
                pyplot.pause(0.00000001)
            else:   # it wasn't pressed in last frame
                spaceDown = True
                # store point in buffer
                pt = [x, y, z]
                l.append(pt)
        elif spaceDown: # space bar was just lifted
            # clear buffer
            pt = []
            pts.append(l)
            l = []
            spaceDown = False

    cv2.destroyAllWindows()

    # smooth data and plot final result
    fig2 = pyplot.figure()
    ax = Axes3D(fig2, azim=90, elev=-90)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis()
    for line in pts:
        if len(line) > 1:
            x, y, z = smooth_points(line)
            ax.scatter(x, y, z, marker=".")
            for i in range(len(x)-1):
                line = Line3D((x[i], x[i+1]), (y[i], y[i+1]), (z[i], z[i+1]))
                ax.add_line(line)
    pyplot.show()

def main():
    show_webcam()

if __name__ == '__main__':
    main()

# ######################################################################
# ##############################TEST VIDEO##############################
# ######################################################################
# lowerBound1, upperBound1 = (1, 80, 170), (10, 150, 230)
# lowerBound2, upperBound2 = (175, 50, 170), (180, 120, 260)
# TRUE_RADIUS = 0.65625
# focalLength = 453.2713099888393
# pts = []
#
# # load video
# cap = cv2.VideoCapture('Images/test.mp4')
# out = cv2.VideoWriter('Images/test1results.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (600, 337))
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret == True:
#         frame, mask = preprocess_image(frame, lowerBound1, upperBound1, lowerBound2, upperBound2)
#         (x, y), radius = find_sphere(mask) # need to make sure red shirt isn't showing
#         frame = draw_circle(frame,x,y,radius)
#         d = distance_to_camera(TRUE_RADIUS, focalLength, radius)
#         x, y, z = get_world_coordinates(x, y, d, focalLength, 600, 337)
#         pts.append([x, y, z])
#         # cv2.imshow('frame',frame)
#         # # cv2.waitKey(0)
#         # if cv2.waitKey(0) & 0xFF == ord('q'):
#         #     break
#         out.write(frame)
#     else:
#         break
# cap.release()
# out.release()
# cv2.destroyAllWindows()
#
#
# ######################################################################
# ###########################TEST CALIBRATION###########################
# ######################################################################
# lowerBound1, upperBound1 = (0, 89, 20), (7, 225, 255)
# lowerBound2, upperBound2 = (175,89,20), (180, 225, 225)
# # distance measured in inches
# TRUE_RADIUS = 0.65625
# CALIBRATION_DISTANCE = 24
#
# frame = cv2.imread("Images/calibration.jpg")
# frame, mask = preprocess_image(frame,lowerBound1,upperBound1,lowerBound2,upperBound2)
# # (x, y), radius = find_sphere(mask) # need to make sure red shirt isn't showing
# (x, y), radius = ((310.84747314453125, 171.4067840576172), 12.394137382507324)
# focalLength = get_focal_length(CALIBRATION_DISTANCE, TRUE_RADIUS, radius)
#
# cv2.imshow("image", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
