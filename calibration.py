import numpy as np
import cv2 as cv
import glob

grid1 = (5, 4)
grid2 = (4, 3)
framesize = (1280, 720)
objp = np.zeros((grid1[0] * grid1[1] + grid2[0] * grid2[1], 3), np.float32)
objp[:20, :2] = np.mgrid[0:grid1[0], 0:grid1[1]].T.reshape(-1, 2) * 55/3
objp[20:, :2] = np.mgrid[0:grid2[0], 0:grid2[1]].T.reshape(-1, 2) * 55/3 + 55/6
objpoints = []
imgpoints = []
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
parameters = cv.SimpleBlobDetector_Params()
parameters.filterByArea = True
parameters.minArea = 75
parameters.maxArea = 800
parameters.filterByCircularity = True
parameters.minCircularity = 0.01
parameters.filterByConvexity = True
parameters.minConvexity = 0.8
parameters.filterByInertia = True
parameters.minInertiaRatio = 0.01
detector = cv.SimpleBlobDetector_create(parameters)
images = glob.glob('point/*.jpg')
for image in images:
    n = 0
    a = []
    objpoints.append(objp)
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    keypoints = detector.detect(img)
    blobs = cv.drawKeypoints(img, keypoints, np.zeros((1, 1)), (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("blobs:")
    for k in keypoints: 
        n += 1
        a.append([[k.pt[0], k.pt[1]]])
        cv.circle(blobs, (int(k.pt[0]), int(k.pt[1])), 2, (0, 255, 0), -1)
        print(k.pt[0], k.pt[1])
    corners2 = cv.cornerSubPix(gray, np.array(a, dtype=np.float32), (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)
    print("count: ", n)
    cv.imshow('blobs', blobs)
    cv.waitKey(1000)
cv.destroyAllWindows()
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, framesize, None, None)
print("mat:\n", cameraMatrix)
print("dist:\n", dist)
