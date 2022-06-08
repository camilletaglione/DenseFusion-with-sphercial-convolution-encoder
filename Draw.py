from pickletools import uint8
from sqlite3 import paramstyle
import scipy.io.matlab as io
import cv2
import os
import time
import json
import numpy as np
import math
from skimage.measure import label, regionprops, find_contours

GT_path_i = "./ycb/Densefusion_iterative_result/" # format i.mat
GT_path_r = "./ycb/Densefusion_wo_refine_result/" # format i.mat
Mask_path = "./data/res2/" # format masked_i.png
Image_masked_path = "./data/res3/" # format imgmasked_i.png
img_path= "./data/resultat/" # format images_i.png
camera_parameter = "./data/cameras/asus-uw.json"
video_name = 'video.avi'
width = 640
height = 480

def mask2contour(mask):
    height, width = mask.shape
    border = np.zeros((height,width))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255
            border = np.uint8(border)
    return border

def mask2bbox(mask):
    bboxes = []

    border = mask2contour(mask)
    lbl = label(border)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]
        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def add_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

def quaternion_rotation_matrix(Q):
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])                           
    return rot_matrix

def isRotationMatrix(R):
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6

def rotationMatrix2EulerAngles(R):

    #assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def load_cam(path):
    file = open(path)
    data = json.load(file)
    data = data['rig']
    K = np.zeros((3,3))
    count = 0
    for i in data['camera']:
        if count==0:
            param = i["camera_model"]
            intrisec = param["params"]
            K[0,0] = (intrisec[0])#/1000
            K[1,1] = (intrisec[1])#/1000
            K[0,2] = (intrisec[2])#/1000
            K[1,2] = (intrisec[3])#/1000
            K[2,2] = 1
            dist = (intrisec[4],intrisec[5],0,0,intrisec[6])
        count+=1
    file.close() 
    return K, dist

def draw_axis(img, R, t, K, scale=30, dist=None):
    """
    Draw a 6dof axis (XYZ -> RGB) in the given rotation and translation
    :param img - rgb numpy array
    :rotation_vec - euler rotations, numpy array of length 3,
                    use cv2.Rodrigues(R)[0] to convert from rotation matrix
    :t - 3d translation vector, in meters (dtype must be float)
    :K - intrinsic calibration matrix , 3x3
    :scale - factor to control the axis lengths
    :dist - optional distortion coefficients, numpy array of length 4. If None distortion is ignored.
    """
    rotV, _ = cv2.Rodrigues(R)
    #print(rotV)
    #img = img.astype(np.float32)
    dist = np.zeros(4, dtype=float) if dist is None else dist
    points = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, dist)
    if np.isnan(axisPoints).any() == False:
        origin = tuple(axisPoints[3].ravel())
        p1 = tuple(axisPoints[0].ravel())
        p2 = tuple(axisPoints[1].ravel())
        p3 = tuple(axisPoints[2].ravel())
        img = cv2.line(img,(int(origin[0]), int(origin[1])),(int(p1[0]), int(p1[1])), (255,0,0), 3)
        img = cv2.line(img,(int(origin[0]), int(origin[1])),(int(p2[0]), int(p2[1])), (0,255,0), 3)
        img = cv2.line(img,(int(origin[0]), int(origin[1])),(int(p3[0]), int(p3[1])), (0,0,255), 3)
        #img = cv2.line(img,(int(origin[0]+320), int(origin[1]+240)),(int(p1[0]+320), int(p1[1]+240)), (255,0,0), 3)
        #img = cv2.line(img,(int(origin[0]+320), int(origin[1]+240)),(int(p2[0]+320), int(p2[1]+240)), (0,255,0), 3)
        #img = cv2.line(img,(int(origin[0]+320), int(origin[1]+240)),(int(p3[0]+320), int(p3[1]+240)), (0,0,255), 3)
    return img


(K, dist) = load_cam(camera_parameter)
video = cv2.VideoWriter(video_name, 0, 5, (2*width,height))

for i in range(0,2949):#(len)
    I = str(i).zfill(4)
    GT = io.loadmat(GT_path_i + I + '.mat')
    P = GT['poses']

    mask = cv2.imread(Mask_path + 'masked_' + I + '.png', cv2.IMREAD_GRAYSCALE)
    img_masked = cv2.imread(Image_masked_path + 'imgmasked_' + I + '.png')
    img = cv2.imread(img_path + 'images_' + I + '.png')
    for j in range(len(P)):
        p = 1000*P[j]
        R = p[0:4]
        t_vecs = p[4:7]
        r = quaternion_rotation_matrix(R)
        r_vecs = rotationMatrix2EulerAngles(r)
        #print(r_vecs)
        print("\nFrame {0}/2948, Object {1},\n Rotation {2}\n Translation {3}\n Intinsic parameter \n{4}".format(i,j,r_vecs, t_vecs, K))
        img = draw_axis(img,r,t_vecs,K)
        #img = img.astype(uint8)

    bboxes = mask2bbox(mask)
    for bbox in bboxes:
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
    img = np.concatenate([img, add_mask(mask)], axis=1)

    video.write(img) 


cv2.destroyAllWindows()
video.release()
