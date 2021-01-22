from PIL import Image
import cv2
import os
import json
import numpy as np
import pickle
from config_ver1 import config, NUM, IMG_SIZE, DATA_SEG_DIR, DATA_JOINTS_DIR, MAP_SEG, MAP_JOINTS

data_dict = {}

def load_images_from_folder(folder):
    image_ind = 0
    f = open(MAP_SEG, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        img = cv2.imread(os.path.join(folder, line))
        encode_image_to_dict(img, image_ind)
        image_ind+=1
        

def encode_image_to_dict(image, ind):
    im = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    data_dict['image_' + ind] = [im]

def load_joints_from_folder(folder):
    joint_ind = 0
    f = open(MAP_JOINTS, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        joints = json.load(os.path.join(folder, line))
        data = np.asarray(data['people'][0]['pose_keypoints_2d'])
        data = data.reshape(25,3)
        encode_joints_to_dict(data, joint_ind)
        joint_ind+=1

def encode_joints_to_dict(joints, ind):
    data_dict['J_2d_' + ind] = [joints]

def getVertexLabel():
    TEMPLATE = pickle.load(
        open('assets/allTemplate_withBoundaries_symm.pkl', 'rb'), encoding="latin1")
    data_dict["vertexlabel"]=np.zeros((1,27554,1))   #batch_size x vertices x 1
    data_dict["vertexlabel"][np.where(TEMPLATE['Pants'][1])[0]] = 1
    data_dict["vertexlabel"][np.where(TEMPLATE['TShirtNoCoat'][1])[0]] = 5

def savePickleFile():
    outfile = open('assets/inp_data.pkl', 'wb')
    pickle.dump(data_dict, outfile)
    outfile.close()

if __name__ == '__main__':
    load_images_from_folder(DATA_SEG_DIR)
    load_joints_from_folder(DATA_JOINTS_DIR)
    getVertexLabel()
    savePickleFile()
