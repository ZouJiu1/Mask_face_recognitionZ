# 路径置顶
import sys 
import os 
sys.path.append(os.getcwd()) 
import cv2
import numpy as np 
from tqdm import tqdm 
import dlib 

# def preprocess(image_path, detector, predictor, img_size):
#     image = dlib.load_rgb_image(image_path)
#     save = False
#     dets = detector(image, 1)
    
#     if len(dets) == 1:
#         faces = dlib.full_object_detections()
#         faces.append(predictor(image, dets[0]))
#         images = dlib.get_face_chips(image, faces, size=img_size)
#         image = np.array(images[0]).astype(np.uint8)
#         # 生成人脸mask
#         dets = detector(image, 1)

#         if len(dets) == 1:
#             save = True

#     if save == False:
#         os.remove(image_path)
#         return 0
#     return 1

def preprocess(image_path):

    img = cv2.imread(image_path)
    w, h, c = img.shape
    if w < 250 or h < 250:
        os.remove(image_path)
        return 0
    return 1


# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# img_size = 200


data_path = 'vggface2_train_250'
files_list = os.listdir(data_path)
# 文件循环
kill_img = 0
not_kill = 0
for man in tqdm(files_list):
    pic_list = os.listdir(os.path.join(data_path, man))
    for pic in pic_list:
        img_path = os.path.join(data_path, man, pic)
        a = preprocess(img_path)
        if a == 0:
            kill_img += 1
        elif a == 1:
            not_kill += 1
print('kill:', kill_img)
print('not_kill:', not_kill)
print('all:', kill_img+not_kill)
