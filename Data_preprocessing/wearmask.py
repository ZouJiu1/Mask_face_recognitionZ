import os
import sys
import argparse
import numpy as np
import cv2

import math
import dlib
from PIL import Image, ImageFile

__version__ = '0.3.0'

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mask')
# IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
DEFAULT_IMAGE_PATH = os.path.join(IMAGE_DIR, 'default-mask.png')
BLACK_IMAGE_PATH = os.path.join(IMAGE_DIR, 'black-mask.png')
BLUE_IMAGE_PATH = os.path.join(IMAGE_DIR, 'blue-mask.png')
RED_IMAGE_PATH = os.path.join(IMAGE_DIR, 'red-mask.png')


def rect_to_bbox(rect):
    """获得人脸矩形的坐标信息"""
    # print(rect)
    x = rect[3]
    y = rect[0]
    w = rect[1] - x
    h = rect[2] - y
    return (x, y, w, h)


def face_alignment(faces):
    # 预测关键点
    predictor = dlib.shape_predictor(r"C:\Users\EDZ\Desktop\projects\masked_face\Mask-face-recognition\Data_preprocessing\shape_predictor_68_face_landmarks.dat")
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        shape = predictor(np.uint8(face), rec)
        # left eye, right eye, nose, left mouth, right mouth
        order = [36, 45, 30, 48, 54]
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
        # 计算两眼的中心坐标
        eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2, (shape.part(36).y + shape.part(45).y) * 1. / 2)
        dx = (shape.part(45).x - shape.part(36).x)
        dy = (shape.part(45).y - shape.part(36).y)
        # 计算角度
        angle = math.atan2(dy, dx) * 180. / math.pi
        # 计算仿射矩阵
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
        # 进行仿射变换，即旋转
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))
        faces_aligned.append(RotImg)
    return faces_aligned


def cli(pic_path = '/Users/wuhao/lab/wear-a-mask/spider/new_lfw/Aaron_Tippin/Aaron_Tippin_0001.jpg',save_pic_path = ''):
    parser = argparse.ArgumentParser(description='Wear a face mask in the given picture.')
    # parser.add_argument('pic_path', default='/Users/wuhao/lab/wear-a-mask/spider/new_lfw/Aaron_Tippin/Aaron_Tippin_0001.jpg',help='Picture path.')
    # parser.add_argument('--show', action='store_true', help='Whether show picture with mask or not.')
    parser.add_argument('--model', default='hog', choices=['hog', 'cnn'], help='Which face detection model to use.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--black', action='store_true', help='Wear black mask')
    group.add_argument('--blue', action='store_true', help='Wear blue mask')
    group.add_argument('--red', action='store_true', help='Wear red mask')
    args = parser.parse_args()

    if not os.path.exists(pic_path):
        print(f'Picture {pic_path} not exists.')
        sys.exit(1)

    # if args.black:
    #     mask_path = BLACK_IMAGE_PATH
    # elif args.blue:
    #     mask_path = BLUE_IMAGE_PATH
    # elif args.red:
    #     mask_path = RED_IMAGE_PATH
    # else:
    #     mask_path = DEFAULT_IMAGE_PATH
    mask_path = np.random.choice([BLUE_IMAGE_PATH,BLACK_IMAGE_PATH,RED_IMAGE_PATH,DEFAULT_IMAGE_PATH], 1)[0]

    FaceMasker(pic_path, mask_path, True, 'hog',save_pic_path).mask()


class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, mask_path, show=False, model='hog',save_path = ''):
        self.face_path = face_path
        self.mask_path = mask_path
        self.save_path = save_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None
        self.face_wearmask_path = self.save_path.replace('vggface2_train_250_wearmask',
                                                         'vggface2_train_250_wearmask_face')
        self.mask_wearmask_path = self.save_path.replace('vggface2_train_250_wearmask',
                                                         'vggface2_train_250_wearmask_mask')
        facep = os.sep.join(self.face_wearmask_path.split(os.sep)[:-1])
        maskp = os.sep.join(self.mask_wearmask_path.split(os.sep)[:-1])
        if not os.path.exists(facep):
            os.makedirs(facep)
        if not os.path.exists(maskp):
            os.makedirs(maskp)

    def mask(self):
        import face_recognition

        face_image_np = face_recognition.load_image_file(self.face_path)
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
        face_image_np = cv2.cvtColor(cv2.imread(self.face_path), cv2.COLOR_BGR2RGB)

        predictor = dlib.shape_predictor(
            r"C:\Users\EDZ\Desktop\projects\masked_face\Mask-face-recognition\Data_preprocessing\shape_predictor_68_face_landmarks.dat")
        detector = dlib.get_frontal_face_detector()
        image = dlib.load_rgb_image(self.face_path)
        face_img, mask_img = None, None
        # 人脸对齐、切图
        dets = detector(image, 1)
        img_size = 200
        if len(dets) == 1:
            faces = dlib.full_object_detections()
            faces.append(predictor(image, dets[0]))
            images = dlib.get_face_chips(image, faces, size=img_size)

            image = np.array(images[0]).astype(np.uint8)
            face_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 生成人脸mask
            dets = detector(image, 1)
            if len(dets) == 1:
                point68 = predictor(image, dets[0])
                landmarks = list()
                INDEX = [0, 2, 14, 16, 17, 18, 19, 24, 25, 26]
                eyebrow_list = [19, 24]
                eyes_list = [36, 45]
                eyebrow = 0
                eyes = 0

                for eb, ey in zip(eyebrow_list, eyes_list):
                    eyebrow += point68.part(eb).y
                    eyes += point68.part(ey).y
                add_pixel = int(eyes / 2 - eyebrow / 2)

                for idx in INDEX:
                    x = point68.part(idx).x
                    if idx in eyebrow_list:
                        y = (point68.part(idx).y - 2 * add_pixel) if (point68.part(idx).y - 2 * add_pixel) > 0 else 0
                    else:
                        y = point68.part(idx).y
                    landmarks.append((x, y))

                landmarks = np.array(landmarks)
                hull = cv2.convexHull(landmarks)
                mask = np.zeros(face_img.shape, dtype=np.uint8)
                mask_img = cv2.fillPoly(mask, [hull], (255, 255, 255))

        self._face_img = Image.fromarray(face_image_np)
        self._mask_img = Image.open(self.mask_path)

        found_face = False
        for face_landmark in face_landmarks:
            # check whether facial features meet requirement
            skip = False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break
            if skip:
                continue

            # mask face
            found_face = True
            self._mask_face(face_landmark)

        if (found_face) and (np.max(face_img) is not None and np.max(mask_img) is not None):
            # cv2.imwrite(self.mask_wearmask_path, mask_img)
            # cv2.imwrite(self.face_wearmask_path, face_img)
            # align
            src_faces = []
            src_face_num = 0
            with_mask_face = np.asarray(self._face_img)
            for (i, rect) in enumerate(face_locations):
                src_face_num = src_face_num + 1
                (x, y, w, h) = rect_to_bbox(rect)
                detect_face = with_mask_face[y:y + h, x:x + w]
                src_faces.append(detect_face)
            # 人脸对齐操作并保存
            faces_aligned = face_alignment(src_faces)
            face_num = 0
            for faces in faces_aligned:
                face_num = face_num + 1
                faces = cv2.cvtColor(faces, cv2.COLOR_RGBA2BGR)
                size = (int(200), int(200))
                faces_after_resize = cv2.resize(faces, size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(self.face_wearmask_path, faces_after_resize)
            # if self.show:
            #     self._face_img.show()
            # save
            self._save()
        else:
            # 在这里记录没有裁的图片
            print('Found no face.'+self.save_path)

    def _mask_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)

    def _save(self):
        path_splits = os.path.splitext(self.save_path)
        new_face_path = path_splits[0] + '-with-mask' + path_splits[1]
        self._face_img.save(new_face_path)
        print(f'Save to {new_face_path}')

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':
    dataset_path = r'C:\Users\EDZ\Desktop\projects\masked_face\Mask-face-recognition\Datasets\vggface2_train_250'
    save_dataset_path = r'C:\Users\EDZ\Desktop\projects\masked_face\Mask-face-recognition\Datasets\vggface2_train_250_wearmask'
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            new_root = root.replace(dataset_path, save_dataset_path)
            # if not os.path.exists(new_root):
            #     os.makedirs(new_root)
            # deal
            imgpath = os.path.join(root, name)
            save_imgpath = os.path.join(new_root, name)
            ndir = os.sep.join(save_imgpath.split(os.sep)[:-1])
            if not os.path.exists(ndir):
                os.mkdir(ndir)
            cli(imgpath, save_imgpath)