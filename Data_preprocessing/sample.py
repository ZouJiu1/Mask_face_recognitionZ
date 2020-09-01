import os
import numpy as np

pwd = os.path.abspath('../')
face_path = os.path.join(pwd, f'Datasets{os.sep}vggface2_train_250_face')
mask_path = os.path.join(pwd, f'Datasets{os.sep}vggface2_train_250_mask')

dir = os.listdir(face_path)
x = np.random.choice(dir, 30)

sample_dir =  os.path.join(pwd, f'Datasets{os.sep}sample')
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

nface = os.path.join(pwd, f'Datasets{os.sep}sample{os.sep}vggface2_train_250_face')
nmask = os.path.join(pwd, f'Datasets{os.sep}sample{os.sep}vggface2_train_250_mask')
if not os.path.exists(nface):
    os.mkdir(nface)
if not os.path.exists(nmask):
    os.mkdir(nmask)

for i in x:
    os.system('mv %s %s'%(os.path.join(face_path, i), nface))
    os.system('mv %s %s'%(os.path.join(mask_path, i), nmask))
