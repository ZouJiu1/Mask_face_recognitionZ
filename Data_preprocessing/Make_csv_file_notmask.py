'''
用于为生成csv格式的数据清单(训练集）。
输入：训练集文件夹路径
输出：csv文件路径
功能：讲训练集的文件整理成列表，包含id/name/class三项信息。
# id是图片文件名（不包含文件后缀）
# name是人名
# class和人名一一对应，是int的数字，方便传入模型用的
'''
# 路径置顶
import sys
import os
sys.path.append(os.getcwd())
# 导入包
from tqdm import tqdm
import csv 

pwd = os.path.abspath('../')

# 定义路径
data_path = os.path.join(pwd, f'Datasets{os.sep}vggface2_train_face_notmask')
csv_path = os.path.join(pwd, f'Datasets{os.sep}vggface2_train_face_notmask.csv')

# 打开一个csv文件
f = open(csv_path, 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(f)
# 写入表头
csv_writer.writerow(["id","name","class"])

files_list = os.listdir(data_path)
files_list.sort()
# 文件循环
for index, man in tqdm(enumerate(files_list)):
    pic_list = os.listdir(os.path.join(data_path, man))
    pic_list.sort()
    for pic in pic_list:
        picid = pic.split('.')
        csv_writer.writerow([picid[0],man,index])

f.close()
