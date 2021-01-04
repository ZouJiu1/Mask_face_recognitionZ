# 配置文件
config = dict()



# 测试路径（不戴口罩）
# config['train_data_path'] = 'datasets_test/test_train'
# config['train_data_index'] = 'datasets_test/test_train.csv'
# config['train_triplets_path'] = 'datasets/test_train.npy'
# config['LFW_data_path'] = 'datasets_test/lfw_funneled'
# config['LFW_pairs'] = 'datasets_test/LFW_pairs.txt'

config['resume_path'] = 'Model_training_checkpoints/model_resnet34_cheahom_triplet_epoch_20_roc0.9337.pt'

config['model'] = 34 # 18 34 50 101 152
config['optimizer'] = 'adagrad'      # sgd\adagrad\rmsprop\adam
config['predicter_path'] = 'Data_preprocessing/shape_predictor_68_face_landmarks.dat'

config['Learning_rate'] = 0.00001
config['image_size'] = 256        # inceptionresnetv2————299
config['epochs'] = 190          #验证集的AUC达到最大时就可以停止训练了不要过拟合

config['train_batch_size'] = 30#136
config['test_batch_size'] = 30

config['margin'] = 0.5
config['embedding_dim'] = 128
config['pretrained'] = False
config['save_last_model'] = True
config['num_train_triplets'] = 1000       #git clone代码里面的图片数量少所以三元组数量少，下载全部图片数据以后，需要设置为100000
config['num_workers'] = 6


config['train_data_path'] = 'Datasets/vggface2_train_face_notmask'
config['mask_data_path'] = 'Datasets/vggface2_train_mask_notmask'
config['train_data_index'] = 'Datasets/vggface2_train_face_notmask.csv'
config['train_triplets_path'] = 'Datasets/training_triplets_' + str(config['num_train_triplets']) + 'notmask.npy'
config['test_pairs_paths'] = 'Datasets/test_pairs.npy'
config['LFW_data_path'] = 'Datasets/lfw_funneled'
config['LFW_pairs'] = 'Datasets/LFW_pairs.txt'
