# -*- coding: utf-8 -*-
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

tf.__version__

''' CREATE ALL NECESSARY FILES FOR TENSORBOARD TSNE'''

PATH = os.getcwd() #+ '/analysis/Tensorboard'
FOLDER = '/embedding-everything' # CHANGE HERE!!!
LOG_DIR = PATH + FOLDER
# metadata = os.path.join(LOG_DIR, 'metadata2.tsv')
filename = 'selected_traj' # CHANGE HERE!!! This is a file you WILL create!
# %%
# data_path = PATH + '/data'
# data_dir_list = os.listdir(data_path)

# Load and create the data
pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/selected_drop_traj_everything.tj','rb') # CHANGE HERE!!!
obs = pickle.load(pickle_in)
dims = (len(obs),256)
fc = np.zeros(dims)
for x in range(0,len(obs)):
    fc[x] = obs[x]['fc']

# img_data = []
# for dataset in data_dir_list:
#     img_list = os.listdir(data_path + '/' + dataset)
#     print('Loaded the images of dataset-' + '{}\n'.format(dataset))
#     for img in img_list:
#         input_img = cv2.imread(data_path + '/' + dataset + '/' + img)
#         input_img_resize = cv2.resize(input_img, (224, 224))
#         img_data.append(input_img_resize)
#
# img_data = np.array(img_data)

# %%

# feature_vectors = np.loadtxt('feature_vectors_400_samples.txt')
# print("feature_vectors_shape:", feature_vectors.shape)
# print("num of images:", feature_vectors.shape[0])
# print("size of individual feature vector:", feature_vectors.shape[1])

# num_of_samples = feature_vectors.shape[0]
# num_of_samples_each_class = 100

features = tf.Variable(fc, name='features')

metadata_file = open(os.path.join(LOG_DIR, 'selected_traj.tsv'), 'w')
metadata_file.write('episode\ttimestep\ttarget\tactions\tvalues\taction_label\n')


img_data = []
for i in range(len(obs)):
    img_data.append(obs[i]['images']['alt_view'][0])
    epis = obs[i]['episode']
    tstep = obs[i]['timestep']
    target = obs[i]['target']
    actions = obs[i]['actions']
    values = obs[i]['values'] # FIX DECIMALS ITS UGLY!
    action_label = obs[i]['action_label']
    metadata_file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(epis, tstep, target, actions, round(values,3), action_label)) # round doesnt work in writing to a file
metadata_file.close()
img_data = np.array(img_data)


# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
    # Inverting the colors seems to look better for MNIST
    # data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


# %%
sprite = images_to_sprite(img_data)
cv2.imwrite(os.path.join(LOG_DIR, 'sprite.png'), sprite)
# scipy.misc.imsave(os.path.join(LOG_DIR, 'sprite.png'), sprite)

# %%
with tf.Session() as sess: #(MINE) with the ckpt you take the data for TSNE, tsv file has the metadata and finally you create the pbtxt file
    saver = tf.train.Saver([features])

    sess.run(features.initializer)
    saver.save(sess, os.path.join(LOG_DIR, filename + '.ckpt')) # save data to be analyzed

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = features.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, filename + '.tsv') # HERE TAKE OUT THE LOG_DIR no needed
    # Comment out if you don't want sprites
    embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite.png') # HERE TAKE OUT THE LOG_DIR no needed
    embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)