#https://williamjshipman.wordpress.com/2019/06/23/saving-and-loading-tensorflow-neural-networks-part-1-everythings-deprecated-what-now/
#https://www.tensorflow.org/guide/keras/save_and_serialize

import tensorflow as tf


#from keras_applications.mobilenet import relu6
# from keras.layers import DepthwiseConv2D
# from keras.utils import CustomObjectScope
#from keras.applications.mobilenet_v2 import MobileNetV



import argparse
import shutil
import os
from os.path import expanduser, join



#tf.logging.set_verbosity(tf.logging.ERROR)

#Parse input parameters
parser = argparse.ArgumentParser(description='Fashion MNIST Keras Model')
parser.add_argument('--modelPath', type=str, dest='MODEL_DIR', default="./export", help='location to store the model artifacts')
parser.add_argument('--version', type=str, dest='VERSION', default="1", help='model version')
args = parser.parse_args()

MODEL_DIR = args.MODEL_DIR
VERSION = args.VERSION

#tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.version)
print(tf.keras.__version__)



tf.keras.layers.DepthwiseConv2D

with tf.keras.utils.CustomObjectScope(
        {'relu6': tf.compat.v2.keras.layers.ReLU, 'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D}):
    nsfw_model = tf.keras.models.load_model('./models/nsfw_mobilenet2.224x224.h5')

nsfw_model.summary()


if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    export_path = os.path.join(MODEL_DIR, VERSION)
    print(export_path)
    tf.saved_model.save(nsfw_model, export_path)



# tf.keras.models.save_model(
#     model,
#     filepath,
#     overwrite=True,
#     include_optimizer=True,
#     save_format=None,
#     signatures=None,
#     options=None
# )


# #Save model

#     export_path = os.path.join(MODEL_DIR, VERSION)
#     print('export_path = {}\n'.format(export_path))
#
#     tf.saved_model.save(
#         tf.keras.backend.get_session(),
#         export_path,
#         inputs={'input_image': nsfw_model.input},
#         outputs={t.name:t for t in nsfw_model.outputs})
#
#     print('\nModel saved to ' + MODEL_DIR)
# else:
#     print('\nExisting model found at ' + MODEL_DIR)
#     print('\nDid not overwrite old model. Run the job again with a different location to store the model')
#




# print(os.listdir('./storage'))
#
# print(os.listdir(MODEL_DIR))
# shutil.rmtree('./storage/model3')
#
# shutil.rmtree(MODEL_DIR)
# print(os.listdir('./storage'))

#Save model
# if not os.path.exists(MODEL_DIR):
#     os.makedirs(MODEL_DIR)
#     export_path = os.path.join(MODEL_DIR, VERSION) + '.pb'
#     print('export_path = {}\n'.format(export_path))
#
#     shutil.copy('./storage/quant_nsfw_mobilenet.pb', export_path)
#
#
#     print('\nModel saved to ' + MODEL_DIR)
# else:
#     print('\nExisting model found at ' + MODEL_DIR)
#     print('\nDid not overwrite old model. Run the job again with a different location to store the model')
#
#
# print(os.listdir(MODEL_DIR))



#/storage/quant_nsfw_mobilenet.pb