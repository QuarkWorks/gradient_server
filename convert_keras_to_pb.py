import tensorflow as tf

import keras

#from keras.layers import ReLU
from keras.layers import DepthwiseConv2D
from keras.utils import CustomObjectScope
from keras.applications.mobilenet_v2 import MobileNetV2

import argparse
import shutil
import os
from os.path import expanduser, join

def relu6(x):
    return keras.relu(x, max_value=6)

tf.logging.set_verbosity(tf.logging.ERROR)

#Parse input parameters
parser = argparse.ArgumentParser(description='Fashion MNIST Keras Model')
parser.add_argument('--modelPath', type=str, dest='MODEL_DIR', help='location to store the model artifacts')
parser.add_argument('--version', type=str, dest='VERSION', default="1", help='model version')
args = parser.parse_args()

MODEL_DIR = args.MODEL_DIR
VERSION = args.VERSION

tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.VERSION)
print(keras.__version__)

with CustomObjectScope(
        {'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D}):
    nsfw_model = keras.models.load_model('./models/nsfw_mobilenet2.224x224.h5')

#Save model
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    export_path = os.path.join(MODEL_DIR, VERSION)
    print('export_path = {}\n'.format(export_path))

    tf.saved_model.simple_save(
        keras.backend.get_session(),
        export_path,
        inputs={'input_image': nsfw_model.input},
        outputs={t.name:t for t in nsfw_model.outputs})

    print('\nModel saved to ' + MODEL_DIR)
else:
    print('\nExisting model found at ' + MODEL_DIR)
    print('\nDid not overwrite old model. Run the job again with a different location to store the model')





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