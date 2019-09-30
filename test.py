import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import argparse
import shutil
tf.logging.set_verbosity(tf.logging.ERROR)

#Parse input parameters
parser = argparse.ArgumentParser(description='Fashion MNIST Keras Model')
parser.add_argument('--modelPath', type=str, dest='MODEL_DIR', help='location to store the model artifacts')
parser.add_argument('--version', type=str, dest='VERSION', default="1", help='model version')
args = parser.parse_args()

MODEL_DIR = args.MODEL_DIR
VERSION = args.VERSION


shutil.rmtree(MODEL_DIR)

#Save model
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    export_path = os.path.join(MODEL_DIR, VERSION)
    print('export_path = {}\n'.format(export_path))

    tf.saved_model.simple_save(
        keras.backend.get_session(),
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name:t for t in model.outputs})

    print('\nModel saved to ' + MODEL_DIR)
else:
    print('\nExisting model found at ' + MODEL_DIR)
    print('\nDid not overwrite old model. Run the job again with a different location to store the model')