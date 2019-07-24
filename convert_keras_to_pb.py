import keras
import tensorflow as tf

nsfw_model = keras.models.load_model('./models/nsfw_mobilenet2.224x224.h5')

tf.saved_model.simple_save(
        keras.backend.get_session(),
        './models/nsfw_model',
        inputs={'input_image': nsfw_model.input},
        outputs={t.name:t for t in nsfw_model.outputs})