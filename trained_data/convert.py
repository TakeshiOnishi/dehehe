from keras.models import load_model
import tensorflowjs as tfjs

# TensorFlow.js用にConvert
model = load_model('./fer2013_mini_XCEPTION.110-0.65.hdf5', compile=False)
tfjs.converters.save_keras_model(model, './output/')
