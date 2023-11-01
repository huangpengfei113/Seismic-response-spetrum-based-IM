from keras.models import load_model
from functions import *
import os


if __name__ == '__main__':
    path_model0 = r'file of trained ANN model'
    # ['col1', 'col2'], ['acc', 'vel', 'disp']
    h5_name = 'MLP-{}-{}.h5'.format('col1', 'acc')
    path_model = os.path.join(path_model0, h5_name)
    temp_model = load_model(path_model, custom_objects={'return_log': return_log})
    # the name of input layer, hidden layer and output layer are spec_inp, spec_mid, and spec_out, respectively
    layer_temp = temp_model.get_layer('spec_out')
    temp_weight = layer_temp.get_weights()
    print(temp_weight)
