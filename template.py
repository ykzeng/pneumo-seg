from model import Model as PneumoModel
from keras import Input
from keras.layers import Conv2D
from keras.models import Model
from keras.layers import concatenate
from keras.layers import MaxPooling2D
from keras.layers import Conv2DTranspose
from utils import dice_coef


class dummy(PneumoModel):
    def __init__(self):
        self.nn_setup()
        pass

    def nn_setup(self, im_chan=1):
        # TODO setting up the neural networks in this function
        self.model = None
        pass

    def train(self, x, y, modelPath=False):
        # TODO fit your model with data, persist model somewhere if needed
        self.model.fit()
        self.model.save()
        pass

    def predict(self, x):
        # TODO predict for new image pixel matrix based on your trained model
        return self.model.predict(x)
