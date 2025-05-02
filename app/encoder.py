from img2vec_pytorch import Img2Vec
from PIL import Image

class Encoder(object):
    def __init__(self, model = 'vgg'):
        self.img2vec_model = Img2Vec(model=model, cuda=False)

    def encode(self, image_path):
        img = Image.open(image_path).convert('RGB')
        vector = self.img2vec_model.get_vec(img, tensor=False)

        return vector

if __name__ == '__main__':
    pass