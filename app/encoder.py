from img2vec_pytorch import Img2Vec
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch


class Encoder(object):
    def __init__(self, model = 'vgg'):
        self.img2vec_model = Img2Vec(model=model, cuda=False)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

    def encode(self, image_path):
        img = Image.open(image_path).convert('RGB')
        vector = self.img2vec_model.get_vec(img, tensor=False)

        return vector.tolist()
    
    def encode_clip(self, image_path):
        image = Image.open(image_path).convert("RGB")  # garante que está em RGB

        # Processa a imagem
        inputs = self.processor(images=image, return_tensors="pt")

        # Desativa o cálculo de gradientes
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        return outputs.squeeze().tolist()

if __name__ == '__main__':
    pass