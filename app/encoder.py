from img2vec_pytorch import Img2Vec
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os
from io import BytesIO


class Encoder(object):
    def __init__(self, model = 'vgg'):
        self.img2vec_model = Img2Vec(model=model, cuda=False)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        self.model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        print(self.device)

    def encode(self, image_path):
        img = Image.open(image_path).convert('RGB')
        vector = self.img2vec_model.get_vec(img, tensor=False)

        return vector.tolist()
    
    def encode_clip(self, image_path):
        image = Image.open(image_path).convert("RGB")  # garante que está em RGB

        print("image path: ", image_path, self.device)

        if isinstance(image_path, (str, os.PathLike)):
            image = Image.open(image_path).convert("RGB")
        else:
        # Caso seja um objeto BytesIO
            image = Image.open(image_path).convert("RGB")

        # Processa a imagem
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Desativa o cálculo de gradientes
        with torch.no_grad():
            outputs = self.model_clip.get_image_features(**inputs)

        return outputs.squeeze().tolist()

if __name__ == '__main__':
    pass