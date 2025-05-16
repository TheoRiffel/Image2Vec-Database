from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Carrega modelo e processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

# Função para gerar embedding de uma imagem
def generate_image_embedding(image_path: str):
    image = Image.open(image_path).convert("RGB")  # garante que está em RGB

    # Processa a imagem
    inputs = processor(images=image, return_tensors="pt")

    # Desativa o cálculo de gradientes
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Normaliza o vetor de embedding
    embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

    return embedding.squeeze().tolist()

# Execução direta via terminal
if __name__ == "__main__":

    image_path = '5d4bde2acf0b3a0f3f335ad7.JPG'
    embedding = generate_image_embedding(image_path)
    print("Embedding gerado com sucesso! Vetor com", len(embedding), "dimensões.")
    print(embedding)
