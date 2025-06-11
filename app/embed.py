import os
import requests
from dotenv import load_dotenv
from encoder import Encoder
from io import BytesIO

load_dotenv()

encoder = Encoder()

def getEmbeddings(imgPath):
    base = os.getenv('IMAGE_HOST')
    response = requests.get(base+imgPath)
    img_path = BytesIO(response.content)
    
    vector = encoder.encode(img_path)

    print("Embedding completed!")

    return vector

def getEmbeddings_clip(imgPath):
    base = os.getenv('IMAGE_HOST')
    response = requests.get(base+imgPath)
    img_path = BytesIO(response.content)
    
    vector = encoder.encode_clip(img_path)

    print("Embedding clip completed!")

    return vector
