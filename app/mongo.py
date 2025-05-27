# importing Mongoclient from pymongo
from pymongo import MongoClient 
import pandas as pd
import json

myclient = MongoClient("mongodb+srv://<usuario>:<senha>@cluster0.x0dra.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

def save_data(item, data):
    info = {
        "_id":                item['id'], 
        "country_name":       item['country.name'], 
        "country_id":         item['country.id'], 
        "region_id":          item['region.id'], 
        "image_rel_path":     item['imageRelPath'], 
        "topics":             json.loads(item['topics']), 
        "place":              item['place'], 
        "income":             round(item['income'],3), 
        "imagenet_synonyms":  json.loads(item['imagenet_synonyms']), 
        "imagenet_sysnet_id": json.loads(item['imagenet_sysnet_id']),
        "embeddings":         json.loads(item['vector'])
    }

    data.append(info)

if __name__ == "__main__":
    df = pd.read_csv("dollar_street_test.csv")

    data = []
    df.apply(lambda item: save_data(item, data), axis=1)

    # database 
    db = myclient["arqsgbd"]

    #db.img_vector.drop()

    # Created or Switched to collection
    collection = db["img_vector"]

    # Inserting the entire list in the collection
    #collection.insert_many(data)

    vector = data[0]['embeddings']

    print("https://vercel-image-hosting.vercel.app/"+data[0]['image_rel_path'])

    query = [{
        "$vectorSearch": {
            "exact": False,
            "index": "vector_index",
            "limit": 5,
            "numCandidates": 500,
            "path": "embeddings",
            "queryVector": vector
        }
    }]

    try:
        cursor = db.img_vector.aggregate(query)

        for img in cursor:
            print("https://vercel-image-hosting.vercel.app/"+img['image_rel_path'])

    except Exception as e:
        print(e)

    