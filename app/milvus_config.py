import pandas as pd
import json
from pymilvus import MilvusClient, DataType
from embed import getEmbeddings, getEmbeddings_clip
from datetime import datetime

def save_data(item, data):
    info = {
        "id":                item['id'], 
        "country_name":       item['country.name'], 
        "country_id":         item['country.id'], 
        "region_id":          item['region.id'], 
        "image_rel_path":     item['imageRelPath'], 
        "topics":             json.loads(item['topics']), 
        "place":              item['place'], 
        "income":             round(item['income'],3), 
        "imagenet_synonyms":  json.loads(item['imagenet_synonyms']), 
        "imagenet_sysnet_id": json.loads(item['imagenet_sysnet_id']),
        "vector_768":         json.loads(item['vector_clip']),
        "vector_4096":        json.loads(item['vector'])
    }

    data.append(info)

def createSchema():

    schema = MilvusClient.create_schema()

    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        auto_id=False,
        max_length=24,
    )
    schema.add_field(
        field_name="vector_768",
        datatype=DataType.FLOAT_VECTOR,
        dim=768,
    )
    schema.add_field(
        field_name="vector_4096",
        datatype=DataType.FLOAT_VECTOR,
        dim=4096,
    )
    schema.add_field(
        field_name="country_name",
        datatype=DataType.VARCHAR,
        max_length=30,
    )
    schema.add_field(
        field_name="country_id",
        datatype=DataType.VARCHAR,
        max_length=2,
    )
    schema.add_field(
        field_name="region_id",
        datatype=DataType.VARCHAR,
        max_length=2,
    )
    schema.add_field(
        field_name="image_rel_path",
        datatype=DataType.VARCHAR,
        max_length=55,
    )
    schema.add_field(field_name="topics",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=10, max_length=50,
    )
    schema.add_field(
        field_name="place",
        datatype=DataType.VARCHAR,
        max_length=50,
    )
    schema.add_field(
        field_name="income",
        datatype=DataType.FLOAT,
    )
    schema.add_field(
        field_name="imagenet_synonyms",
        datatype=DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_capacity=10, max_length=50,
    )
    schema.add_field(
        field_name="imagenet_sysnet_id",
        datatype=DataType.ARRAY,
        element_type=DataType.INT8,
        max_capacity=10,
    )

    return schema

if __name__ == "__main__":

    df = pd.read_csv("dollar_street_test.csv")

    if 'vector' not in df.columns:
        df['vector'] = df['imageRelPath'].apply(getEmbeddings)
        df.to_csv("dollar_street_test.csv", index=False)
    print("Embeddings normal completed!")

    if 'vector_clip' not in df.columns:
        df['vector_clip'] = df['imageRelPath'].apply(getEmbeddings_clip)
        df.to_csv("dollar_street_test.csv", index=False)
    print("Embeddings clip completed!")

    client = MilvusClient("default.db")

    data = []
    df.apply(lambda item: save_data(item, data), axis=1)

    schema = createSchema()

    if client.has_collection(collection_name="ImageData"):
        client.drop_collection(collection_name="ImageData")
    client.create_collection(collection_name="ImageData", 
                             schema=schema, 
                             dimension=768, 
                             enable_dynamic_field=True)

    try:
        start = datetime.now()
        res = client.insert(collection_name="ImageData", data=data)
        end = datetime.now()
            
        print("Tempo de Execução:", end-start)
        
    except Exception as e:
        print(f"Error inserting data: {e}")
        res = None