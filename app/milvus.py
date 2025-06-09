import pandas as pd
import json
from pymilvus import MilvusClient, DataType

client = MilvusClient("default.db")

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
        nullable=True,
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
        nullable=True
    )
    schema.add_field(
        field_name="imagenet_sysnet_id",
        datatype=DataType.ARRAY,
        element_type=DataType.INT8,
        max_capacity=10,
        nullable=True
    )

    return schema

if __name__ == "__main__":


    df = pd.read_csv("dollar_street_test.csv")

    vector = json.loads(df['vector_clip'][0])
    print(len(vector))

    # data = []
    # df.apply(lambda item: save_data(item, data), axis=1)

    # schema = createSchema()

    # if client.has_collection(collection_name="ImageData"):
    #     client.drop_collection(collection_name="ImageData")
    # client.create_collection(collection_name="ImageData", schema=schema, dimension=768, enable_dynamic_field=True)

    # try:
    #     res = client.insert(collection_name="ImageData", data=data)
    # except Exception as e:
    #     print(f"Error inserting data: {e}")
    #     res = None
    index_params = MilvusClient.prepare_index_params()

    # 4.2. Add an index on the vector field.
    index_params.add_index(
        field_name="vector_768",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        index_name="vector_index",
        params={ "nlist": 128 }
    )

    # 4.3. Create an index file
    client.create_index(
        collection_name="ImageData",
        index_params=index_params,
        sync=False # Whether to wait for index creation to complete before returning. Defaults to True.
    )

    res = client.search(collection_name="ImageData",
                        data=[vector],
                        limit=10,
                        anns_field="vector_768",
                        output_fields=["image_rel_path"],
                        search_params={
                            "metric_type": "COSINE",
                            "params": {"nprobe": 10}
                        })
    print(res)




    