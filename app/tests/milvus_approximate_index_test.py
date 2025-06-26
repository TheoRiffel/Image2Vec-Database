from datetime import datetime
import numpy as np
import pandas as pd
from pymilvus import MilvusClient

def generate_random_vector(size):
    vec = np.random.uniform(high=1, low=-1, size=size)
    return [float(x) for x in vec]

def generate_random_vector_by_field(field_name):
    if '768' in field_name:
        return generate_random_vector(768)

    return generate_random_vector(4096)

def get_index_params(index_type, field, metric_type):
    
    use_index = (index_type != "FLAT")

    index_params = MilvusClient.prepare_index_params()

    if not use_index:
        index_params.add_index(
            field_name=field,
            metric_type=metric_type,
            index_type="FLAT",
            index_name="vector_index",
            params = {}
        )
    else:
        index_params.add_index(
            field_name=field,
            metric_type=metric_type,
            index_type=index_type,
            index_name="vector_index",
            params = {"n_list": 128}
        )

        index_params.add_index(
            field_name="country_name",
            index_type="INVERTED",
            index_name="country_name_idx"
        )

        index_params.add_index(
            field_name="region_id",
            index_type="INVERTED",
            index_name="region_id_idx"
        )

        index_params.add_index(
            field_name="income",
            index_type="INVERTED",
            index_name="income_idx"
        )
        
    return index_params


def test_approximate_index(vector, field, metric_type, index_type):
    try:
        client = MilvusClient("default.db")

        client.drop_index(collection_name="ImageData", index_name="vector_index")

        if index_type == "FLAT":
            client.drop_index(collection_name="ImageData", index_name="country_name_idx")
            client.drop_index(collection_name="ImageData", index_name="region_id_idx")
            client.drop_index(collection_name="ImageData", index_name="income_idx")

        index_params = get_index_params(index_type, field, metric_type)

        client.create_index(
            collection_name="ImageData",
            index_params=index_params,
            sync=True
        )

        res = client.search(collection_name="ImageData",
                            data=[vector],
                            limit=6,
                            anns_field=field,
                            output_fields=["country_name","income", "imagenet_synonyms", "image_rel_path"],
                            search_params={
                                "metric_type": metric_type,
                                "params": {"nprobe": 100}
                            })

        client.close()

        return res[0]
    except Exception as e:
        print(e)


fields = ["vector_768", "vector_4096"]
metric_types = ["COSINE", "L2", "IP"]
index_types = ["FLAT", "IVF_FLAT"]

rows = []
test_amount = 15

for field in fields:
    for metric in metric_types:
        for test in range(test_amount):
            vector = generate_random_vector_by_field(field)
            no_index = test_approximate_index(vector, field, metric, index_types[0])
            index = test_approximate_index(vector, field, metric, index_types[1])

            for i in range(min(len(index), len(no_index))):
              rows.append([test, field, metric, index[i].distance, no_index[i].distance])

            print(f"Testing field '{field}' with metric '{metric}' ({test+1}/{test_amount})")
            
df = pd.DataFrame(rows, columns=['test_num', 'table', 'operator', 'index_similarity', 'no_index_similarity'])
df.to_csv('./tests/milvus_approximate_index_test.csv')