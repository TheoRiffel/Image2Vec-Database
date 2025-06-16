from flask import Flask, render_template, request, jsonify
import psycopg
from pymilvus import MilvusClient, Collection
import os
from dotenv import load_dotenv
from encoder import Encoder
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.app_context().push()

DB_URL = os.getenv('DB_URL')
IMAGE_HOST = os.getenv('IMAGE_HOST')

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template("index.html")

@app.route('/metadata', methods=['GET'])
def get_metadata():
    print("get metadata")
    country_names = []
    regions = []
    with psycopg.connect(DB_URL) as conn:
        cursor = conn.cursor()
        sql_get_countries = "SELECT DISTINCT(country_name) FROM metadata ORDER BY country_name"
        sql_get_regions = "SELECT DISTINCT(region_id) FROM metadata ORDER BY region_id"
        sql_get_income = "SELECT min(income), max(income) FROM metadata"

        try:
            cursor.execute(sql_get_countries)
            countries = cursor.fetchall()

            country_names = [country_name[0] for country_name in countries]
            
            cursor.execute(sql_get_regions)
            region_ids = cursor.fetchall()

            regions = [region_id[0] for region_id in region_ids]

            cursor.execute(sql_get_income)
            min_income, max_income = cursor.fetchone()
            print(min_income, max_income)
            
        except Exception as e:
            print(e)
            jsonify({"error": "Erro na base de dados!"})
    
    return jsonify({"countries": country_names, "regions": regions, "min_income": min_income, "max_income": max_income})

def build_query_array(table, operator, vector, metadata):
    distances = {
        'l1': ('SUM(ABS(a - b))', 'ASC'), # Manhattan
        'l2': ('SQRT(SUM(POWER(a - b, 2)))', 'ASC'), # Euclidean
        'cosine': ('1 - SUM(a * b) / (sqrt(SUM(a * a)) * sqrt(SUM(b * b)))', 'ASC'), # Cosine
        'ip': ('SUM(a * b)', 'DESC'), # Inner Product
    }

    vector_str = 'ARRAY[' + ', '.join(map(str, vector)) + ']'

    query_start = f'''
                    SELECT arrays.id
                    FROM {table} arrays
                '''
    
    if len(metadata) != 0:
        join_metadata = ' AND '.join(metadata)
        query_start += f'''
                    JOIN metadata ON metadata.id = arrays.id
                    WHERE {join_metadata}
                    '''

    query_end = f'''
                    ORDER BY (
                        SELECT {distances[operator][0]}
                        FROM unnest(embedding) WITH ORDINALITY AS t1(a, i) 
                        JOIN unnest({vector_str}) WITH ORDINALITY AS t2(b, j)
                        ON i = j
                    ) {distances[operator][1]}
                    LIMIT 6;
                '''

    sql_search_images = query_start + query_end

    return sql_search_images

def build_query_vector(table, distance, vector, metadata):
    distances = {
        'l1': '<+>',
        'l2': '<->',
        'cosine': '<=>',
        'ip': '<#>'
    }

    operator = distances[distance]
     
    query_start = f'''
                    SELECT vectors.id,
                           vectors.embedding {operator} '{vector}' 
                    FROM {table} vectors
                '''

    if len(metadata) != 0:
        join_metadata = ' AND '.join(metadata)
        query_start += f'''
                    JOIN metadata ON metadata.id = vectors.id
                    WHERE {join_metadata}
                    '''
        
    query_end = f'''
                    ORDER BY vectors.embedding {operator} '{vector}' 
                    LIMIT 6
                ''' 
                
    sql_search_images = query_start + query_end
    
    return sql_search_images


def get_images_milvus(indexes, vector_index, field, distance, vector, metadata):
    field = "vector_768" if "clip" in field else "vector_4096"

    print(metadata)
    try:
        client = MilvusClient("default.db")
        client.drop_index(collection_name="ImageData", index_name="vector_index")

        index_params = MilvusClient.prepare_index_params()

        if indexes == None or len(vector_index) == 0:
            client.drop_index(collection_name="ImageData", index_name="country_name_idx")
            client.drop_index(collection_name="ImageData", index_name="region_id_idx")
            client.drop_index(collection_name="ImageData", index_name="income_idx")

            index_params.add_index(
                field_name=field,
                metric_type=distance.upper(),
                index_type="FLAT",
                index_name="vector_index",
                params = {}
            )
            
        else:
            index_params.add_index(
                field_name=field,
                metric_type=distance.upper(),
                index_type=vector_index.upper(),
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


        print(index_params)

        client.create_index(
            collection_name="ImageData",
            index_params=index_params,
            sync=False # Whether to wait for index creation to complete before returning. Defaults to True.
        )


        print(client.list_indexes(collection_name="ImageData"))

        start = datetime.now()
        res = client.search(collection_name="ImageData",
                            data=[vector],
                            limit=6,
                            anns_field=field,
                            filter = metadata,
                            output_fields=["country_name","income", "imagenet_synonyms", "image_rel_path"],
                            search_params={
                                "metric_type": distance.upper(),
                                "params": {"nprobe": 100}
                            })
        end = datetime.now()
        query_time = end - start

        images = res[0]
        image_src = IMAGE_HOST

        images_metadata_formatted = [
            {
            "image_path": image_src + image['entity']['image_rel_path'].strip(),
            "image_country": image['entity']['country_name'],
            "image_income": image['entity']['income'],
            "image_synonyms": list(image['entity']['imagenet_synonyms']),
            "image_distance": str(image['distance'])[0:18]
            }
            for image in images 
        ]

        return {"images_metadata": images_metadata_formatted, "query_time": str(query_time)}
    

    except Exception as e:
        print(e)
        return {"error": "erro"}


def get_images_postgresql(sql_search_images, indexes, vector_index, operator):
    with psycopg.connect(DB_URL) as conn:
        cursor = conn.cursor()

        try:
            if indexes is None:
                cursor.execute("BEGIN;")
                cursor.execute("DROP INDEX IF EXISTS country_name_idx;")
                cursor.execute("DROP INDEX IF EXISTS region_id_idx;")
                cursor.execute("DROP INDEX IF EXISTS income_idx;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_cosine;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_l1;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_l2;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_ip;")
                cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_l2;")
                cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_ip;")
                cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_cosine;")

            if vector_index == 'ivf_flat':
                cursor.execute("BEGIN;")

                operators = ['l1', 'l2', 'ip', 'cosine']

                for op in operators:
                    cursor.execute(f"DROP INDEX IF EXISTS hnsw_idx_{op};")
                    if op != operator:
                        cursor.execute(f"DROP INDEX IF EXISTS ivfflat_idx_{op};")
            elif vector_index == 'hnsw':
                cursor.execute("BEGIN;")
                operators = ['l1', 'l2', 'ip', 'cosine']

                for op in operators:
                    cursor.execute(f"DROP INDEX IF EXISTS ivfflat_idx_{op};")
                    if op != operator:
                        cursor.execute(f"DROP INDEX IF EXISTS hsnw_idx_{op};")
            

            start = datetime.now()
            cursor.execute(sql_search_images)
            end = datetime.now()

            print("Tempo de execução: ", end - start)
            query_time = end - start

            images = cursor.fetchall()

            if len(images) == 0:
                return jsonify({"error": "Nenhuma imagem encontrada!"})
            
            images_id = [image[0] for image in images]
            images_dist = [str(image[1]) for image in images] if len(images[0]) > 1 else [''] * len(images)

            images_id_format = tuple(images_id)

            params = ['%s'] * len(images_id)
            params_str = ','.join(params)

            sql_get_images = f'''SELECT metadata.img_rel_path,
                                        metadata.country_name, 
                                        metadata.income, 
                                        metadata.imagenet_synonyms 
                                 FROM metadata 
                                 WHERE id IN ({params_str})'''

            cursor.execute(sql_get_images, images_id_format)
            images_metadata = cursor.fetchall()

            image_src = IMAGE_HOST

            images_metadata_formatted = [
                {
                    "image_path": image_src + image[0].strip(),
                    "image_country": image[1],
                    "image_income": float(image[2]),
                    "image_synonyms": image[3],
                    "image_distance": dist[0:18]
                }
                for image, dist in zip(images_metadata, images_dist)
            ]

            print(images_metadata_formatted)

            cursor.execute("ROLLBACK;")
        
            return {"images_metadata": images_metadata_formatted, "query_time": str(query_time)}

        except Exception as e:
            print(e)
            return {"error": "erro"}
    

def get_query_analysis_postgresql(sql_search_images, indexes, vector_index, operator):
    with psycopg.connect(DB_URL) as conn:
        cursor = conn.cursor()
        
        print(len(vector_index))
        try:
            if indexes is None:
                cursor.execute("BEGIN;")
                cursor.execute("DROP INDEX IF EXISTS country_name_idx;")
                cursor.execute("DROP INDEX IF EXISTS region_id_idx;")
                cursor.execute("DROP INDEX IF EXISTS income_idx;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_cosine;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_l1;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_l2;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_ip;")
                cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_l2;")
                cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_ip;")
                cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_cosine;")

            if vector_index == 'ivf_flat':
                cursor.execute("BEGIN;")

                operators = ['l1', 'l2', 'ip', 'cosine']

                for op in operators:
                    cursor.execute(f"DROP INDEX IF EXISTS hnsw_idx_{op};")
                    if op != operator:
                        cursor.execute(f"DROP INDEX IF EXISTS ivfflat_idx_{op};")
            elif vector_index == 'hnsw':
                cursor.execute("BEGIN;")
                operators = ['l1', 'l2', 'ip', 'cosine']

                for op in operators:
                    cursor.execute(f"DROP INDEX IF EXISTS ivfflat_idx_{op};")
                    if op != operator:
                        cursor.execute(f"DROP INDEX IF EXISTS hsnw_idx_{op};")



            explain_query = "EXPLAIN ANALYZE " + sql_search_images
            cursor.execute(explain_query)

            analysis = cursor.fetchall()

            cursor.execute("ROLLBACK;")

            return {"query_analysis": analysis}
                
        except Exception as e:
            return {"error": "erro"}
           

def format_income(income):
    income = income.replace("$", "")
    income = income.replace(" ", "")
    min_income, max_income = income.split("-")

    return [min_income, max_income]
    

def postgresql(action, indexes, vector_index, table, operator, vector, metadata):
    sql_search_images = build_query_vector(table, operator, vector, metadata) \
                            if 'vector' in table \
                            else build_query_array(table, operator, vector, metadata)

                
    if action == 'get-images':
        return jsonify(get_images_postgresql(sql_search_images, indexes, vector_index, operator))
        
    if action == 'get-analysis':
        return jsonify(get_query_analysis_postgresql(sql_search_images, indexes, vector_index, operator))


def milvus(action, indexes, vector_index, table, operator, vector, filter):
    if action == 'get-images':
        return jsonify(get_images_milvus(indexes, vector_index, table, operator, vector, filter))
        
    if action == 'get-analysis':
        return jsonify({"error":"Nao foi implementado ainda!"})

def comparar(indexes, vector_index, table, operator, vector, metadata, filter):
    pg_table = 'img_pg'+table
    sql_search_images = build_query_vector(pg_table, operator, vector, metadata)

    try:
        res_postgresql = get_images_postgresql(sql_search_images, indexes, vector_index, operator)
        res_milvus = get_images_milvus(indexes, vector_index, table, operator, vector, filter)
        
        res_postgresql = res_postgresql['images_metadata'][:3]
        res_milvus = res_milvus['images_metadata'][:3]

        images_compare = {
            'images_postgresql': res_postgresql,
            'images_milvus': res_milvus
        }

        return jsonify({"images_compare": images_compare})
    
    except Exception as e:
        print(e)
        return jsonify({"error": "Erro!"})


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    database = request.form.get('banco')
    operator = request.form.get('operador')
    country = request.form.get('pais')
    table = request.form.get('tabela')
    region = request.form.get('regiao')
    action = request.form.get('acao')
    income = format_income(request.form.get('income'))
    indexes = request.form.get('use-indexes')
    vector_index = request.form.get('indice')

    print(database)
    
    if file and file.filename != '':
        encoder = Encoder()

        vector = encoder.encode_clip(file) if 'clip' in table else encoder.encode(file)

        metadata = []
        filter = ""
        filter_params = {}
        if country != "":
            metadata.append(f"country_name = '{country}'")
            filter_params["country"] = country
            filter += f"country_name == '{country}'"
            
        if region != "":
            metadata.append(f"region_id = '{region}'")
            filter_params["region"] = region
            if len(filter) != 0:
                filter += " AND "
            filter += f"region_id == '{region}'"


        if income != "":
            metadata.append(f"income BETWEEN {income[0]} AND {income[1]}")
            filter_params["min_income"] = income[0]
            filter_params["max_income"] = income[1]
            if len(filter) != 0:
                filter += " AND "
            filter += f"income <= {income[0]} AND "
            filter += f"income >= {income[1]}"
        

        if database == "postgresql":
            return postgresql(action, indexes, vector_index, table, operator, vector, metadata)
        
        if database == "milvus":
            return milvus(action, indexes, vector_index, table, operator, vector, filter)
        
        if database == "comparar":
            return comparar(indexes, vector_index, table, operator, vector, metadata, filter)

        
        
    return "Nenhuma imagem recebida."
    

if __name__ == "__main__":
    app.run(debug = True)