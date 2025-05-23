from flask import Flask, render_template, request, jsonify
import psycopg
import os
from dotenv import load_dotenv
from encoder import Encoder
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.app_context().push()

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template("index.html")

@app.route('/metadata', methods=['GET'])
def get_metadata():
    print("get metadata")
    country_names = []
    regions = []
    with psycopg.connect(os.getenv('DB_URL')) as conn:
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
        '<+>': ('SUM(ABS(a - b))', 'ASC'), # Manhattan
        '<->': ('SQRT(SUM(POWER(a - b, 2)))', 'ASC'), # Euclidean
        '<=>': ('1 - SUM(a * b) / (sqrt(SUM(a * a)) * sqrt(SUM(b * b)))', 'ASC'), # Cosine
        '<#>': ('SUM(a * b)', 'DESC'), # Inner Product
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

def build_query_vector(table, operator, vector, metadata):
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

def get_images(sql_search_images, indexes):
    with psycopg.connect(os.getenv('DB_URL')) as conn:
        cursor = conn.cursor()
                         
        try:
            if indexes is None:
                cursor.execute("BEGIN;")
                cursor.execute("DROP INDEX IF EXISTS country_name_idx;")
                cursor.execute("DROP INDEX IF EXISTS region_id_idx;")
                cursor.execute("DROP INDEX IF EXISTS income_idx;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_cosine;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_l1;")
                cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_l2;")
                cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_ip;")
            
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

            image_src = os.getenv('IMAGE_HOST')

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

            if indexes is None:
                cursor.execute("ROLLBACK;")
        
            return jsonify({"images_metadata": images_metadata_formatted, "query_time": str(query_time)})

        except Exception as e:
            print(e)
            return jsonify({"error": "erro"})
    

def get_query_analysis(sql_search_images, indexes):
    with psycopg.connect(os.getenv('DB_URL')) as conn:
        cursor = conn.cursor()
              
        try:
            if indexes is None:
                cursor.execute("BEGIN;")
                cursor.execute("DROP INDEX IF EXISTS country_name_idx;")
                cursor.execute("DROP INDEX IF EXISTS region_id_idx;")
                cursor.execute("DROP INDEX IF EXISTS income_idx;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_cosine;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_l1;")
                cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_l2;")
                cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_ip;")

            
            explain_query = "EXPLAIN ANALYZE " + sql_search_images
            cursor.execute(explain_query)

            analysis = cursor.fetchall()

            if indexes is None:
                cursor.execute("ROLLBACK;")

            return jsonify({"query_analysis": analysis})
                
        except Exception as e:
            print(e)
           

def format_income(income):
    income = income.replace("$", "")
    income = income.replace(" ", "")
    min_income, max_income = income.split("-")

    return [min_income, max_income]
    
    

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    operator = request.form.get('operador')
    country = request.form.get('pais')
    table = request.form.get('tabela')
    region = request.form.get('regiao')
    action = request.form.get('acao')
    income = format_income(request.form.get('income'))
    indexes = request.form.get('use-indexes')
    
    if file and file.filename != '':
        encoder = Encoder()

        vector = encoder.encode_clip(file) if 'clip' in table else encoder.encode(file)

        metadata = []
        if country != "":
            metadata.append(f"country_name = '{country}'")
            
        if region != "":
            metadata.append(f"region_id = '{region}'")

        if income != "":
            metadata.append(f"income BETWEEN {income[0]} AND {income[1]}")

        sql_search_images = build_query_vector(table, operator, vector, metadata) \
                            if 'vector' in table \
                            else build_query_array(table, operator, vector, metadata)

                
        if action == 'get-images':
            return get_images(sql_search_images, indexes)
        
        if action == 'get-analysis':
            return get_query_analysis(sql_search_images, indexes)
        
    return "Nenhuma imagem recebida."
    

if __name__ == "__main__":
    app.run(debug = True)