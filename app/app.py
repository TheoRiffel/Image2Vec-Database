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
            
        except Exception as e:
            print(e)
            jsonify({"error": "Erro na base de dados!"})
    
    return jsonify({"countries": country_names, "regions": regions, "min_income": min_income, "max_income": max_income})

def build_query_array(operator, vector):
    distances = {
        '<+>': ('SUM(ABS(a - b))', 'ASC'), # Manhattan
        '<->': ('SQRT(SUM(POWER(a - b, 2)))', 'ASC'), # Euclidean
        '<=>': ('1 - SUM(a * b) / (sqrt(SUM(a * a)) * sqrt(SUM(b * b)))', 'ASC'), # Cosine
        '<#>': ('SUM(a * b)', 'DESC'), # Inner Product
    }

    vector_str = 'ARRAY[' + ', '.join(map(str, vector)) + ']'
    sql_search_images = \
    f'''
        SELECT id
        FROM img_pgarray
        ORDER BY (
            SELECT {distances[operator][0]}
            FROM unnest(embedding) WITH ORDINALITY AS t1(a, i)
            JOIN unnest({vector_str}) WITH ORDINALITY AS t2(b, j)
            ON i = j
        ) {distances[operator][1]}
        LIMIT 6;
    '''

    return sql_search_images

def build_query_vector(operator, vector, metadata):
    query_start = f'''SELECT img_pgvector.id
                      FROM img_pgvector
                '''

    if len(metadata) != 0:
        join_metadata = ' AND '.join(metadata)
        query_start += f'''
                      JOIN metadata ON metadata.id = img_pgvector.id
                      WHERE {join_metadata}
                    '''
        
        
    query_end = f'''ORDER BY img_pgvector.embedding {operator} '{vector}' 
                    LIMIT 6
                ''' 
                
    sql_search_images = query_start + query_end
    
    return sql_search_images

def get_images(sql_search_images):
    with psycopg.connect(os.getenv('DB_URL')) as conn:
        cursor = conn.cursor()
                         
        try:
            start = datetime.now()
            cursor.execute(sql_search_images)
            end = datetime.now()

            print("Tempo de execução: ", end - start)
            query_time = end - start

            images = cursor.fetchall()

            images_id = [id[0] for id in images]

            if len(images_id) == 0:
                return jsonify({"error": "Nenhuma imagem encontrada!"})
            
            images_id_format = tuple(images_id)

            params = ['%s'] * len(images_id)
            params_str = ','.join(params)

            sql_get_images = f"SELECT * FROM metadata WHERE id IN ({params_str})"

            cursor.execute(sql_get_images, images_id_format)
            images_metadata = cursor.fetchall()

            image_src = os.getenv('IMAGE_HOST')
            images_path = []
            for image in images_metadata:
                images_path.append(image_src + image[4].strip())

            return jsonify({"images_path": images_path, "query_time": str(query_time)})

        except Exception as e:
            print(e)
            return jsonify({"error": "erro"})
    

def get_query_analysis(sql_search_images):
    with psycopg.connect(os.getenv('DB_URL')) as conn:
        cursor = conn.cursor()
              
        try:
            explain_query = "EXPLAIN ANALYZE " + sql_search_images
            cursor.execute(explain_query)

            analysis = cursor.fetchall()

            #formatar a analise para enviar corretamente
            return jsonify({"query_analysis": analysis})
                
        except Exception as e:
            print(e)
           

    

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    operator = request.form.get('Operador')
    country = request.form.get('Pais')
    table = request.form.get('Tabela')
    region = request.form.get('Regiao')
    action = request.form.get('acao')

    if file and file.filename != '':
        encoder = Encoder()
        vector = encoder.encode(file)
        vector = vector.tolist()

        metadata = []
        if country != "":
            metadata.append(f"country_name = '{country}'")
            
        if region != "":
            metadata.append(f"region_id = '{region}'")
                
        sql_search_images = build_query_vector(operator, vector, metadata) \
                            if table == 'img_pgvector'\
                            else build_query_array(operator, vector)

        if action == 'getImages':
            return get_images(sql_search_images)
        
        if action == 'getAnalysis':
            return get_query_analysis(sql_search_images)
        
    return "Nenhuma imagem recebida."
    

if __name__ == "__main__":
    app.run(debug = True)