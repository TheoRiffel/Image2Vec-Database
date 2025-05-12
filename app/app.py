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

@app.route('/countries', methods=['GET'])
def get_countries():
    print("get countries")
    country_names = []
    with psycopg.connect(os.getenv('DB_URL')) as conn:
        cursor = conn.cursor()
        sql_get_countries = f"SELECT DISTINCT(country_name) FROM metadata"

        try:
            cursor.execute(sql_get_countries)
            countries = cursor.fetchall()

            for country_name in countries:
                name = country_name[0]
                country_names.append(name)
            
        except Exception as e:
            print(e)
    return jsonify({"data": country_names})

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

def build_query_vector(operator, vector):
    sql_search_images = f'''SELECT id 
                            FROM img_pgvector 
                            ORDER BY embedding {operator} '{vector}' 
                            LIMIT 6
                        ''' 
    
    return sql_search_images

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    operator = request.form.get('Operador')
    country = request.form.get('Pais')
    table = request.form.get('Tabela')

    if file and file.filename != '':
        encoder = Encoder()
        vector = encoder.encode(file)
        vector = vector.tolist()

        with psycopg.connect(os.getenv('DB_URL')) as conn:
            cursor = conn.cursor()
            
            sql_search_images = build_query_vector(operator, vector) \
                                if table == 'img_pgvector'\
                                else build_query_array(operator, vector)
            
           
            try:
                start = datetime.now()
                cursor.execute(sql_search_images)
                end = datetime.now()

                print("Tempo de execução: ", end - start)
                query_time = end - start

                images = cursor.fetchall()

                images_id = [id[0] for id in images]
                
                sql_get_images = f"SELECT * FROM metadata WHERE id IN {tuple(images_id)}"
                cursor.execute(sql_get_images)
                images_metadata = cursor.fetchall()

                image_src = os.getenv('IMAGE_HOST')
                images_path = []
                for image in images_metadata:
                    images_path.append(image_src + image[4].strip())

                return jsonify({"images_path": images_path, "query_time": str(query_time)})

            except Exception as e:
                print(e)
    
    return "Nenhuma imagem recebida."

if __name__ == "__main__":
    app.run(debug = True)