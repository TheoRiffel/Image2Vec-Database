from flask import Flask, render_template, request, jsonify
import psycopg
import os
from dotenv import load_dotenv
from encoder import Encoder

load_dotenv()

app = Flask(__name__)
app.app_context().push()

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    
    if file and file.filename != '':
        encoder = Encoder()
        vector = encoder.encode(file)
        vector = vector.tolist()

        with psycopg.connect(os.getenv('DB_URL')) as conn:
            cursor = conn.cursor()
            sql_search_images = f"SELECT * FROM img_pgvector ORDER BY embedding <-> '{vector}' LIMIT 5"

            try:
                cursor.execute(sql_search_images)
                images = cursor.fetchall()

                images_id = []
                for image in images:
                    id, _ = image
                    images_id.append(id)
                
                sql_get_images = f"SELECT * FROM metadata WHERE id IN {tuple(images_id)}"
                cursor.execute(sql_get_images)
                images_metadata = cursor.fetchall()

                image_src = os.getenv('IMAGE_HOST')
                images_path = []
                for image in images_metadata:
                    images_path.append(image_src + image[4].strip())
                    
                print(images_path)

                return jsonify({"images_path": images_path})

            except Exception as e:
                print(e)
    
    return "Nenhuma imagem recebida."

if __name__ == "__main__":
    app.run(debug = True)