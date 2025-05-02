
import os
import requests
import psycopg
import json
import pandas as pd
from dotenv import load_dotenv
from encoder import Encoder
from io import BytesIO

load_dotenv()

db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASS')
db_data = os.getenv('DB_DATA')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')

global conn, cursor

def getEmbeddings(imgPath):
    base = os.getenv('IMAGE_HOST')
    response = requests.get(base+imgPath)
    img_path = BytesIO(response.content)
    
    vector = encoder.encode(img_path)
    vector = vector.tolist()

    print("Embedding completed!")

    return vector

def insertImages(vectors, metadata):
    cursor.execute("SELECT count(*) FROM img_pgvector")
    count = cursor.fetchone()[0]
    
    if count == 0:
        sql = "INSERT INTO img_pgvector (id, embedding) VALUES (%s, %s)"

        try:
            cursor.executemany(sql, vectors)
            conn.commit()
            print("Inserted")
        except Exception as e:
            print(e)
    
    cursor.execute("SELECT count(*) FROM metadata")
    count = cursor.fetchone()[0]

    if count == 0:
        sql = '''INSERT INTO metadata (
            id, 
            country_name, 
            country_id, 
            region_id,
            img_rel_path,
            topics,
            place,
            income,
            imagenet_synonyms,
            imagenet_sysnet_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'''

        try: 
            cursor.executemany(sql, metadata)
            conn.commit()
            print("Inserted")
        except Exception as e:
            print(e)

def saveData(item, vectors, metadata):
    vectors.append((item['id'], item['vector']))
    metadata.append((item['id'], 
                    item['country.name'], 
                    item['country.id'], 
                    item['region.id'], 
                    item['imageRelPath'], 
                    json.loads(item['topics']), 
                    item['place'], 
                    round(item['income'],3), 
                    json.loads(item['imagenet_synonyms']), 
                    json.loads(item['imagenet_sysnet_id'])
                ))

if __name__ == "__main__":
    try:
        conn = psycopg.connect(
                            user = db_user,
                            password = db_pass,
                            dbname = db_data,
                            host = db_host,
                            port = db_port)
        cursor = conn.cursor()
    except Exception as e:
        print(e)
        print("I am unable to connect to the database")

    encoder = Encoder()
    df = pd.read_csv("dollar_street_test.csv")
    if 'vector' not in df.columns:
        df['vector'] = df['imageRelPath'].apply(getEmbeddings)
        df.to_csv("dollar_street_test.csv", index=False)

    vectors = []
    metadata = []

    df.apply(lambda item: saveData(item, vectors, metadata), axis=1)

    insertImages(vectors, metadata)

    conn.close()