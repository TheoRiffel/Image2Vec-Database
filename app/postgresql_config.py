
import os
import psycopg
import json
import pandas as pd
from dotenv import load_dotenv
from embed import getEmbeddings, getEmbeddings_clip
from datetime import datetime

load_dotenv()

db_user = os.getenv('DB_USER')
db_pass = os.getenv('DB_PASS')
db_data = os.getenv('DB_DATA')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')

global conn, cursor


def insertImages(vectors, vectors_clip, metadata):
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
            
    cursor.execute("SELECT count(*) FROM img_pgvector")
    count = cursor.fetchone()[0]
    
    if count == 0:
        sql = "INSERT INTO img_pgvector (id, embedding) VALUES (%s, %s)"

        try:
            start = datetime.now()
            cursor.executemany(sql, vectors)
            end = datetime.now()
            
            conn.commit()
            print("Inserted")
            print("Tempo de Execução (Vector 4096):", end-start)
        except Exception as e:
            print(e)
    
    cursor.execute("SELECT count(*) FROM img_pgarray")
    count = cursor.fetchone()[0]
    
    if count == 0:
        sql = "INSERT INTO img_pgarray (id, embedding) VALUES (%s, %s)"

        vectors_pg = [(vid, json.loads(vec)) for vid, vec in vectors]
        try:
            cursor.executemany(sql, vectors_pg)
            
            conn.commit()
            print("Inserted")
        except Exception as e:
            print(e)

    cursor.execute("SELECT count(*) FROM img_pgvector_clip")
    count = cursor.fetchone()[0]
    
    if count == 0:
        sql = "INSERT INTO img_pgvector_clip (id, embedding) VALUES (%s, %s)"

        try:
            start = datetime.now()
            cursor.executemany(sql, vectors_clip)
            end = datetime.now()
            
            conn.commit()
            print("Inserted")
            print("Tempo de Execução (Vector 768):", end-start)
        except Exception as e:
            print(e)
    
    cursor.execute("SELECT count(*) FROM img_pgarray_clip")
    count = cursor.fetchone()[0]
    
    if count == 0:
        sql = "INSERT INTO img_pgarray_clip (id, embedding) VALUES (%s, %s)"

        vectors_pg = [(vid, json.loads(vec)) for vid, vec in vectors_clip]
        try:
            cursor.executemany(sql, vectors_pg)
            conn.commit()
            print("Inserted")
        except Exception as e:
            print(e)

def saveData(item, vectors, vectors_clip, metadata):
    vectors.append((item['id'], item['vector']))
    vectors_clip.append((item['id'], item['vector_clip']))
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

        df = pd.read_csv("dollar_street_test.csv")

        if 'vector' not in df.columns:
            df['vector'] = df['imageRelPath'].apply(getEmbeddings)
            df.to_csv("dollar_street_test.csv", index=False)
        print("Embeddings normal completed!")

        if 'vector_clip' not in df.columns:
            df['vector_clip'] = df['imageRelPath'].apply(getEmbeddings_clip)
            df.to_csv("dollar_street_test.csv", index=False)
        print("Embeddings clip completed!")

        vectors = []
        vectors_clip = []
        metadata = []

        df.apply(lambda item: saveData(item, vectors, vectors_clip, metadata), axis=1)

        insertImages(vectors, vectors_clip, metadata)

        conn.close()

    except Exception as e:
        print(e)
        print("I am unable to connect to the database")

    