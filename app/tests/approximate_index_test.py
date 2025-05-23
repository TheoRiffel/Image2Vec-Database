import psycopg
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv('DB_URL')

def generate_random_vector(size=768):
    vec = np.random.uniform(high=1, low=-1, size=size)
    return [float(x) for x in vec]

def build_query_vector(table_name, operator, vector):
    query = f'''
                SELECT vectors.id,
                       vectors.embedding {operator} '{vector}' 
                FROM {table_name} vectors
                ORDER BY vectors.embedding {operator} '{vector}' 
                LIMIT 6
            '''
            
    return query

def test_run_query_without_index(table_name, operator, vector):
            
    with psycopg.connect(db_url) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN;")
            cursor.execute("DROP INDEX IF EXISTS country_name_idx;")
            cursor.execute("DROP INDEX IF EXISTS region_id_idx;")
            cursor.execute("DROP INDEX IF EXISTS income_idx;")
            cursor.execute("DROP INDEX IF EXISTS hnsw_idx_cosine;")
            cursor.execute("DROP INDEX IF EXISTS hnsw_idx_l1;")
            cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_l2;")
            cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_ip;")
            
            
            search_query = build_query_vector(table_name, operator, vector)
            cursor.execute(search_query)
        
            result = cursor.fetchall()
            
            final_result = []
            for _, similarity in result:
                final_result.append(float(similarity))
                
            cursor.execute('ROLLBACK;')

            return final_result

        except Exception as e:
            print(e)

def test_run_query_with_index(table_name, operator, vector):
            
    with psycopg.connect(db_url) as conn:
        cursor = conn.cursor()
        try:
            search_query = build_query_vector(table_name, operator, vector)
            cursor.execute(search_query)
        
            result = cursor.fetchall()
            

            final_result = []
            for _, similarity in result:
                final_result.append(float(similarity))
            
            return final_result

        except Exception as e:
            print(e)
            

operators = ['<+>', '<->', '<=>', '<#>']
table = 'img_pgvector_clip'
rows = []
test_amount = 15
for operator in operators:
    for test in range(test_amount):
        vec = generate_random_vector()
        index = test_run_query_with_index(table, operator, vec)
        no_index = test_run_query_without_index(table, operator, vec)
        
        for i in range(min(len(index), len(no_index))):
            rows.append([test, operator, index[i], no_index[i]])
        
        print(f"Testing operator {operator} ({test+1}/{test_amount})")
    
df = pd.DataFrame(rows, columns=['test_num', 'operator', 'index_similarity', 'no_index_similarity'])
df.to_csv('./app/tests/approximate_index_test.csv')