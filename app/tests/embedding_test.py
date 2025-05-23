import psycopg
import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv('DB_URL')

def generate_random_vector(size):
    vec = np.random.uniform(high=1, low=-1, size=size)
    return [float(x) for x in vec]

def generate_random_vector_by_table(table_name):
    if 'clip' in table_name:
        return generate_random_vector(768)

    return generate_random_vector(4096)

def build_query_vector(table_name, operator, vector):
    query = f'''
                EXPLAIN ANALYZE
                SELECT vectors.id,
                       vectors.embedding {operator} '{vector}' 
                FROM {table_name} vectors
                ORDER BY vectors.embedding {operator} '{vector}' 
                LIMIT 6
            '''
            
    return query

def build_query_array(table, operator, vector):
    distances = {
        '<+>': ('SUM(ABS(a - b))', 'ASC'), # Manhattan
        '<->': ('SQRT(SUM(POWER(a - b, 2)))', 'ASC'), # Euclidean
        '<=>': ('1 - SUM(a * b) / (sqrt(SUM(a * a)) * sqrt(SUM(b * b)))', 'ASC'), # Cosine
        '<#>': ('SUM(a * b)', 'DESC'), # Inner Product
    }

    vector_str = 'ARRAY[' + ', '.join(map(str, vector)) + ']'

    query = f'''
                EXPLAIN ANALYZE
                SELECT arrays.id
                FROM {table} arrays
                ORDER BY (
                    SELECT {distances[operator][0]}
                    FROM unnest(embedding) WITH ORDINALITY AS t1(a, i) 
                    JOIN unnest({vector_str}) WITH ORDINALITY AS t2(b, j)
                    ON i = j
                ) {distances[operator][1]}
                LIMIT 6;
            '''
    
    return query

def generate_query_by_table(table_name, operator):
    vector = generate_random_vector_by_table(table_name)

    if 'vector' in table_name:
        return build_query_vector(table_name, operator, vector)
    
    return build_query_array(table_name, operator, vector)

def test_times_by_table_and_operator(table_name, operator, use_index):
            
    with psycopg.connect(db_url) as conn:
        cursor = conn.cursor()
        try:
            if use_index is False:
                cursor.execute("BEGIN;")
                cursor.execute("DROP INDEX IF EXISTS country_name_idx;")
                cursor.execute("DROP INDEX IF EXISTS region_id_idx;")
                cursor.execute("DROP INDEX IF EXISTS income_idx;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_cosine;")
                cursor.execute("DROP INDEX IF EXISTS hnsw_idx_l1;")
                cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_l2;")
                cursor.execute("DROP INDEX IF EXISTS ivfflat_idx_ip;")
            
            
            search_query = generate_query_by_table(table_name, operator)
            cursor.execute(search_query)
        
            result = cursor.fetchall()
            
            regex = r'[\d]*[.][\d]+'
            
            # this is hard-coded, maybe we should check if we see the string 'Planning Time'
            # or 'Execution Time' in each element of the result set.
            planning_time = float(re.findall(regex, result[-2][0])[0])
            if('JIT' in result[-5][0]):
                planning_time = float(re.findall(regex, result[-6][0])[0])
                    
            execution_time = float(re.findall(regex, result[-1][0])[0])
            
            
            if use_index is False:
                cursor.execute("ROLLBACK;")
            
            return {'planning_time': planning_time, 'execution_time': execution_time}

        except Exception as e:
            print(e)
            
    
operators = ['<+>', '<->', '<=>', '<#>']
tables = [
    ('img_pgvector', True), 
    ('img_pgvector_clip', True), 
    ('img_pgvector_clip', False), 
    ('img_pgarray', True), 
    ('img_pgarray_clip', True)
]

rows = []
test_amount = 20
for selected_table, use_index in tables:
    for operator in operators:
        for i in range(test_amount):
            result = test_times_by_table_and_operator(selected_table, operator, use_index)
            if not use_index:
                selected_table += ' (no index)'
            rows.append([selected_table, operator, result['execution_time'], result['planning_time']])
            print(f"Testing table '{selected_table}' with operator '{operator}' ({i+1}/{test_amount})")
    
df = pd.DataFrame(rows, columns=['table', 'operator', 'execution_time', 'planning_time'])
df.to_csv('./app/tests/embedding_test.csv')
