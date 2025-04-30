import psycopg
import os
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

database = os.getenv('DATABASE_URL')
print(database)

try:
    conn = psycopg.connect(database)
except Exception as e:
    print(e)
    print("I am unable to connect to the database")


# df = pd.read_csv('dollar_street_test.csv')

# df.to_csv('dollar_street.csv')    