import pandas as pd
import json

json_data_path = 'data/raw/pyrokinesis_raw.json'
csv_data_path = 'data/raw/pyrokinesis_raw.csv'

with open(json_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    
    df = pd.DataFrame(data)
    df.to_csv(csv_data_path, index=False, encoding='utf-8')
