import os
import pandas as pd
from util import tuple_key, write_json

path = './input/synapse_size_matrices.xlsx'
all_dfs = pd.ExcelFile(path)
sheet_names = all_dfs.sheet_names

index = {}
edge_list = {}

for sheet in sheet_names:
    
    edge_list[sheet] = {}
    df = all_dfs.parse(sheet, usecols = 'B: GA', skiprows= [0,1], header = None, index_col = 0)

    df_list = df.values.tolist()

    neurons = df_list[0][1:]

    for i, neuron in enumerate(neurons):
        index[i] = neuron

 
    df_list = df_list[1:]
   
    for row in df_list:
        post = row[0]
        for i, num in enumerate(row [1:]):
            pre = index[i]
            size = row[i+1]
            key = tuple_key(pre, post)

            edge_list[sheet][str(key)]= size


write_json('./input/nondauer_synapse_size.json', edge_list)

# making a dictionary of numbers and their corresponding partners
# for i, neuron in enumerate(df_temp):
#     index[i] = neuron


# for sheet in df_temp.keys():
#     for neuron in contact_nodes:
#         pre = neuron
        
            

            

    
       