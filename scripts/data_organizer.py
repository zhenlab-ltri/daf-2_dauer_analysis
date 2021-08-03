import json
import csv
import copy
import pandas as pd
import numpy as np

from util import open_json, write_json, tuple_key, merge, get_key, remove_space, comma_string_to_list
from neuron_info import ntype, nclass, npair

#Converting the daf2 json to the same format as the nondauers
def input_dauer_json_converter(file,name):

    dauer_original = open_json(file)
    dauer_connections = {}
    dauer_connections[name] = {}
    
    
    for i, con in enumerate(dauer_original):

        key = str(tuple_key(dauer_original[i]['partners'][0], dauer_original[i]['partners'][1]))

        if key not in dauer_connections[name].keys():
            dauer_connections[name][key] = 1
        
        else:
            dauer_connections[name][key] += 1
    
    return dauer_connections

def rename_nondauers(nondauer):

    nondauer_original = open_json(nondauer)
    nondauer_renamed = {}

    #Renamed the dictionary with the appropriate developmental stage
    for dataset, connections in nondauer_original.items():
        if dataset == 'Dataset1':
            nondauer_renamed['L1_1'] = nondauer_original[dataset]
        
        elif dataset == 'Dataset2':
            nondauer_renamed['L1_2'] = nondauer_original[dataset]
        
        elif dataset == 'Dataset3':
            nondauer_renamed['L1_3'] = nondauer_original[dataset]

        elif dataset == 'Dataset4':
            nondauer_renamed['L1_4'] = nondauer_original[dataset]
        
        elif dataset == 'Dataset5':
            nondauer_renamed['L2'] = nondauer_original[dataset]

        elif dataset == 'Dataset6':
            nondauer_renamed['L3'] = nondauer_original[dataset]
        
        elif dataset == 'Dataset7':
            nondauer_renamed['adult_TEM'] = nondauer_original[dataset]

        elif dataset == 'Dataset8':
            nondauer_renamed['adult_SEM'] = nondauer_original[dataset]
        
        else:
            print('There are unknown dataset names')
            continue
    return nondauer_renamed

#Added the dauer synapse counts to the nondauer connections
def append_dauer_to_nondauers(nondauer, stig2, stig3, daf2):
    
    connections = merge(nondauer, daf2, stig2, stig3)

    output_path = './output/transformed-connections.json'

    with open(output_path, 'w') as f:
        json.dump(connections, f, indent= 2)
    
    return output_path

#Get connections based on cell-to-cell, neuron pairs, or neuron classes
def get_connections(data, connection_type = 'cell-to-cell'):
    
    assert connection_type in ('cell-to-cell', 'neuron_pair', 'neuron_class')

    #open the json file
    data = open_json(data)

    connections = {}

    connection_list = []

    for dataset in data.keys():
        connections[dataset] = {}
        for connection in data[dataset]:

            connection_temp1 = connection.strip('"()')
            connection_temp2 = connection_temp1.replace("'", "")
            neuron_list = connection_temp2.split(', ')
            
            pre = neuron_list[0]
            post = neuron_list[1]

            if connection_type == 'cell-to-cell':
                key = get_key(pre, post)
            
            elif connection_type == 'neuron_pair':
                key = get_key(npair(pre), npair(post))

            else:
                key = get_key(nclass(pre), nclass(post))

            if key not in connections[dataset].keys():
                connections[dataset][key] = data[dataset][connection]
        
            else:
                connections[dataset][key] += data[dataset][connection]

    for dataset, connection in connections.items():
        for key in connection:
            if key not in connection_list:
                connection_list.append(key)

    #Help check numbers with nemanode to make sure it's currectly added
    #print(connections['L1_1']['SAA$AVA'], connections['L1_2']['SAA$AVA'], connections['L1_3']['SAA$AVA'], connections['L2']['SAA$AVA'])

    output_path = './output/connections.json'

    with open(output_path, 'w') as f:
        json.dump(connections, f, indent= 2)

    return output_path, connection_list


def get_inputs(class_json):

    full_data = open_json(class_json)

    inputs = {}

    for dataset in full_data:

        inputs[dataset] = {}
        for connection in full_data[dataset]:

            pre, post = connection.split('$')

            if post not in inputs[dataset].keys():
                inputs[dataset][post] = {}
                inputs[dataset][post][pre] = full_data[dataset][connection]
            
            elif post in inputs[dataset].keys() and pre not in inputs[dataset][post]:
                inputs[dataset][post][pre]= full_data[dataset][connection]
    
    for dataset in inputs:
        for neuron in inputs[dataset]:
            inputs[dataset][neuron]['Total'] = sum(inputs[dataset][neuron].values())
    
    output_path = './output/inputs.json'

    with open(output_path, 'w') as f:
        json.dump(inputs, f, indent= 2)

    return output_path

def get_outputs(class_json):
    
    full_data = open_json(class_json)

    outputs = {}

    for dataset in full_data:

        outputs[dataset] = {}
        for connection in full_data[dataset]:

            pre, post = connection.split('$')

            if pre not in outputs[dataset].keys():
                outputs[dataset][pre]= {}
                outputs[dataset][pre][post] = full_data[dataset][connection]
            
            elif pre in outputs[dataset].keys() and post not in outputs[dataset][pre]:
                outputs[dataset][pre][post] = full_data[dataset][connection]
    
    for dataset in outputs:
        for neuron in outputs[dataset]:
            outputs[dataset][neuron]['Total'] = sum(outputs[dataset][neuron].values())
    
    output_path = './output/outputs.json'

    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent= 2)

    return output_path

def get_entire_dataset_percentages(class_json, connections_list, outpath, synapse_type = 'count'):

    full_data = open_json(class_json)

    entire_dataset = {}

    if synapse_type == 'count':
        dataset_names = ['L1_1', 'L1_2', "L1_3", 'L1_4', 'L2', 'L3', 'adult_TEM', 'adult_SEM', 'daf2-dauer', 'stigloher2', 'stigloher3']

    else:
        dataset_names = ['L1_1', 'L1_2', "L1_3", 'L1_4', 'L2', 'L3', 'adult_SEM', 'daf2-dauer', 'stigloher2', 'stigloher3']
    
    for connection in connections_list:
        
        entire_dataset[connection] = {}
        for name in dataset_names:
            if connection in full_data[name].keys():
                total = sum(full_data[name].values())
                percentage = full_data[name][connection]/total*100
                entire_dataset[connection][name] = percentage
            else:
                entire_dataset[connection][name] = 0.0

    with open(outpath, 'w') as f:
        json.dump(entire_dataset, f, indent= 2)


def get_connection_input_percentages(input_json, connections_list, outpath, synapse_type = 'count'):

    input_info = open_json(input_json)
    input_percentages = {}
    consolidated_input_percentages = {}

    if synapse_type == 'count':
        dataset_names = ['L1_1', 'L1_2', "L1_3", 'L1_4', 'L2', 'L3', 'adult_TEM', 'adult_SEM', 'daf2-dauer', 'stigloher2', 'stigloher3']

    else:
        dataset_names = ['L1_1', 'L1_2', "L1_3", 'L1_4', 'L2', 'L3', 'adult_SEM', 'daf2-dauer', 'stigloher2', 'stigloher3']

    for dataset in input_info:
        input_percentages[dataset] = {}
        for neuron in input_info[dataset]:
            for input in input_info[dataset][neuron]:

                if input_info[dataset][neuron]['Total'] != 0 and input != 'Total':
                    key = get_key(input, neuron)
                    
                    percentage = input_info[dataset][neuron][input]/input_info[dataset][neuron]['Total']*100
                    input_percentages[dataset][key] = percentage

    for connection in connections_list:
    
        consolidated_input_percentages[connection] = {}
        for name in dataset_names:
            if connection in input_percentages[name].keys():
                consolidated_input_percentages[connection][name] = input_percentages[name][connection] 
            else:
                consolidated_input_percentages[connection][name] = 0.0


    with open(outpath, 'w') as f:
        json.dump(consolidated_input_percentages, f, indent= 2)


def get_connection_output_percentages(output_json,connections_list, outpath, synapse_type = 'count'):
    
    output_info = open_json(output_json)
    output_percentages = {}
    consolidated_output_percentages = {}

    if synapse_type == 'count':
        dataset_names = ['L1_1', 'L1_2', "L1_3", 'L1_4', 'L2', 'L3', 'adult_TEM', 'adult_SEM', 'daf2-dauer', 'stigloher2', 'stigloher3']

    else:
        dataset_names = ['L1_1', 'L1_2', "L1_3", 'L1_4', 'L2', 'L3', 'adult_SEM', 'daf2-dauer', 'stigloher2', 'stigloher3']

    for dataset in output_info:
        output_percentages[dataset] = {}
        for neuron in output_info[dataset]:
            
            for output in output_info[dataset][neuron]:

                key = neuron + '$' + output

                if output_info[dataset][neuron]['Total'] != 0 and output != 'Total':
                    percentage = output_info[dataset][neuron][output]/output_info[dataset][neuron]['Total']*100
                    output_percentages[dataset][key] = percentage

    for connection in connections_list:
        consolidated_output_percentages[connection] = {}
        for name in dataset_names:
            if connection in output_percentages[name].keys():
                consolidated_output_percentages[connection][name] = output_percentages[name][connection] 
            else:
                consolidated_output_percentages[connection][name] = 0.0

    with open(outpath, 'w') as f:
        json.dump(consolidated_output_percentages, f, indent= 2)


#make nondauer classifications into a dictionary
def get_classification_dict (connection_type):
    
    assert connection_type in ('cell-to-cell', 'neuron_pair', 'neuron_class')

    if connection_type == 'cell-to-cell':
        classification_file = './input/connection_classifications.csv'
    
    else:
        classification_file = './input/connection_classifications_npair.csv'

    reader = csv.reader(open(classification_file, 'r'))

    next(reader)

    nondauer_classification = {}
    for row in reader:
        pre, post, classification = row
        key = get_key(pre, post)

        nondauer_classification[key] = classification

    row = []

    output_json = './output/nondauer_classification.json'

    for con in nondauer_classification:
        ccon = nondauer_classification[con]
        current_row = [con]
        current_row.extend(ccon)
        row.append(current_row)

    with open(output_json, 'w') as f:
        json.dump(nondauer_classification, f, indent= 2)

    return nondauer_classification

def contactome_edge_list (connection_type, compare_contactome_with):

    assert compare_contactome_with in ('all_nondauers', 'L1-L3' )
    
    path = './input/contactome.csv'

    dt = pd.read_csv(path).to_dict()
    dt_temp = dt.copy()

    contact_nodes = list(dt_temp.keys())
    contact_nodes.pop(0)

    edge_list = {}

    index = {}

    #making a dictionary of numbers and their corresponding partners
    for i, neuron in enumerate(contact_nodes):
        index[i] = neuron
        
    i = 0
    #making a dictionary for the neurons corresponding to each index
    for partners in contact_nodes:
        pre = partners
        for num in dt_temp[partners]:
            
            post = index[num]

            #convert adjacency matrix values into numpy array
            contacts = comma_string_to_list(remove_space(dt_temp[partners][num]))

            key = get_key(pre, post)
            
            if compare_contactome_with == 'all_nondauers':
            
                if np.all(contacts == '0'):
                    #if neurons don't make contact, nothing is in cell
                    edge_list[key] = 'N'
                else:
                    #if neurons make contact in non-dauer, make cell Y
                    edge_list[key] = 'Y'
            
            elif compare_with == 'L1-L3':

                if np.all(contacts[:-1] == '0'):
                    #if neurons don't make contact, nothing is in cell
                    edge_list[key] = 'N'
                else:
                    #if neurons make contact in non-dauer, make cell Y
                    edge_list[key] = 'Y'
            
    if connection_type == 'cell-to-cell':
        
        json_path = './output/contactome.json'     
        write_json(json_path, edge_list)
            
    elif connection_type == 'neuron_pair':

        npair_edge_list = {}
        for connection in edge_list:
            pair = connection.split('$')
            key = get_key(npair(pair[0]), npair(pair[1]))

            if key not in npair_edge_list.keys():
                npair_edge_list[key] = edge_list[connection]
            else:
                if npair_edge_list[key] == 'N':
                    npair_edge_list[key] = edge_list[connection]
                else:
                    continue
        
        json_path = './output/contactome.json'     
        write_json(json_path, npair_edge_list)
                    

if __name__ == '__main__':

    #change if needed 
    connection_type = 'neuron_pair' #cell-to-cell, neuron_pair
    compare_contactome_with = 'all_nondauers' #all_nondauers, L1-L3
    normalize_by = 'entire_dataset' #input, output, entire_dataset
    synapse_type = 'count' #count, size
    nondauers = f'./input/nondauer_synapse_{synapse_type}.json' #nondauer_synapse_size.json, nondauer_synapse_count.json
    
    daf2_dauer = './input/daf2-dauer.json'
    stig2 = './input/stigloher2.json'
    stig3 = './input/stigloher3.json'

    input_path = './output/input_percentages.json'
    output_path = './output/output_percentages.json'
    total_path = './output/total_percentages.json'


    nondauer_renamed = rename_nondauers(nondauers.strip())
    daf2_formatted = input_dauer_json_converter(daf2_dauer, 'daf2-dauer')
    stig2_formatted = input_dauer_json_converter(stig2, 'stigloher2')
    stig3_formatted = input_dauer_json_converter(stig3, 'stigloher3')

    nondauer_and_dauer = append_dauer_to_nondauers(nondauer_renamed, stig2_formatted, stig3_formatted, daf2_formatted)

    class_connections, connections_list = get_connections(nondauer_and_dauer, connection_type = connection_type)

    #normalize against whole dataset
    get_entire_dataset_percentages(class_connections, connections_list, total_path, synapse_type = synapse_type)

    #normalize by input and outputs
    inputs = get_inputs(class_connections)
    outputs = get_outputs(class_connections)

    input_percentages = get_connection_input_percentages(inputs, connections_list, input_path, synapse_type = synapse_type)
    output_percentages = get_connection_output_percentages(outputs, connections_list, output_path, synapse_type = synapse_type)

    #normalize across whole datasets

    get_classification_dict(connection_type)

    contactome_edge_list(connection_type,compare_contactome_with)