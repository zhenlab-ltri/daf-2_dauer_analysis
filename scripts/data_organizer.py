import json
import csv
import copy

from util import open_json
from neuron_info import ntype, nclass, npair


def get_neuron_pair_key (pre, post):
    return (pre, post)

def merge(dict1, dict2, dict3, dict4):
    new = {**dict1, **dict2, **dict3, **dict4}
    return new

def input_dauer_json_converter(file,name):

    dauer_original = open_json(file)
    dauer_connections = {}
    dauer_connections[name] = {}
    #Converting the daf2 json to the same format as the nondauers
    for i, con in enumerate(dauer_original):

        key = str(get_neuron_pair_key(dauer_original[i]['partners'][0], dauer_original[i]['partners'][1]))

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

        else:
            nondauer_renamed['adult_SEM'] = nondauer_original[dataset]
        
    return nondauer_renamed

def append_dauer_to_nondauers(nondauer, stig2, stig3, daf2):
    #Added the daf2 dauer connections to the nondauer connections
    connections = merge(nondauer, daf2, stig2, stig3)

    output_path = 'daf2-dauer-comparisons\\output\\transformed-connections.json'

    with open(output_path, 'w') as f:
        json.dump(connections, f, indent= 2)
    
    return output_path

def get_class_connections(data):
    
    #open the json file
    data = open_json(data)

    class_connections = {}

    class_connection_list = []

    for dataset in data.keys():
        class_connections[dataset] = {}
        for connection in data[dataset]:

            connection_temp1 = connection.strip('"()')
            connection_temp2 = connection_temp1.replace("'", "")
            connection_list = connection_temp2.split(', ')
            
            pre = connection_list[0]
            post = connection_list[1]

            key = pre + '$' + post

            if key not in class_connections[dataset].keys():
                class_connections[dataset][key] = data[dataset][connection]
        
            else:
                class_connections[dataset][key] += data[dataset][connection]
    
    for dataset, connections in class_connections.items():
        for key in connections:
            if key not in class_connection_list:
                class_connection_list.append(key)

    #Help check numbers with nemanode to make sure it's currectly added
    #print(class_connections['L1_1']['SAA$AVA'], class_connections['L1_2']['SAA$AVA'], class_connections['L1_3']['SAA$AVA'], class_connections['L2']['SAA$AVA'])

    output_path = 'daf2-dauer-comparisons\output\connections.json'

    with open(output_path, 'w') as f:
        json.dump(class_connections, f, indent= 2)

    return output_path, class_connection_list

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
    
    output_path = 'daf2-dauer-comparisons\output\inputs.json'

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
    
    output_path = 'daf2-dauer-comparisons\output\outputs.json'

    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent= 2)

    return output_path
    
def get_class_connection_input_percentages(input_json, connections_list,dataset_names):

    input_info = open_json(input_json)
    input_percentages = {}
    consolidated_input_percentages = {}

    for dataset in input_info:
        input_percentages[dataset] = {}
        for neuron in input_info[dataset]:
            for input in input_info[dataset][neuron]:

                key = input + '$' + neuron

                if input_info[dataset][neuron]['Total'] != 0 and input != 'Total':
                    percentage = input_info[dataset][neuron][input]/input_info[dataset][neuron]['Total']*100
                    input_percentages[dataset][key] = round(percentage, 2)

    for connection in connections_list:
        consolidated_input_percentages[connection] = {}
        for name in dataset_names:
            if connection in input_percentages[name].keys():
                consolidated_input_percentages[connection][name] = input_percentages[name][connection] 
            else:
                consolidated_input_percentages[connection][name] = 0.0
    

    output_path = 'daf2-dauer-comparisons\output\input_percentages.json'

    with open(output_path, 'w') as f:
        json.dump(consolidated_input_percentages, f, indent= 2)

    return output_path

def get_class_connection_output_percentages(output_json,connections_list,dataset_names):
    
    output_info = open_json(output_json)
    output_percentages = {}
    consolidated_output_percentages = {}

    for dataset in output_info:
        output_percentages[dataset] = {}
        for neuron in output_info[dataset]:
            
            for output in output_info[dataset][neuron]:

                key = neuron + '$' + output

                if output_info[dataset][neuron]['Total'] != 0 and output != 'Total':
                    percentage = output_info[dataset][neuron][output]/output_info[dataset][neuron]['Total']*100
                    output_percentages[dataset][key] = round(percentage, 2)

    for connection in connections_list:
        consolidated_output_percentages[connection] = {}
        for name in dataset_names:
            if connection in output_percentages[name].keys():
                consolidated_output_percentages[connection][name] = output_percentages[name][connection] 
            else:
                consolidated_output_percentages[connection][name] = 0.0
    
    output_path = 'daf2-dauer-comparisons\output\output_percentages.json'

    with open(output_path, 'w') as f:
        json.dump(consolidated_output_percentages, f, indent= 2)

    return output_path



if __name__ == '__main__':


    nondauers = 'daf2-dauer-comparisons\\input\\nondauer_connections.json'
    daf2_dauer = 'daf2-dauer-comparisons\input\daf2-dauer.json'
    stig2 = 'daf2-dauer-comparisons\input\stigloher2.json'
    stig3 = 'daf2-dauer-comparisons\input\stigloher3.json'

    dataset_names = ['L1_1', 'L1_2', "L1_3", 'L1_4', 'L2', 'L3', 'adult_TEM', 'adult_SEM', 'daf2-dauer', 'stigloher2', 'stigloher3']
    comparison_dataset_names = ['L1_1', 'L1_2', "L1_3", 'L1_4', 'L2', 'daf2-dauer']


    nondauer_renamed = rename_nondauers(nondauers.strip())
    daf2_formatted = input_dauer_json_converter(daf2_dauer, 'daf2-dauer')
    stig2_formatted = input_dauer_json_converter(stig2, 'stigloher2')
    stig3_formatted = input_dauer_json_converter(stig3, 'stigloher3')

    nondauer_and_dauer = append_dauer_to_nondauers(nondauer_renamed, stig2_formatted, stig3_formatted, daf2_formatted)

    class_connections, connections_list = get_class_connections(nondauer_and_dauer)

    inputs = get_inputs(class_connections)
    outputs = get_outputs(class_connections)

    input_percentages = get_class_connection_input_percentages(inputs, connections_list, dataset_names)
    output_percentages = get_class_connection_output_percentages(outputs, connections_list, dataset_names)

 