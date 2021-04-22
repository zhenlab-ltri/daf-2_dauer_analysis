import json
import math
import csv
import pandas
import numpy as np
import scipy as sc

from scipy import stats
from random import choices
from neuron_info import nclass
from util import open_json, write_json

#make nondauer classifications into a dictionary
def get_classification_dict ():
    
    classification_file = 'daf2-dauer-comparisons\input\connection_classifications.csv'

    reader = csv.reader(open(classification_file, 'r'))

    next(reader)

    nondauer_classification = {}
    for row in reader:
        pre, post, classification = row
        key = pre + '$' + post

        nondauer_classification[key] = classification

    row = []

    output_json = 'daf2-dauer-comparisons\\output\\nondauer_classification.json'

    for con in nondauer_classification:
        ccon = nondauer_classification[con]
        current_row = [con]
        current_row.extend(ccon)
        row.append(current_row)

    with open(output_json, 'w') as f:
        json.dump(nondauer_classification, f, indent= 2)

    return nondauer_classification

#find nondauer classification of a connection
def get_classification(con):

    dictionary = get_classification_dict()

    if con in dictionary.keys():
        return str(dictionary[con])
    
    else:
        return ' '

def contactome_edge_list ():
    
    path = 'daf2-dauer-comparisons\\input\\L1-L3_contactome.csv'

    dt = pandas.read_csv(path).to_dict()
    dt_temp = dt.copy()

    contact_nodes = list(dt_temp.keys())
    contact_nodes.pop(0)

    edge_list ={}

    index = {}

    #making a dictionary of numbers and their corresponding partners
    for i, neuron in enumerate(contact_nodes):
        index[i] = neuron
        
    i = 0
    #making a dictionary for the neurons corresponding to each index
    for partners in dt_temp:
        pre = partners
        for num in dt_temp[partners]:
            
            post = index[num]
            key = pre + '$' + post
            if dt_temp[partners][num] == '0, 0, 0, 0, 0, 0':
                #if neurons make contact in non-dauer, make cell Y
                edge_list[key] = 'N'
            else:
                #if neurons don't make contact, nothign is in cell
                edge_list[key] = 'Y'
 
    return edge_list

# filter the data for just L1 to L2 and the 3 dauer datasets
def calculate_corecoeff(data, output_path):
    dataset = open_json(data)
    w_calculations = {}
    timepoints = [4.3, 16, 23, 27, 50, 50]

    for connections, dataset in dataset.items():
        percentages = list(dataset.values())

        early_L1 = np.mean(percentages[0:3])
        values = [early_L1]
        values.extend(percentages[3:8])
        values = np.array(values)

        is_all_zero = np.all((values == 0))

        if is_all_zero:
            pearsons = 0
            spearmans = 0

        else:
            pearsons = np.corrcoef(timepoints, values)[0][1]
            spearmans = sc.stats.spearmanr(timepoints, values).correlation

        w_calculations[connections] = percentages[0:9]
        w_calculations[connections].extend([pearsons, spearmans])

    with open(output_path, 'w') as f:
        json.dump(w_calculations, f, indent=2)

    return w_calculations


def analysis_results_to_csv(filepath, data):
    row = [['Pre Class', 'Post Class', 'Pre', 'Post', 'Early L1', 'Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM', 'daf2-dauer',
            "Pearson's correlation", "Spearman's correlation", 'classification', 'nondauer contact']]

    contactome = contactome_edge_list()

    for con in data:
        name = con.split('$')
        classification = get_classification(con)

        current_row = [nclass(name[0]), nclass(name[1]), name[0], name[1]]
        
        early_L1 = [np.mean(data[con][0:3])]
        current_row.extend(early_L1)
        
        current_row.extend(data[con][3:])
        current_row.append(classification)
        
        if con in contactome.keys():
            current_row.append(contactome[con])
        else:
            current_row.append('not in nondauer contactome')

        row.append(current_row)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(row)


if __name__ == '__main__':

    inputs = 'daf2-dauer-comparisons\output\input_percentages.json'
    outputs = 'daf2-dauer-comparisons\output\output_percentages.json'

    inputs_coreff = 'daf2-dauer-comparisons\output\modified_pipeline\input_with_pearsons_spearmans.json'
    outputs_coreff = 'daf2-dauer-comparisons\output\modified_pipeline\output_with_pearsons_spearmans.json'

    # csv output paths
    csv_input_changes_outpath = 'daf2-dauer-comparisons\output\modified_pipeline\input_cell_to_cell_changes.csv'
    csv_output_changes_outpath = 'daf2-dauer-comparisons\output\modified_pipeline\output_cell_to_cell_changes.csv'

    # json output paths
    json_input_changes_outpath = 'daf2-dauer-comparisons\output\modified_pipeline\input_cell_to_cell_changes.json'
    json_output_changes_outpath = 'daf2-dauer-comparisons\output\modified_pipeline\output_cell_to_cell_changes.json'
    
    input_w_analysis = calculate_corecoeff(inputs, inputs_coreff)
    output_w_analysis = calculate_corecoeff(outputs, outputs_coreff)
    
    write_json(json_input_changes_outpath, input_w_analysis)
    analysis_results_to_csv(csv_input_changes_outpath, input_w_analysis)

    write_json(json_output_changes_outpath, output_w_analysis)
    analysis_results_to_csv(csv_output_changes_outpath, output_w_analysis)
