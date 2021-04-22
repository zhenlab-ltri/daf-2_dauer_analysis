
import json
import math
import csv
import pandas
import numpy as np

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
def format_data(data, output_path):
    dataset = open_json(data)
    relevant_percentages = {}

    for connections, dataset in dataset.items():
        percentages = list(dataset.values())

        relevant_percentages[connections] = percentages[0:6]
        relevant_percentages[connections].extend(percentages[-3:])

    with open(output_path, 'w') as f:
        json.dump(relevant_percentages, f, indent=2)

    return relevant_percentages

# Check whether the connections exist in the two N2 datasets
def is_in_N2(data):

    if round(data[-2], 1) != 0 and round(data[-1], 1) != 0:
        return 2

    elif (data[-2] != 0) != (data[-1] != 0):
        return 1

    else:
        return 0

# calculates the nondauer median/mean and the effect size (difference between nondauer and daf2)

def additional_calculations(data):

    new_data = {}
    for connection in data:

        new_data[connection] = []

        # Calculate and take the difference as the effect size
        med = np.median(data[connection][0:6])
        dif = data[connection][6] - med

        new_data[connection] = data[connection][0:7]
        new_data[connection].extend([med, dif])
        new_data[connection].append(is_in_N2(data[connection]))

    return new_data


def bootstrap_sampling(data):

    random_sample_times = 1000

    # load dataset
    for connection in data:
        i = 0
        sample_median = []
        values = data[connection][0:6]
        daf2_value = data[connection][8]
        

        # Bootstrap Sampling
        while i < random_sample_times:
            random_sample = choices(values, k=6)
            med = np.median(random_sample)

            sample_median.append(med)
            i += 1

        if daf2_value > np.median(sample_median):
            pval = (sum(i > daf2_value for i in sample_median) + 1) / (1000 + 1)

        elif daf2_value <= np.median(sample_median):
            pval = (sum(i <= daf2_value for i in sample_median) + 1) / (1000 + 1)

        data[connection].append(pval)

    return data


def analysis_results_to_csv(filepath, data):
    row = [['Pre Class', 'Post Class', 'Pre', 'Post', 'L1_1', 'L1_2', "L1_3", 'L1_4', 'L2', 'L3', 'daf2-dauer',
            'nondauer median', 'effect size', 'in how many N2 dauers', 'p-value', 'classification', 'nondauer contact']]

    contactome = contactome_edge_list()

    for con in data:
        name = con.split('$')
        classification = get_classification(con)

        current_row = [nclass(name[0]), nclass(name[1]), name[0], name[1]]
        current_row.extend(data[con])
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

    cleaned_inputs = 'daf2-dauer-comparisons\output\input_cell_to_cell_percentages_comparisons_only.json'
    cleaned_outputs = 'daf2-dauer-comparisons\output\output_cell_to_cell_percentages_comparions_only.json'

    # csv output paths
    csv_input_changes_outpath = 'daf2-dauer-comparisons\output\input_cell_to_cell_changes.csv'
    csv_output_changes_outpath = 'daf2-dauer-comparisons\output\output_cell_to_cell_changes.csv'

    # json output paths
    json_input_changes_outpath = 'daf2-dauer-comparisons\output\input_cell_to_cell_changes.json'
    json_output_changes_outpath = 'daf2-dauer-comparisons\output\output_cell_to_cell_changes.json'
    
    input_dict = format_data(inputs, cleaned_inputs)
    output_dict = format_data(outputs, cleaned_outputs)

    input_with_median = additional_calculations(input_dict)
    input_changes_analysis = bootstrap_sampling(input_with_median)
    write_json(json_input_changes_outpath, input_changes_analysis)
    analysis_results_to_csv(csv_input_changes_outpath, input_changes_analysis)

    output_with_median = additional_calculations(output_dict)
    output_changes_analysis = bootstrap_sampling(output_with_median)
    write_json(json_output_changes_outpath, output_changes_analysis)
    analysis_results_to_csv(csv_output_changes_outpath, output_changes_analysis)
