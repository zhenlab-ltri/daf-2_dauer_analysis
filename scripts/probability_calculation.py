
import json
import math
import csv
import numpy as np

from random import choices
from util import open_json

def format_data(data, output_path):
	dataset = open_json(data)
	relevant_percentages = {}

	for connections, dataset in dataset.items():
		percentages = list(dataset.values())
		relevant_percentages[connections] = percentages[0:5]
		relevant_percentages[connections].append(percentages[-1])
	
	with open(output_path, 'w') as f:
		json.dump(relevant_percentages, f, indent= 2)
		
	return relevant_percentages

def bootstrap_sampling(data):
    
	random_sample_times = 1000

	# load dataset
	for connection in data:
		i = 0
		sample_mean = []
		values = data[connection]

		# Bootstrap Sampling
		while i < random_sample_times:
			random_sample = choices(values, k = 5)
			avg = round(np.mean(random_sample),2)

			sample_mean.append(avg)
			i += 1
		
		if data[connection][-1] > np.mean(sample_mean):
			pval = (sum(i > data[connection][-1] for i in sample_mean)+ 1)/ (1000 +1)


		elif data[connection][-1] <= np.mean(sample_mean):
			pval = (sum(i <= data[connection][-1] for i in sample_mean) + 1)/(1000 +1)

		
		data[connection].append(pval)
	
	return data

def analysis_results_to_csv(filepath, data):
	row = [['Pre', 'Post', 'L1_1', 'L1_2', "L1_3", 'L1_4', 'L2', 'daf2-dauer', 'p-value']]

	for con in data:
		name = con.split('$')
		current_row = [name[0], name[1]]
		current_row.extend(data[con])
		row.append(current_row)
    
	with open(filepath, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(row)


if __name__ == '__main__':

	inputs = './output/input_percentages.json'
	outputs = './output/output_percentages.json'
	classification_path = './input/connection_classifications_npair.csv'

	cleaned_inputs = './output/input_percentages_comparisons_only.json'
	cleaned_outputs = './output/output_percentages_comparions_only.json'

	input_changes_outpath = './output/input_changes.csv'
	output_changes_outpath = './output/output_changes.csv'


	input_dict = format_data(inputs, cleaned_inputs)
	output_dict = format_data(outputs, cleaned_outputs)

	input_changes_analysis = bootstrap_sampling(input_dict)
	analysis_results_to_csv(input_changes_outpath, input_changes_analysis)

	output_changes_analysis = bootstrap_sampling(output_dict)
	analysis_results_to_csv(output_changes_outpath, output_changes_analysis)
	
	