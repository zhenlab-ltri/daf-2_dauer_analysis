import json
import math
import statistics
import csv
import pandas as pd
import numpy as np
import scipy as sc
from pathlib import Path

from scipy import stats
from random import choices
from neuron_info import nclass
from util import open_json, fdr_correction

#find nondauer classification of a connection
def get_classification(con):

    classification = './output/nondauer_classification.json'
    dictionary = open_json(classification)

    if con in dictionary.keys():
        return str(dictionary[con])
    
    else:
        return ' '

def get_contactome_info(pair):

    contactome = open_json('./output/contactome.json')
    
    if pair in contactome.keys():
        return contactome[pair]
    else:
        return 'not in nondauer contactome'


def calculate_corecoeff(data, output_path, average_percentage = 'early_L1', synapse_type = 'count', zero_filter = 10, fdr_correction = False):
    dataset = open_json(data)
    w_calculations = {}
    #timepoints = [0, 5, 8, 16, 23, 27, 50, 50]

    if average_percentage == 'early_L1' and synapse_type == 'count':
        timepoints = [4.3, 16, 23, 27, 50, 50]
    
    else:
        timepoints = [4.3, 16, 23, 27, 50]

    for connections, dataset in dataset.items():
        percentages = list(dataset.values())

        if average_percentage == 'early_L1':
            early_L1 = np.mean(percentages[0:3])
            values = [early_L1]
        
        if synapse_type == 'count':
            values.extend(percentages[3:8])
            values = np.array(values)

        
        else:
            values.extend(percentages[3:7])
            values = np.array(values)

        count_nonzeros = np.count_nonzero(values)
        
        #filter for 1 zero in early development, i.e. early or late L1
        if zero_filter == 'early_development': 
            if (len(values) - count_nonzeros == 1 and (values[0] == 0 or values[1] == 0)) or (count_nonzeros == len(values)):
                pearsons = sc.stats.pearsonr(timepoints, values)[0]
                spearmans = sc.stats.spearmanr(timepoints, values).correlation
                pearsons_pvalue = sc.stats.pearsonr(timepoints, values)[1]
                spearmans_pvalue = sc.stats.spearmanr(timepoints, values).pvalue
                
                #calculate standard deviation and median aboslute deviation and appending them to the list
                stdev = statistics.stdev(values)
                mad = sc.stats.median_abs_deviation(values)
                values = np.append(values, [stdev, mad])

                w_calculations[connections] = list(values)

                if synapse_type == 'count':
                    w_calculations[connections].extend([percentages[9],pearsons, spearmans, pearsons_pvalue, spearmans_pvalue])
                
                else:
                    w_calculations[connections].extend([pearsons, spearmans, pearsons_pvalue, spearmans_pvalue])

        elif type(zero_filter) == str and zero_filter != 'early_development':
            print( 'You spelled early_development wrong!')
            
        else:       
            if len(values) - count_nonzeros <= zero_filter:

                pearsons = sc.stats.pearsonr(timepoints, values)[0]
                spearmans = sc.stats.spearmanr(timepoints, values).correlation
                pearsons_pvalue = sc.stats.pearsonr(timepoints, values)[1]
                spearmans_pvalue = sc.stats.spearmanr(timepoints, values).pvalue
                
                #calculate standard deviation and median aboslute deviation and appending them to the list
                stdev = statistics.stdev(values)
                mad = sc.stats.median_abs_deviation(values)
                values = np.append(values, [stdev, mad])

                w_calculations[connections] = list(values)

                if synapse_type == 'count':
                    w_calculations[connections].extend([percentages[9],pearsons, spearmans, pearsons_pvalue, spearmans_pvalue])
                
                else:
                    w_calculations[connections].extend([pearsons, spearmans, pearsons_pvalue, spearmans_pvalue])
                    
    #FDR correction
    if fdr_correction == True:
        connections = [k for k in w_calculations.keys()]
        pearsons_pvalue = np.array([v[11] for v in w_calculations.values()])
        print(pearsons_pvalue)
        spearman_pvalue = np.array([v[12] for v in w_calculations.values()])

        pearsons_corrected, pearson_significance= fdr_correction(pearsons_pvalue)

        spearmans_corrected, spearman_significance = fdr_correction(spearman_pvalue)

        pearson_dict = dict(zip(connections, pearsons_corrected))
        pearson_sig = dict(zip(connections, pearson_significance))

        spearman_dict = dict(zip(connections, spearmans_corrected))
        spearman_sig = dict(zip(connections, spearman_significance))

        for connections, values in w_calculations.items():
            w_calculations[connections].append(pearson_dict[connections])
            w_calculations[connections].append(pearson_sig[connections])
            w_calculations[connections].append(spearman_dict[connections])
            w_calculations[connections].append(spearman_sig[connections])

    with open(output_path, 'w') as f:
        json.dump(w_calculations, f, indent=2)
    
    return w_calculations

def filter_by_pvalue (data, sig_outpath, low_outpath, pvalue_cutoff, fdr_correction = False):

    df = pd.read_csv(data)

    if fdr_correction == True:
        df_sig = df[(df["Pearson significance"] == 1) | (df["Spearman significance"] == 1)]
        df_low = pd.concat([df, df_sig]).drop_duplicates(keep=False)
    else:
        df_sig = df[(df["Pearson pvalue"] < pvalue_cutoff) | (df["Spearman pvalue"] < pvalue_cutoff)]
        df_low = pd.concat([df, df_sig]).drop_duplicates(keep=False)

    df_sig.to_csv(sig_outpath, index = False)
    df_low.to_csv(low_outpath, index = False)


def analysis_results_to_csv(filepath, data, synapse_type, fdr_correction = False):

    data = open_json(data)

    if fdr_correction == True:

        if synapse_type == 'count':
            row = [['Pre Class', 'Post Class', 'Pre', 'Post', 'Early L1', 'Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM', 'standard deviation', 'median absolute deviation', 'daf2-dauer',
                "Pearson's correlation", "Spearman's correlation", 'Pearson pvalue', 'Spearman pvalue', 'Adjusted Pearsons pvalue', 'Pearson significance', 'Adjusted Spearmans pvalue', 'Spearman significance','classification', 'nondauer contact']]

        elif synapse_type =='size':
            row = [['Pre Class', 'Post Class', 'Pre', 'Post', 'Early L1', 'Late L1', 'L2', 'L3', 'adult_SEM', 'standard deviation', 'median absolute deviation',
                "Pearson's correlation", "Spearman's correlation", 'Pearson pvalue', 'Spearman pvalue', 'Adjusted Pearsons pvalue', 'Pearson significance', 'Adjusted Spearmans pvalue','Spearman significance','classification', 'nondauer contact']]
        
    else:
        if synapse_type == 'count':
            row = [['Pre Class', 'Post Class', 'Pre', 'Post', 'Early L1', 'Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM', 'standard deviation', 'median absolute deviation', 'daf2-dauer',
                "Pearson's correlation", "Spearman's correlation", 'Pearson pvalue', 'Spearman pvalue','classification', 'nondauer contact']]

        elif synapse_type =='size':
            row = [['Pre Class', 'Post Class', 'Pre', 'Post', 'Early L1', 'Late L1', 'L2', 'L3', 'adult_SEM', 'standard deviation', 'median absolute deviation',
                "Pearson's correlation", "Spearman's correlation", 'Pearson pvalue', 'Spearman pvalue', 'classification', 'nondauer contact']]
        

    for con in data:
        name = con.split('$')
        classification = get_classification(con)

        current_row = [nclass(name[0]), nclass(name[1]), name[0], name[1]]
        
        current_row.extend(data[con])
        
        current_row.append(classification)
        
        current_row.append(get_contactome_info(con))

        row.append(current_row)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(row)


if __name__ == '__main__':
    job_dir = Path('./output/202105271854')

    input_percentages = Path(f'{job_dir}/input_percentages.json')    
    inputs_coreff = Path(f'{job_dir}/input_with_pearsons_spearmans.json')
    synapse_type = 'count'
    zero_filter = 10

    input_w_analysis = calculate_corecoeff(input_percentages, inputs_coreff, synapse_type= synapse_type, zero_filter = zero_filter)