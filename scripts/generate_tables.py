import sys
sys.path.append('./scripts')

import pandas as pd
import data_organizer as do
import correlation_calculation as cc

from pathlib import Path
from util import write_json, get_job_dir, open_json


def generate_table():
    job_dir = get_job_dir()
    #parameters used for each run, change if needed
    connection_type = 'neuron_pair'
    compare_contactome_with = 'all_nondauers' #all_nondauers, L1-L3
    synapse_type = 'count' #count, size
    nondauers = './input/nondauer_synapse_count.json' #nondauer_synapse_size.json, nondauer_synapse_count.json
    zero_filter = 10 #0-infinity, 'early_development'
    pvalue_cutoff = 0.05
    fdr_correction = False

    # connection_type = param_dict['connection_type']
    # compare_contactome_with = param_dict['compare_contactome_with']
    # normalize_by = param_dict['normalize_by']
    # synapse_type = param_dict['synapse_type']
    # nondauers = param_dict['nondauers']
    # zero_filter = param_dict['zero_filter']
    # pvalue_cutoff = param_dict['pvalue_cutoff']
    # fdr_correction = param_dict['fdr_correction']
    # zero_filter = param_dict['zero_filter']

    #makes a file with the parameters used
    parameter = f'{job_dir}/parameters.txt'
    f = open(parameter, "a")
    f.write(f'connection: {connection_type}\n')
    f.write(f'compared contactome with: {compare_contactome_with}\n')
    f.write(f'synapse type: {synapse_type}\n')
    f.write(f'zero filter: {zero_filter}\n')
    f.write(f'pvalue cutoff: {pvalue_cutoff}\n')
    f.write (f'FDR correction for pvalue: {fdr_correction}')


    daf2_dauer = './input/daf2-dauer.json'
    stig2 = './input/stigloher2.json'
    stig3 = './input/stigloher3.json'

    #get nondauer classifications and contactome information
    do.get_classification_dict(connection_type)
    do.contactome_edge_list(connection_type,compare_contactome_with)

    #rename nondauer datasets to content rather than dataset 1-8
    nondauer_renamed = do.rename_nondauers(nondauers.strip())

    daf2_formatted = do.input_dauer_json_converter(daf2_dauer, 'daf2-dauer')
    stig2_formatted = do.input_dauer_json_converter(stig2, 'stigloher2')
    stig3_formatted = do.input_dauer_json_converter(stig3, 'stigloher3')

    nondauer_and_dauer = do.append_dauer_to_nondauers(nondauer_renamed, stig2_formatted, stig3_formatted, daf2_formatted)

    class_connections, connections_list = do.get_connections(nondauer_and_dauer, connection_type = connection_type)

    #normalize by input and outputs
    inputs = do.get_inputs(class_connections)
    outputs = do.get_outputs(class_connections)

    input_percentages = f'{job_dir}/input_percentages.json'
    do.get_connection_input_percentages(inputs, connections_list, input_percentages, synapse_type = synapse_type)

    output_percentages = f'{job_dir}/output_percentages.json'
    do.get_connection_output_percentages(outputs, connections_list,output_percentages, synapse_type = synapse_type, )

    #calculating correlation co-efficient

    inputs_coreff = f'{job_dir}/input_with_pearsons_spearmans.json'
    outputs_coreff = f'{job_dir}/output_with_pearsons_spearmans.json'

    # csv output paths
    csv_input_changes_outpath = f'{job_dir}/input_changes.csv'

    csv_input_sig_pvalue = f'{job_dir}/input_changes_p<0.05.csv'
    csv_input_low_pvalue = f'{job_dir}/input_changes_p>0.05.csv'

    csv_output_changes_outpath = f'{job_dir}/output_changes.csv'

    csv_output_sig_pvalue = f'{job_dir}/output_changes_p<0.05.csv'
    csv_output_low_pvalue = f'{job_dir}/output_changes_p>0.05.csv'

    # json output paths
    json_input_changes_outpath = f'{job_dir}/input_changes.json'
    json_output_changes_outpath = f'{job_dir}/output_changes.json'

    #calculate correlation coefficients
    input_w_analysis = cc.calculate_corecoeff(input_percentages, inputs_coreff, synapse_type= synapse_type, zero_filter = zero_filter, fdr_correction = fdr_correction)
    output_w_analysis = cc.calculate_corecoeff(output_percentages, outputs_coreff, synapse_type= synapse_type, zero_filter = zero_filter, fdr_correction = fdr_correction)

    #outputs to json
    write_json(json_input_changes_outpath, input_w_analysis)
    write_json(json_output_changes_outpath, output_w_analysis)

    #ouptut to csv
    cc.analysis_results_to_csv(csv_input_changes_outpath, json_input_changes_outpath, synapse_type= synapse_type, fdr_correction = fdr_correction)
    cc.analysis_results_to_csv(csv_output_changes_outpath, json_output_changes_outpath, synapse_type= synapse_type, fdr_correction = fdr_correction)

    #split csv based on pvalues
    cc.filter_by_pvalue(csv_input_changes_outpath, csv_input_sig_pvalue, csv_input_low_pvalue, pvalue_cutoff = pvalue_cutoff, fdr_correction = fdr_correction)

    cc.filter_by_pvalue(csv_output_changes_outpath, csv_output_sig_pvalue,csv_output_low_pvalue, pvalue_cutoff = pvalue_cutoff, fdr_correction = fdr_correction)

    #normalize against whole dataset
    total = f'{job_dir}/total_percentages.json'

    do.get_entire_dataset_percentages(class_connections, connections_list, total, synapse_type = synapse_type)

    total_coreff = f'{job_dir}/total_with_pearsons_spearmans.json'

    # csv output paths
    csv_total_changes_outpath = f'{job_dir}/total_changes.csv'

    csv_total_sig_pvalue = f'{job_dir}/total_changes_p<0.05.csv'
    csv_total_low_pvalue = f'{job_dir}/total_changes_p>0.05.csv'

    # json output paths
    json_total_changes_outpath = f'{job_dir}/total_changes.json'

    #calculate correlation coefficients
    total_w_analysis = cc.calculate_corecoeff(total, total_coreff, synapse_type= synapse_type, zero_filter = zero_filter, fdr_correction = fdr_correction)

    #outputs to json
    write_json(json_total_changes_outpath, total_w_analysis)

    #ouptut to csv
    cc.analysis_results_to_csv(csv_total_changes_outpath, json_total_changes_outpath, synapse_type= synapse_type, fdr_correction = fdr_correction)

    #split csv based on pvalues
    cc.filter_by_pvalue(csv_total_changes_outpath, csv_total_sig_pvalue, csv_total_low_pvalue, pvalue_cutoff = pvalue_cutoff, fdr_correction = fdr_correction)


if __name__ == '__main__':
    generate_table()
    # generate_tables_param_list_path = Path('./input/generate_table_params.json')

    # params_list = open_json(str(generate_tables_param_list_path))

    # for param_dict in params_list:
    #     print(param_dict)
    #     generate_table(param_dict)



