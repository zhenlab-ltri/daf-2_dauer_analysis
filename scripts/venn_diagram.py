from re import M
import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats
from matplotlib_venn import venn2, venn3
from util import make_connection_key, merge2, merge3

def venn_diagram_generator (input_path, output_path, total_path, input_all, output_all, total_all, outpath):

    df_input = make_connection_key(input_path)
    df_output= make_connection_key(output_path)
    df_total= make_connection_key(total_path)

    df_input_all = make_connection_key(input_all)
    df_output_all = make_connection_key(output_all)
    df_total_all = make_connection_key(total_all)


    #plt.figure(figsize=(4,4))
    input = set(df_input['connection'])
    print(input)
    output = set(df_output['connection'])
    total = set(df_total['connection'])


    # different parts of the venn diagram
    input_only = (input - (output | total)) 
    output_only = (output - (input| total))
    total_only = (total - (input | output))

    input_output_shared = ((input & output) - total)
    input_total_shared = ((input & total) - output)
    output_total_shared = ((output & total) - input)

    shared = (input & output & total)
    
    venn3([input, output, total], ('Input', 'Output', 'Total'))
    plt.show()

    df_shared = merge3(df_input, df_output, df_total, shared, 'input', 'output')

    df_input_only = merge3(df_input, df_output_all, df_total_all, input_only, 'input', 'output')
    df_output_only = merge3(df_input_all, df_output, df_total_all, output_only, 'input', 'output')
    df_total_only = merge3(df_input_all, df_output_all, df_total, total_only, 'input', 'output')

    df_input_output_shared = merge3(df_input, df_output, df_total_all, input_output_shared, 'input', 'output')
    df_input_total_shared = merge3(df_input, df_total, df_output_all, input_total_shared, 'input', 'total')
    df_output_total_shared = merge3(df_output, df_total, df_input_all, output_total_shared, 'output', 'total')

    #excel outputs for each venn diagram portion
    with pd.ExcelWriter(outpath) as writer:
        df_input.to_excel (writer, sheet_name = 'input', index = False)
        df_output.to_excel (writer, sheet_name = 'output', index = False)
        df_total.to_excel (writer, sheet_name = 'total', index = False)

        df_shared.to_excel(writer, sheet_name = 'shared by all', index = False)
    
        df_input_output_shared.to_excel(writer, sheet_name='shared by input and output', index = False)
        df_input_total_shared.to_excel(writer, sheet_name='shared by input and total', index = False)
        df_output_total_shared.to_excel(writer, sheet_name='shared by output and total', index = False)

        df_input_only.to_excel(writer, sheet_name='only in input', index = False)
        df_output_only.to_excel(writer, sheet_name = 'only in output', index = False)
        df_total_only.to_excel(writer, sheet_name='only in total', index = False)



    #input_only, output_only, total_only, input_output_shared, output_total_shared, input_total_shared, all
    return df_input, df_output, df_total, df_shared

def compare_differences (dataset1, dataset2, all1, all2, name1, name2, outpath):

    df_all1 = make_connection_key(all1)

    df_all2 = make_connection_key(all2)

    dataset1_set = set(dataset1['connection'])
    dataset2_set = set(dataset2['connection'])

    #plot venn diagram
    venn2([dataset1_set, dataset2_set], (name1, name2))
    plt.show()

    dataset1_only_con = (dataset1_set - dataset2_set)
    dataset2_only_con = (dataset2_set - dataset1_set)
    shared = (dataset1_set & dataset2_set)

    #generating dataframe for shared connections
    dataset1_shared = dataset1[dataset1['connection'].isin(shared)]
    dataset2_shared = dataset2[dataset2['connection'].isin(shared)]
    df_dataset1_shared = dataset1_shared.drop(['classification', 'nondauer contact'], axis = 1)
    df_dataset2_shared = dataset2_shared.drop (['Pre Class', 'Post Class', 'Pre', 'Post' ], axis = 1)
    df_shared = pd.merge(df_dataset1_shared, df_dataset2_shared, on = 'connection', suffixes = (f'_{name1}', f'_{name2}'))

    #generating dataframe for dataset specific connections
    dataset1_only_1 = dataset1[dataset1['connection'].isin(dataset1_only_con)]
    dataset1_only_2 = df_all2[df_all2['connection'].isin(dataset1_only_con)]
    df_dataset1_only_1 = dataset1_only_1.drop(['classification', 'nondauer contact'], axis = 1)
    df_dataset1_only_2 = dataset1_only_2.drop (['Pre Class', 'Post Class', 'Pre', 'Post' ], axis = 1)
    df_dataset1_only = pd.merge(df_dataset1_only_1, df_dataset1_only_2, on = 'connection', suffixes = (f'_{name1}', f'_{name2}'))
    
    dataset2_only_1 = dataset2[dataset2['connection'].isin(dataset2_only_con)]
    dataset2_only_2 = df_all1[df_all1['connection'].isin(dataset2_only_con)]
    df_dataset2_only_1 = dataset2_only_1.drop(['classification', 'nondauer contact'], axis = 1)
    df_dataset2_only_2 = dataset2_only_2.drop (['Pre Class', 'Post Class', 'Pre', 'Post' ], axis = 1)
    df_dataset2_only = pd.merge(df_dataset2_only_1, df_dataset2_only_2, on = 'connection', suffixes = (f'_{name2}', f'_{name1}'))


    with pd.ExcelWriter(outpath) as writer:
        dataset1.to_excel (writer, sheet_name = f'{name1}', index = False)
        dataset2.to_excel (writer, sheet_name = f'{name2}', index = False)
        df_shared.to_excel(writer, sheet_name = 'shared by size and count', index = False)
        df_dataset1_only.to_excel(writer, sheet_name = f'only in {name1}', index = False)
        df_dataset2_only.to_excel(writer, sheet_name = f'only in {name2}', index = False)

def compare_shared_differences (dataset1, dataset2, name1, name2, outpath):
    
    # #get read datasets
    # df_d1input = make_connection_key(d1i)
    # df_d1output = make_connection_key(d1o)
    # df_d1total = make_connection_key(d1t)

    # df_d2input = make_connection_key(d2i)
    # df_d2output = make_connection_key(d2o)
    # df_d2output = make_connection_key(d2t)

    dataset1_set = set(dataset1['connection'])
    dataset2_set = set(dataset2['connection'])

    #plot venn diagram
    venn2([dataset1_set, dataset2_set], (name1, name2))
    #plt.show()

    dataset1_only_con = (dataset1_set - dataset2_set)
    dataset2_only_con = (dataset2_set - dataset1_set)
    shared = (dataset1_set & dataset2_set)

    #generating dataframe for shared connections
    df_shared = dataset1[dataset1['connection'].isin(shared)]

    #generating dataframe for dataset specific connections
    df_dataset1_only = dataset1[dataset1['connection'].isin(dataset1_only_con)]
    df_dataset2_only = dataset2[dataset2['connection'].isin(dataset2_only_con)]


    with pd.ExcelWriter(outpath) as writer:
        dataset1['connection'].to_excel (writer, sheet_name = f'{name1}', index = False)
        dataset2['connection'].to_excel (writer, sheet_name = f'{name2}', index = False)
        df_shared['connection'].to_excel(writer, sheet_name = 'shared by size and count', index = False)
        df_dataset1_only['connection'].to_excel(writer, sheet_name = f'only in {name1}', index = False)
        df_dataset2_only['connection'].to_excel(writer, sheet_name = f'only in {name2}', index = False)     

if __name__ == '__main__':

    zero_filter = 'all_connections'

    #paths for all connections
    input_count_all = f'./analysis/{zero_filter}/count/input_changes.csv'
    output_count_all = f'./analysis/{zero_filter}/count/output_changes.csv'
    total_count_all = f'./analysis/{zero_filter}/count/total_changes.csv'

    input_size_all = f'./analysis/{zero_filter}/size/input_changes.csv'
    output_size_all = f'./analysis/{zero_filter}/size/output_changes.csv'
    total_size_all = f'./analysis/{zero_filter}/size/total_changes.csv'

    #paths for normalization by input, output, and total synapse count
    input_count = f'./analysis/{zero_filter}/count/input_changes_p<0.05.csv'
    output_count = f'./analysis/{zero_filter}/count/output_changes_p<0.05.csv'
    total_count = f'./analysis/{zero_filter}/count/total_changes_p<0.05.csv'

    #paths for normalization by input, output, and total synapse size
    input_size = f'./analysis/{zero_filter}/size/input_changes_p<0.05.csv'
    output_size = f'./analysis/{zero_filter}/size/output_changes_p<0.05.csv'
    total_size = f'./analysis/{zero_filter}/size/total_changes_p<0.05.csv'

    #output paths for venn diagram summaries
    count_summary = f'./analysis/{zero_filter}/count_summary.xlsx'
    size_summary = f'./analysis/{zero_filter}/size_summary.xlsx'

    count_input, count_output, count_total, count_shared = venn_diagram_generator(input_count, output_count, total_count, input_count_all, output_count_all, total_count_all, count_summary)
    size_input, size_output, size_total, size_shared = venn_diagram_generator(input_size, output_size, total_size, input_size_all, output_size_all, total_size_all, size_summary)
    
    #output paths for method summaries
    input_summary = f'./analysis/{zero_filter}/input_summary.xlsx'
    output_summary = f'./analysis/{zero_filter}/output_summary.xlsx'
    total_summary = f'./analysis/{zero_filter}/total_summary.xlsx'
    shared_summary = f'./analysis/{zero_filter}/shared_summary.xlsx'
    
    #generating venn diagrams and excel sheets
    compare_differences (count_input, size_input, input_count_all, input_size_all, 'count', 'size', input_summary)
    compare_differences (count_output, size_output, output_count_all, output_size_all, 'count', 'size', output_summary)
    compare_differences (count_total, size_total, total_count_all, total_size_all, 'count', 'size', total_summary)
    compare_shared_differences(count_shared, size_shared, 'count shared', 'size shared', shared_summary)

    # #finding out where the non-overlapping connections fall with respect to size or count
    # count_only = pd.read_excel(shared_summary, 'only in count')
    # size_only = pd. read_excel(shared_summary, 'only in size')

    # df_input_nonsig= pd.read_csv('./output/count/input_cell_to_cell_changes_p>0.05.csv')
    # df_input_nonsig['connection'] = df_input_nonsig['Pre'].str.cat(df_input_nonsig['Post'], sep = '$')
    # df_input[df_input['connection'].isin(input_only)].to_csv('./output/venn_diagrams/input_only.csv')

    
    # #checks if the same connections exist in count and size 
    # size_input_only = size_input - count_input
    # df_input_nonsig= pd.read_csv('./output/count/input_cell_to_cell_changes_p>0.05.csv')
    # df_input_nonsig['connection'] = df_input_nonsig['Pre'].str.cat(df_input_nonsig['Post'], sep = '$')
    # count_input_nonsig= set(df_input_nonsig['connection'])

    # venn2([size_input_only, count_input_nonsig], ('size_input_only', 'count_input_nonsig'))
    # plt.show()

    # print(size_input_only - count_input_nonsig)


    


    # csv outputs for each venn diagram portion
    #df_input[df_input['connection'].isin(input_only)].to_csv('./output/venn_diagrams/input_only.csv')
    # df_output[df_output['connection'].isin(output_only)].to_csv('./output/venn_diagrams/output_only.csv')
    # df_total[df_total['connection'].isin(total_only)].to_csv('./output/venn_diagrams/total_only.csv')