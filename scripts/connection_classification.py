import os
from random import sample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from collections import Counter
from util import make_connection_key
from plotter import multi_histogram


def load_data(filter): 
    #load outputs from 3 different normalization methods
    df_total = make_connection_key(f'./analysis/{filter}/count/total_changes.csv')
    df_input = make_connection_key(f'./analysis/{filter}/count/input_changes.csv')
    df_output = make_connection_key(f'./analysis/{filter}/count/output_changes.csv')

    return df_total, df_input, df_output

def pvalue_filter (data, pvalue_cutoff):
    
    df = data
    df_sig = df[(df["Pearson pvalue"] < pvalue_cutoff) | (df["Spearman pvalue"] < pvalue_cutoff)]
    df_low = pd.concat([df, df_sig]).drop_duplicates(keep=False)

    return df_sig, df_low

#outputs daf2 dauer unique connections, this is the same regardless of normalization method
def daf2_dauer_unique():

    #load outputs from 3 different normalization methods
    df_total, df_input, df_output = load_data('all_connections')

    total_unique = df_total[(df_total['Spearman pvalue'].isnull()) & (df_total['daf2-dauer'] > 0)]['connection']
    input_unique = df_input[(df_total['Spearman pvalue'].isnull()) & (df_input['daf2-dauer'] > 0)]['connection']
    output_unique = df_output[(df_total['Spearman pvalue'].isnull()) & (df_output['daf2-dauer'] > 0)]['connection']

    if total_unique.equals(input_unique) & input_unique.equals(output_unique):
        daf2_dauer_unique = total_unique
        print(f'There are {len(daf2_dauer_unique)} daf2 dauer unique class connections')

    else:
        print('Error!! Different normalization methods yielded different connections! Double check files in all connections -> count')

    return daf2_dauer_unique

def daf2_dauer_pruned(filter, pvalue_cutoff):

    #load outputs from 3 different normalization methods
    total_all, input_all, output_all = load_data(filter)

    total_pruned = total_all[(total_all['Spearman pvalue'].notnull()) & (total_all['daf2-dauer'] == 0)]['connection']
    input_pruned = input_all[(input_all['Spearman pvalue'].notnull()) & (input_all['daf2-dauer'] == 0)]['connection']
    output_pruned = output_all[(output_all['Spearman pvalue'].notnull()) & (output_all['daf2-dauer'] == 0)]['connection']

    if total_pruned.equals(input_pruned) & input_pruned.equals(output_pruned):
        daf2_dauer_pruned = total_pruned
        print(f'There are {len(daf2_dauer_pruned)} daf2 dauer pruned class connections')

    else:
        print('Error!! Different normalization methods yielded different connections! Double check files in all connections -> count')

    total = total_all[total_all['connection'].isin(daf2_dauer_pruned)]
    input = input_all[input_all['connection'].isin(daf2_dauer_pruned)]
    output = output_all[output_all['connection'].isin(daf2_dauer_pruned)]

    total_sig, total_low = pvalue_filter(total, pvalue_cutoff)
    input_sig, input_low = pvalue_filter(input, pvalue_cutoff)
    output_sig, output_low = pvalue_filter(output, pvalue_cutoff)

    common_sig_list = np.intersect1d(total_sig['connection'], np.intersect1d(input_sig['connection'], output_sig['connection']))
    common_sig = total[total['connection'].isin(common_sig_list)]['connection']

    common_low_list = np.intersect1d(total_low['connection'], np.intersect1d(input_low['connection'], output_low['connection']))
    common_low = total[total['connection'].isin(common_low_list)]['connection']

    variable = len(total) - len(common_sig) - len(common_low)

    print(f'Filter type: {filter}')
    print(f'{len(common_sig)} pruned connections have pvalue < {pvalue_cutoff}, and {len(common_low)} connections have pvalue > {pvalue_cutoff}')
    print(f'{variable} pruned connections have pvalues that change depending on normalization method')

    return daf2_dauer_pruned, common_sig, common_low

def get_shared (filter, pvalue_cutoff):

    total_all, input_all, output_all = load_data(filter)

    total_shared = total_all[(total_all['Spearman pvalue'].notnull()) & (total_all['daf2-dauer'] > 0)]
    total_sig, total_low = pvalue_filter(total_shared, pvalue_cutoff)

    input_shared = input_all[(input_all['Spearman pvalue'].notnull()) & (input_all['daf2-dauer'] > 0)]
    input_sig, input_low = pvalue_filter(input_shared, pvalue_cutoff)
    
    output_shared = output_all[(output_all['Spearman pvalue'].notnull()) & (output_all['daf2-dauer'] > 0)]
    output_sig, output_low = pvalue_filter(output_shared, pvalue_cutoff)    

    common_sig_list = np.intersect1d(total_sig['connection'], np.intersect1d(input_sig['connection'], output_sig['connection']))
    total_common_sig = total_shared[total_shared['connection'].isin(common_sig_list)]
    input_common_sig = input_shared[input_shared['connection'].isin(common_sig_list)]
    output_common_sig = output_shared[output_shared['connection'].isin(common_sig_list)]
    common_sig = total_common_sig['connection']

    common_low_list = np.intersect1d(total_low['connection'], np.intersect1d(input_low['connection'], output_low['connection']))
    total_common_low = total_shared[total_shared['connection'].isin(common_low_list)]
    input_common_low = input_shared[input_shared['connection'].isin(common_low_list)]
    output_common_low = output_shared[output_shared['connection'].isin(common_low_list)]
    common_low = total_common_low['connection']

    variable = len(total_shared) - len(common_sig) - len(common_low)

    print(f'Filter type: {filter}')
    print(f'{len(total_shared)} connections are shared between nondauers and daf2 dauers')
    print(f'{len(common_sig)} shared connections have pvalue < {pvalue_cutoff}, and {len(common_low)} connections have pvalue > {pvalue_cutoff}')
    print(f'{variable} shared connections have pvalues that change depending on normalization method')

    return total_common_sig, input_common_sig, output_common_sig, total_common_low, input_common_low, output_common_low, common_sig, common_low

def inrange(data, name):

    max = data[['Early L1', 'Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM']].max(axis=1)
    min = data[['Early L1', 'Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM']].min(axis=1)
    in_range = data['daf2-dauer'].between(min, max).astype(int)
    data.insert(14, 'daf2 in nondauer range', in_range)
    #print(data[['Early L1', 'Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM', 'daf2-dauer', 'daf2 in nondauer range']])
    print(name, Counter(in_range))

    # hist = data['daf2 in nondauer range'].plot(kind = 'hist', bins = 2)
    # hist.set_xlabel('Within (1) or outside (0) nondauer range')
    # hist.set_xticks ([0,1])
    # hist.set_ylabel ('Number of daf2 dauer connections')
    
    # plt.suptitle(f'Normalization against {name} connections\n')
    # plt.show()

def daf2_mapper (data, name):

    df = data[['Early L1','Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM', 'daf2-dauer','connection']]
    timepoints =np.array([[4.3, 16, 23, 27, 50, 50]])
    xmin = min(timepoints[0])
    xmax = max(timepoints[0])
    
   
    all = df[['Early L1','Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM','daf2-dauer','connection']].set_index('connection').T
    filter_to = ['Early L1','Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM','connection']
    ax = all[all.index.isin(filter_to)]
    ax = ax.rename(index = {'Early L1': 4.3 ,'Late L1': 16, 'L2':23, 'L3':27, 'adult_TEM':50, 'adult_SEM':50})

    mapping = {}
    #hr_to_stage = {4.3:'Early L1' , 16: 'Late L1', 23:'L2', 27: 'L3', 50: 'adult'}

    for connection in ax.columns:
        x, y = ax[connection].index.values, ax[connection].values

        # Extend x data to contain another row vector of 1s
        X = np.vstack([x,np.ones(len(x))]).T
        
        for i in range(0, 1000):
            sample_index = np.random.choice(range(0, len(y)), len(y))

            x_sample = X[sample_index]
            y_sample = y[sample_index]

            lr = LinearRegression()
            lr.fit(x_sample, y_sample)
            # plt.plot(x, lr.predict(X), color='grey', alpha=0.2, zorder=1)
        
        # plt.scatter(x,y, marker='o', color='#fc8d62', zorder=4)

        lr = LinearRegression()
        lr.fit(X, y)

        daf2_dauer= all[connection].values[-1]
        # plt.hlines(daf2_dauer,xmin, xmax, linestyles='dashed', color='#d7191c', zorder = 5)
        # plt.plot(x, lr.predict(X), color = '#0571b0', zorder=2)
        
        # plt.title(connection)
        # plt.xticks(timepoints[0][:5],['Early L1','Late L1', 'L2', 'L3', 'adult'])

        y_intercept = lr.intercept_
        slope = lr.coef_[0]
        daf2x = (daf2_dauer - y_intercept)/slope 
        #hr = min(timepoints[0], key=lambda x:abs(x-daf2x))
        mapping[connection] = [daf2x]

        #plt.show()
    
    df_daf2_mapping = pd.DataFrame.from_dict(mapping, orient = 'index')
    df_daf2_mapping.columns = [name]
    # hist = df_daf2_mapping.plot(kind = 'hist')
    # hist.set_xlabel('Developmental Stage')
    # hist.set_xticks (timepoints[0][:5])
    # hist.set_xticklabels(['Early L1','Late L1', 'L2', 'L3', 'adult'])
    # hist.set_ylabel ('Number of daf2 dauer connections')

    # plt.suptitle(f'Normalization against {name} connections\n')
    # plt.show()

    return df_daf2_mapping



    

if __name__ == '__main__':

    filter = '1_zero_in_early_development'
    job_dir = f'./output/connection_lists/{filter}'
    
    cutoff = 0.05

    unique = daf2_dauer_unique()
    pruned, pruned_sig, pruned_not_sig = daf2_dauer_pruned(filter, cutoff)

    total_shared_sig, input_shared_sig, output_shared_sig, total_shared_low, input_shared_low, output_shared_low, shared_sig, shared_not_sig = get_shared(filter, cutoff)
    
    
    # inrange(total_shared_low, 'total')
    # inrange(input_shared_low, 'input')
    # inrange(output_shared_low, 'output')

    total = daf2_mapper(total_shared_sig, 'Total')
    input = daf2_mapper(input_shared_sig, 'Input')
    output = daf2_mapper(output_shared_sig, 'Output')

    multi_histogram(total, input, output)
    
    # #output csv files for daf2 dauer pruned, daf2 dauer unique, and nondauer/daf2 dauer shared connection lists
    # unique.to_csv(f'{job_dir}/daf2_dauer_unique.csv', index = False)
    # pruned_sig.to_csv(f'{job_dir}/daf2_dauer_pruned_p<{cutoff}.csv', index = False)
    # pruned_not_sig.to_csv(f'{job_dir}/daf2_dauer_pruned_p>{cutoff}.csv', index = False)
    # shared_sig.to_csv(f'{job_dir}/shared_p<{cutoff}.csv', index = False)
    # shared_not_sig.to_csv(f'{job_dir}/shared_p>{cutoff}.csv', index = False)



