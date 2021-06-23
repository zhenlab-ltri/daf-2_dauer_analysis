import os
from random import sample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import median_absolute_deviation

from sklearn.linear_model import LinearRegression
from scipy import stats
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages
from util import make_connection_key, clean_data
from plotter import multi_histogram, hist_from_list


def load_data(filter): 
    #load outputs from 3 different normalization methods
    df_total = make_connection_key(f'./analysis/{filter}/count/total_changes.csv')
    df_input = make_connection_key(f'./analysis/{filter}/count/input_changes.csv')
    df_output = make_connection_key(f'./analysis/{filter}/count/output_changes.csv')

    return df_total, df_input, df_output

def pvalue_filter (data, pvalue_cutoff):
    
    df = data
    df_sig = df[(df["Pearson pvalue"] < pvalue_cutoff) | (df["Spearman pvalue"] < pvalue_cutoff)]
    df_lo = pd.concat([df, df_sig]).drop_duplicates(keep=False)

    return df_sig, df_lo

def plot_line(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

#outputs daf2 dauer unique connections, this is the same regardless of normalization method
def daf2_dauer_unique(filter):

    #load outputs from 3 different normalization methods
    df_total, df_input, df_output = load_data('all_connections')

    total_unique = df_total[(df_total['Spearman pvalue'].isnull()) & (df_total['daf2-dauer'] > 0)]['connection']
    input_unique = df_input[(df_input['Spearman pvalue'].isnull()) & (df_input['daf2-dauer'] > 0)]['connection']
    output_unique = df_output[(df_output['Spearman pvalue'].isnull()) & (df_output['daf2-dauer'] > 0)]['connection']

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

    total_sig, total_lo = pvalue_filter(total, pvalue_cutoff)
    input_sig, input_lo = pvalue_filter(input, pvalue_cutoff)
    output_sig, output_lo = pvalue_filter(output, pvalue_cutoff)

    common_sig_list = np.intersect1d(total_sig['connection'], np.intersect1d(input_sig['connection'], output_sig['connection']))
    filter_sig = total[total['connection'].isin(common_sig_list)]
    common_sig = filter_sig[['connection','classification']]
    
    common_lo_list = np.intersect1d(total_lo['connection'], np.intersect1d(input_lo['connection'], output_lo['connection']))
    filter_lo = total[total['connection'].isin(common_lo_list)]
    common_lo = filter_lo[['connection', 'classification']]

    #remove the common significant and not significant connections to get the unstable pvalue connections
    no_common_lo = total[(~total['connection'].isin(common_lo['connection']))]
    excluded = no_common_lo[(~no_common_lo['connection'].isin(common_sig['connection']))]

    print(f'Filter type: {filter}')
    print(f'{len(common_sig)} pruned connections have pvalue < {pvalue_cutoff}, and {len(common_lo)} connections have pvalue > {pvalue_cutoff}')
    print(f'{len(excluded)} pruned connections have pvalues that change depending on normalization method')

    return daf2_dauer_pruned, common_sig, common_lo

def get_shared (filter, pvalue_cutoff):

    total_all, input_all, output_all = load_data(filter)

    total_shared = total_all[(total_all['Spearman pvalue'].notnull()) & (total_all['daf2-dauer'] > 0)]
    total_sig, total_lo = pvalue_filter(total_shared, pvalue_cutoff)

    input_shared = input_all[(input_all['Spearman pvalue'].notnull()) & (input_all['daf2-dauer'] > 0)]
    input_sig, input_lo = pvalue_filter(input_shared, pvalue_cutoff)
    
    output_shared = output_all[(output_all['Spearman pvalue'].notnull()) & (output_all['daf2-dauer'] > 0)]
    output_sig, output_lo = pvalue_filter(output_shared, pvalue_cutoff)    

    common_sig_list = np.intersect1d(total_sig['connection'], np.intersect1d(input_sig['connection'], output_sig['connection']))
    total_common_sig = total_shared[total_shared['connection'].isin(common_sig_list)]
    input_common_sig = input_shared[input_shared['connection'].isin(common_sig_list)]
    output_common_sig = output_shared[output_shared['connection'].isin(common_sig_list)]
    common_sig = total_common_sig[['connection','classification']]

    common_lo_list = np.intersect1d(total_lo['connection'], np.intersect1d(input_lo['connection'], output_lo['connection']))
    total_common_lo = total_shared[total_shared['connection'].isin(common_lo_list)]
    input_common_lo = input_shared[input_shared['connection'].isin(common_lo_list)]
    output_common_lo = output_shared[output_shared['connection'].isin(common_lo_list)]
    common_lo = total_common_lo[['connection', 'classification']]

    #remove the common significant and not significant connections to get the unstable pvalue connections
    no_common_lo = total_shared[(~total_shared['connection'].isin(common_lo['connection']))]
    excluded = no_common_lo[(~no_common_lo['connection'].isin(common_sig['connection']))]

    print(f'Filter type: {filter}')
    print(f'{len(total_shared)} connections are shared between nondauers and daf2 dauers')
    print(f'{len(common_sig)} shared connections have pvalue < {pvalue_cutoff}, and {len(common_lo)} connections have pvalue > {pvalue_cutoff}')
    print(f'{len(excluded)} shared connections have pvalues that change depending on normalization method')

    return total_common_sig, input_common_sig, output_common_sig, total_common_lo, input_common_lo, output_common_lo, common_sig, common_lo

def inrange(data, name,threshold = 2):

    data['max'] = data[['Early L1', 'Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM']].max(axis=1)
    data['min'] = data[['Early L1', 'Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM']].min(axis=1)

    in_range = data['daf2-dauer'].between(data['min'], data['max']).astype(int)
    data.insert(14, 'daf2 in nondauer range', in_range)
    #print(data[['Early L1', 'Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM', 'daf2-dauer', 'daf2 in nondauer range']])

    data['daf2 increase'] = np.where(data['daf2-dauer']> data['max'], 1, 0)
    data['daf2 decrease'] = np.where(data['daf2-dauer']< data['min'], 1, 0)

    # data['median'] = data[['Early L1', 'Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM']].median(axis = 1)
    # data['max_cutoff'] = data['max'] + threshold * data['median absolute deviation']
    # data['min_cutoff'] = data['min'] - threshold * data['median absolute deviation']

    # data['daf2 in nondauer range'] = np.where(data['daf2-dauer'].between(data['min_cutoff'], data['max_cutoff']), 1, 0)
    # data['daf2 increase'] = np.where(data['daf2-dauer']> data['max_cutoff'], 1, 0)
    # data['daf2 decrease'] = np.where(data['daf2-dauer']< data['min_cutoff'], 1, 0)

    print(name, Counter(data['daf2 in nondauer range']))
    print('increase', Counter(data['daf2 increase']))
    print('decrease', Counter(data['daf2 decrease']))

    # if data['daf2-dauer'] > data['max_cutoff']:
    #     data['daf2 in nondauer range'] = 0
    #     data['connections in daf2 dauer'] = 'increase'
    # elif data['daf2-dauer'] < data['min_cutoff']:
    #     data['daf2 in nondauer range'] = 0
    #     data['connections in daf2 dauer'] = 'decrease'
    # else:
    #     data['daf2 in nondauer range'] = 1
    #     data['connections in daf2 dauer'] = f'within {threshold} median absolute deviation '
    
    # print(data['connections in daf2 dauer'].describe)


    # hist = data['daf2 in nondauer range'].plot(kind = 'hist', bins = 2)
    # hist.set_xlabel('Within (1) or outside (0) nondauer range')
    # hist.set_xticks ([0,1])
    # hist.set_ylabel ('Number of daf2 dauer connections')
    
    # plt.suptitle(f'Normalization against {name} connections\n')
    # plt.show()

def mad_method(df, variable_name, threshold):
    #Takes two parameters: dataframe & variable of interest as string

    med = np.median(df, axis = 0)
    mad = np.abs(stats.median_absolute_deviation(df))
    threshold = threshold
    outlier = []
    index=0
    for item in range(len(columns)):
        if columns[item] == variable_name:
            index == item
    for i, v in enumerate(df.loc[:,variable_name]):
        t = (v-med[index])/mad[index]
        if t > threshold:
            outlier.append(i)
        else:
            continue
    return outlier
def get_ci(data):
    mean = np.median(data)
    std = np.std(data)
    ci_hi = mean + 1.96* std
    ci_lo= mean - 1.96* std

    return ci_lo, ci_hi

def daf2_mapper (data, name, cutoff):

    #index to keep track of number of connections
    i = 0

    #filter for connections with pearson's pvalue < cutoff
    df = data[data['Pearson pvalue'] < cutoff]

    df = df[['Early L1','Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM', 'daf2-dauer','connection','classification']]
    timepoints =np.array([[4.3, 16, 23, 27, 50, 50]])
    xmin = min(timepoints[0])
    xmax = max(timepoints[0])


    all = df[['Early L1','Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM','daf2-dauer','connection']].set_index('connection').T
    filter_to = ['Early L1','Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM','connection']
    ax = all[all.index.isin(filter_to)]
    ax = ax.rename(index = {'Early L1': 4.3 ,'Late L1': 16, 'L2':23, 'L3':27, 'adult_TEM':50, 'adult_SEM':50})

    mapping = {}
    dauer_state = {}
    all_neural_ages = []
    ci = [] 
    
    for connection in ax.columns:
        x, y = ax[connection].index.values, ax[connection].values

        con_name = connection.replace('/', '')
        pp = PdfPages(f'./graphs/{name}/{con_name}.pdf')
        fig = plt.figure(figsize=(16,6))
        ax0 = fig.add_subplot(121)

        ##plot data as scatter
        ax0.scatter(x,y, marker='o', color='#fc8d62', zorder=4)

        # Extend x data to contain another row vector of 1s
        X = np.vstack([x,np.ones(len(x))]).T

        #Perform linear regression
        lr = LinearRegression()
        lr.fit(X, y)

        #Get daf2 dauer value
        daf2_dauer= all[connection].values[-1]


        #Plot daf2 (horizontal line) and the regression line
        ax0.hlines(daf2_dauer,xmin, xmax, linestyles='dashed', color='#d7191c', zorder = 5)
        ax0.plot(x, lr.predict(X), color = '#0571b0', zorder=2)
        
        ax0.title.set_text(connection)
        ax0.set_xticks(timepoints[0][:5])
        ax0.set_xticklabels(['Early L1','Late L1', 'L2', 'L3', 'adult'])

        #Get slope and y-intercept for regression line
        slope = lr.coef_[0]
        y_intercept = lr.intercept_
        
        #Using the above slope and intercept to caculate x value (neural age)
        daf2x = (daf2_dauer - y_intercept)/slope 

        # Add neural age cacluated based on real data to a dictionary
        mapping[connection] = [daf2x]

        #making lists for slopes, intercepts and neural ages from bootstrapping
        bs_slopes = []
        bs_intercepts = []
        bs_neural_ages = []
        
        #draw bootstrapping lines
        for i in range(0, 1000):
            #randomly sample from data
            sample_index = np.random.choice(range(0, len(y)), len(y))

            x_sample = X[sample_index]
            y_sample = y[sample_index]

            lr = LinearRegression()
            line = lr.fit(x_sample, y_sample)

            m = line.coef_[0]
            b = line.intercept_

            if m == 0:
                neural_age = b

            else:
                neural_age = (daf2_dauer - b)/m 
            
            #Add the slope and intercept to the master list
            bs_slopes.append(m)
            bs_intercepts.append(b)
            bs_neural_ages.append(neural_age)
    
        
        #calculating the mean, standard deviation and 95% confidence interval of the slope and y intercepts
        ci_lo_slope, ci_hi_slope = get_ci(bs_slopes)
        ci_lo_intercept,ci_hi_intercept = get_ci(bs_intercepts)

        for i in range(0, 1000):
            m = np.random.uniform(ci_lo_slope, ci_hi_slope)
            b = np.random.uniform(ci_lo_intercept, ci_hi_intercept)
            
            y_vals = m * x + b

            plt.plot(x, y_vals, color='grey', alpha=0.2, zorder=1)
        
        transformed_nage = clean_data(bs_neural_ages, [0, 50])
        ax1 = fig.add_subplot(122)
        ax1.hist(transformed_nage, 20)
        ax1.title.set_text(f'{connection} Neural Age')
        
        pp.savefig(fig)
        
        all_neural_ages = all_neural_ages + bs_neural_ages


        ci_lo_nage, ci_hi_nage = get_ci(bs_neural_ages)

        # add confidence interval to master list
        ci.append([ci_lo_nage, ci_hi_nage])

        if daf2_dauer > ci_hi_nage:
            dauer_state[connection] = 'increase'
        elif daf2_dauer < ci_lo_nage:
            dauer_state[connection] = 'decrease'
        else:
            dauer_state[connection] = 'within 95% confidence interval'
        
        pp.close()
    
    hist_from_list(all_neural_ages, name, f'All {name} Neural Age', 20, [0, 50])
    df['connections in daf2 dauer'] = df['connection'].map(dauer_state)
    
    df_daf2_mapping = pd.DataFrame.from_dict(mapping, orient = 'index')
    df_daf2_mapping = df_daf2_mapping.rename_axis("connection")
    df_daf2_mapping.columns = [name]
    df_daf2_mapping['low_error'] = [abs(i[0]) for i in ci]
    df_daf2_mapping['high_error'] = [abs(i[1]) for i in ci]

    classification = df[['connection','classification']]
    with_classification = df_daf2_mapping.merge(classification, on = ['connection'], how = 'inner')


    df = with_classification.sort_values(by = [f'{name}'])
    df_new = df.reset_index()
    df_new['index'] = df_new.index

    error = [df_new['low_error'].tolist()] + [df_new['high_error'].tolist()]
 

    colors = {'developmentally dynamic (weakened)':'red', 'developmentally dynamic (strengthened)':'green', 'variable':'skyblue', 'stable':'navy', 'post-embryonic brain integration' : 'grey'}

    df_new.plot.scatter(x = 'index', y = f'{name}', c= df_new['classification'].map(colors), ylim = (-100, 150), zorder = 100, legend = True, figsize = (16, 6))
    plt.errorbar(df_new['index'], df_new[f'{name}'], yerr = error, ecolor = 'grey', fmt = 'none', capsize = 5, zorder = -32)
    

    # The following two lines generate custom fake lines that will be used as legend entries:
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colors.values()]
    plt.legend(markers, colors.keys(), numpoints=1, title='Connection Classification', bbox_to_anchor=(1.05, 1), loc='upper left')

    
    plt.xlabel('Connection (Pre$Post)')
    plt.xticks(ticks = np.arange(min(df_new['index']), max(df_new['index']+1), step = 1), labels = df_new['connection'], rotation = 45, ha = 'right')
    plt.ylabel (f'Neural age (hrs after birth)')

    plt.suptitle(f'Normalization against {name} connections\n')
    plt.tight_layout()
    plt.savefig(f'./graphs/{name}/Normalization against {name} connections zoomed in.pdf', bbox_inches = 'tight')
    
    return df_daf2_mapping
    

if __name__ == '__main__':

    filter = '1_zero_in_early_development'
    job_dir = f'./output/connection_lists/{filter}'
    
    cutoff = 0.05

    unique = daf2_dauer_unique(filter)
    pruned, pruned_sig, pruned_not_sig = daf2_dauer_pruned(filter, cutoff)

    total_shared_sig, input_shared_sig, output_shared_sig, total_shared_lo, input_shared_lo, output_shared_lo, shared_sig, shared_not_sig = get_shared(filter, cutoff)
    
    
    # inrange(total_shared_lo, 'total')
    # inrange(input_shared_lo, 'input')
    # inrange(output_shared_lo, 'output')

    total = daf2_mapper(total_shared_sig, 'Total', cutoff)
    # input = daf2_mapper(input_shared_sig, 'Input', cutoff)
    # output = daf2_mapper(output_shared_sig, 'Output', cutoff)

    # multi_histogram(total, input, output,'Total', 'Input', 'Output')
    
    # #output csv files for daf2 dauer pruned, daf2 dauer unique, and nondauer/daf2 dauer shared connection lists
    # unique.to_csv(f'{job_dir}/daf2_dauer_unique.csv', index = False)
    # pruned_sig.to_csv(f'{job_dir}/daf2_dauer_pruned_p<{cutoff}.csv', index = False)
    # pruned_not_sig.to_csv(f'{job_dir}/daf2_dauer_pruned_p>{cutoff}.csv', index = False)
    # shared_sig.to_csv(f'{job_dir}/shared_p<{cutoff}.csv', index = False)
    # shared_not_sig.to_csv(f'{job_dir}/shared_p>{cutoff}.csv', index = False)



