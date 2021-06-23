import numpy as np
from numpy.lib.nanfunctions import _nanmedian1d
from numpy.ma import compress
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import plotly.express as px
import plotly.figure_factory as ff
from plotly.offline import plot 

from sklearn.linear_model import LinearRegression 


def scatter_plotter(path, title, synapse_type = 'count'):
    

    df = pd.read_csv(path)

    if synapse_type == 'count':

        fig = px.scatter(df, x= "Pearson's correlation", y = "Spearman's correlation",
                            hover_data= ['Pearson pvalue', 'Spearman pvalue','Pre', 'Post', 'Early L1', 'Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM','nondauer contact'], 
                            color = 'classification')
    
    else:
        fig = px.scatter(df, x= "Pearson's correlation", y = "Spearman's correlation",
                            hover_data= ['Pearson pvalue', 'Spearman pvalue','Pre', 'Post', 'Early L1', 'Late L1', 'L2', 'L3', 'adult_SEM','nondauer contact'], 
                            color = 'classification')

    #center title
    fig.update_layout(title_text= title, title_x=0.5)

    fig.show()

def hist_from_list(data,folder_name, title, bins, range):
    
    filter = np.isfinite(data)
    plot = list(compress(filter, data))

    new_plot = [(range[0]-5) if value < range[0] else (range[1]+5) if value > range[1] else value for value in plot]
    plt.figure(figsize=(8,6))
    plt.hist(new_plot, bins = bins)
    
    #comment out if wanting to see all the data
    #plt.xlim(range[0], range[1])
    #plt.ylim(0, 2500)
    plt.title(title)

    plt.savefig(f'./graphs/{folder_name}/{title}.pdf', bbox_inches = 'tight')

def histogram_from_csv(path, title, plot = plot, synapse_type = 'count'):

    df = pd.read_csv(path)

    if synapse_type == 'count':
        
        fig = px.histogram(df, title = title, x = plot, hover_data = ['Pearson pvalue', 'Spearman pvalue','Pre', 'Post', 'Early L1', 'Late L1', 'L2', 'L3', 'adult_TEM', 'adult_SEM','nondauer contact'],
        color = 'classification' )
    
    else:
        fig = px.histogram(df, x= plot,
                            hover_data= ['Pearson pvalue', 'Spearman pvalue','Pre', 'Post', 'Early L1', 'Late L1', 'L2', 'L3', 'adult_SEM','nondauer contact'], 
                            color = 'classification')
    
    #center title
    fig.update_layout(title_text= title, title_x=0.5)

    fig.show()

def kernel_density_plot(path, title):

    df = pd.read_csv(path)

    colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]

    fig = ff.create_2d_density(
        x = df["Pearson's correlation"], y = df["Spearman's correlation"], colorscale=colorscale,
        hist_color='rgb(43,140,190)', point_size=3, title = title
    )

    #center title
    fig.update_layout(title_text= title, title_x=0.5)

    fig.show()

def multi_histogram(data1, data2, data3, name1, name2, name3):



    data12 = data1.merge(data2, left_index = True, right_index = True)
    dataf = data12.merge(data3, left_index = True, right_index = True)
    dataf.to_csv('./output.csv')

    timepoints = np.array([4.3, 16, 23, 27, 50])
    xlabel = 'Predited hours after birth'
    ylabel = 'Number of daf2 dauer connections'
    title = 'Neural age prediction based on different normalization methods'

    
    #plt.figure(figsize=(8,6))
    #sns.histplot(data = dataf[data1.columns[0]], color = 'skyblue', label=dataf.columns[0], kde = True)
    #sns.histplot(data = dataf[dataf.columns[1]], color= 'gold', label=dataf.columns[1], kde = True)
    #sns.histplot(data = dataf[dataf.columns[2]], color= 'teal', label=dataf.columns[2], kde = True)

     
    x_axis = np.arange(-250, 150, 50)
    xticks = np.concatenate((x_axis, timepoints))

    plt.title(title)
    plt.xlabel(xlabel, size=14)
    plt.xticks(xticks)
    plt.ylabel(ylabel, size=14)

    plt.legend(loc='upper right')
    plt.savefig(f'./analysis/{title}.png')

    plt.show()

#Ainput  -> B output

#A -> B -> C
#     B
#A -> 
#     C -> D

if __name__ == '__main__':

    synapse_type = 'count'

    input = './output/new_pipeline/input_cell_to_cell_changes.csv'
    output = './output/new_pipeline/output_cell_to_cell_changes.csv'

    kernel_density_plot(input, 'Input Correlation Coefficients')
    kernel_density_plot(output, 'Output Correlation Coefficients')

    histogram(input, 'Input Correlation Coefficients', synapse_type= synapse_type)
    histogram(output, 'Output Correlation Coefficients', synapse_type= synapse_type)

    scatter_plotter(input, 'Input Correlation Coefficients', synapse_type= synapse_type)
    scatter_plotter(output, 'Output Correlation Coefficients',synapse_type= synapse_type)

    # input_sig = './output/new_pipeline/input_cell_to_cell_changes_p<0.05.csv'
    # input_low = './output/new_pipeline/input_cell_to_cell_changes_p>0.05.csv'

    # output_sig = './output/new_pipeline/output_cell_to_cell_changes_p<0.05.csv'
    # output_low = './output/new_pipeline/output_cell_to_cell_changes_p>0.05.csv'

    # scatter_plotter(input_sig, 'Input Correlation Coefficients of pvalue < 0.05')
    # scatter_plotter(input_low, 'Input Correlation Coefficients of pvalue > 0.05')

    # scatter_plotter(output_sig, 'Output Correlation Coefficients of pvalue < 0.05')
    # scatter_plotter(output_low, 'Output Correlation Coefficients of pvalue > 0.05')

    

# fig, ax = plt.subplots(figsize = (10, 6))
# ax.scatter(x = df["Pearson's correlation"], y = df["Spearman's correlation"])

# plt.title(title, loc = 'center')
# plt.xlabel("Pearsons's Correlation Coefficient")
# plt.ylabel("Spearman's Correlation Coefficient")

# plt.show()


