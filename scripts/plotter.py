import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

def histogram(path, title, plot = plot, synapse_type = 'count'):

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

def multi_histogram(data1, data2, data3):

    timepoints = np.array([4.3, 16, 23, 27, 50])
    xlabel = 'Predited hours after birth'
    ylabel = 'Number of daf2 dauer connections'
    title = 'Neural age prediction based on different normalization methods'

    plt.figure(figsize=(8,6))
    plt.hist(data1, bins=20, alpha=0.5, label=data1.columns[0])
    plt.hist(data2, bins=20, alpha=0.5, label=data2.columns[0])
    plt.hist(data3, bins=20, alpha=0.5, label=data3.columns[0])

     
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


