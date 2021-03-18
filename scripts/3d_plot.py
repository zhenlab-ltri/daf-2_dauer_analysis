import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots

def plotter3d(file,name, outpath):
    # Get Data
    df = pd.read_csv(file)

    #Add a column called connection
    df['Connection'] = df["Pre"] + '$' + df["Post"]

    #Make connection column first
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]


    fig = go.Figure(data=go.Scatter3d(
        x=df['effect size'],
        y=df['in how many N2 dauers'],
        z=df['p-value'],
        text=df['Connection'],
        mode='markers',
        opacity= 0.8,
        hovertemplate =
        '<b>%{text} </b>' +
        '<br>Effect Size: %{x}%'+
        '<br># N2 dauer: %{y}'+
        '<br><i>p-value</i>: %{z}',
        showlegend = False,
        marker=dict(
            sizemode='diameter',
            size=4,
    
        )

    ))

    fig.update_layout(scene_zaxis_type="log",
                    height=1000, width=1000,
                    title=name,
                    scene = dict(xaxis=dict(title='Effect Size (x)'),
                                yaxis=dict(title='In how many N2 dauers (y)'),
                                zaxis=dict(title='P-value (z)'),
                                
                            ))

    fig.update_layout(scene_aspectmode='manual',
                    scene_aspectratio=dict(x=3, y=1, z=1))

    fig.update_layout(scene_aspectmode='auto')

    fig.show()
    fig.write_html(outpath)

if __name__ == '__main__':
    input_csv = 'daf2-dauer-comparisons\output\input_changes.csv'
    output_csv = 'daf2-dauer-comparisons\output\output_changes.csv'
    
    input_outpath = 'daf2-dauer-comparisons\output\input_changes.html'
    output_outpath = 'daf2-dauer-comparisons\output\output_changes.html'

    plotter3d(input_csv, 'Examining Major Input Connection Changes', input_outpath)
    plotter3d(output_csv, 'Examining Major Output Connection Changes', output_outpath)

    