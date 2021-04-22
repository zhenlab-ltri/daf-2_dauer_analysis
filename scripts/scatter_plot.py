import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px

input = 'daf2-dauer-comparisons\output\modified_pipeline\input_cell_to_cell_changes.csv'
title = 'Input Correlation Coefficients'

df = pd.read_csv(input)

fig = px.scatter(df, x= "Pearson's correlation", y = "Spearman's correlation",
                    hover_data= ['Pre', 'Post', 'classification', 'nondauer contact'])

#center title
fig.update_layout(title_text= title, title_x=0.5)

fig.show()

fig.write_html('daf2-dauer-comparisons\output\modified_pipeline\input_plotly')

# fig, ax = plt.subplots(figsize = (10, 6))
# ax.scatter(x = df["Pearson's correlation"], y = df["Spearman's correlation"])

# plt.title(title, loc = 'center')
# plt.xlabel("Pearsons's Correlation Coefficient")
# plt.ylabel("Spearman's Correlation Coefficient")

# plt.show()