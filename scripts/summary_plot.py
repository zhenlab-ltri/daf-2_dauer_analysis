import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from seaborn.miscplot import palplot

from functools import reduce

def plot_median (df, colour):
    labels = [e.get_text() for e in plt.gca().get_xticklabels()]
    ticks = plt.gca().get_xticks()
    
    w = 0.1
    for idx, method in enumerate(labels):
        idx = labels.index(method)
        plt.hlines(df[df['Normalization'] == method]['Neural Age'].mean(), ticks[idx]-w, ticks[idx]+w, color = colour, zorder = 10)

def subplot_median (df, ax, colour):
    labels = [e.get_text() for e in ax.get_xticklabels()]
    ticks = ax.get_xticks()
    
    w = 0.2
    for idx, method in enumerate(labels):
        idx = labels.index(method)
        ax.hlines(df[df['Normalization'] == method]['Neural Age'].mean(), ticks[idx]-w, ticks[idx]+w, color = colour, zorder = 10)

compare = 'L3'

df = pd.read_csv(f'./graphs/{compare}/nonparametric_bootstrapping/neural_age_info_summary.csv')

#changing slope from value to category
df['slope'] = np.where( df['slope'] > 0 , 'positive', 'negative')

#masks for dividing neural age into ones completely inside, completely outside, and somewhere inside the nondauer range
inside_mask = (df['low_bound']>= 0) & (df['high_bound'] <= 50)
outside_mask = ((df['low_bound']<= 0) & (df['high_bound'] <= 0)|(df['low_bound']>= 50) & (df['high_bound'] >= 50))

inside= df[inside_mask]
outside = df[outside_mask]
between = df[~outside_mask & ~inside_mask]

#Add range 
inside['range'] = 'inside'
outside['range'] = 'outside'
between['range'] = 'between'

dfs = [inside, outside, between]
df_final = pd.concat(dfs)

df_final.to_csv('./output/neural_age_by_normalization.csv', index = False)

#inside and between
inb = df_final[~outside_mask]

#outside and between
onb = df_final[~inside_mask]

cpalette = {"stable": "C0", "variable": "C1", "developmentally dynamic (weakened)": "C02", "developmentally dynamic (strengthened)": "C03", "post-embryonic brain integration": "C04"}
sns.set_style("ticks")

# g = sns.catplot(
#     x="Normalization", 
#     y="Neural Age", 
#     hue="classification", 
#     col="slope", 
#     data=inside, 
#     kind="strip",
#     palette = palette,
#     height=4, 
#     aspect=.7);

#color palette for each iteration of the image
rpalette_1 = {'inside': 'w', 'outside': 'w', 'between': 'w'}
rpalette_2 = {'inside': 'r', 'outside': 'w', 'between': 'w'}

#rpalette_2 = {'inside': 'r', 'outside': '#1b9e77', 'between': '#7570b3'}

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

# code to make sure the same figure is generated for each run 
for x in range(3):
    
    np.random.seed(123)

    catagories = []
    values = []

    for i in range(0,200):
        n = np.random.randint(1,3)
        catagories.append(n)

    for i in range(0,200):
        n = np.random.randint(1,100)
        values.append(n)

    col = x % 3
    axcurr = axes[col]

    all_yaxis = np.arange(-75, 126, 25)
    nondauer_yaxis = np.arange(0, 51, 5)
    extraticks = [4.3, 16.0, 23.0, 27.0]

    if x == 1:
        ax1 = sns.stripplot(
            x = 'Normalization', 
            y = 'Neural Age',
            hue = 'range',
            edgecolor='black',
            palette = rpalette_1,
            linewidth= 1, 
            data = df_final, 
            size = 4, 
            zorder = 5,
            dodge=False,
            ax =axcurr,
            alpha=0.7)
        ax1.get_legend().remove()

        #Adding early L1, late L1, L2, L3, as extra tickmarks 
        ax1.set_yticks(list(all_yaxis) + extraticks)
        old_labels = list(ax1.get_yticks())
        new_labels = []
        for label in old_labels:
            if float(label) in extraticks:
                new_labels.append('')
            else:
                new_labels.append(label)
        
        ax1.set_yticklabels(new_labels)
        
        [t.set_color('red') for t in ax1.xaxis.get_ticklines()]
        

    elif x == 2:
        ax2 = sns.stripplot(
            x = 'Normalization', 
            y = 'Neural Age',
            hue = 'range',
            edgecolor='black',
            palette = rpalette_2,
            linewidth= 1, 
            data = df_final, 
            size = 4, 
            zorder = 5,
            dodge=False,
            ax =axcurr,
            alpha=0.7)
        

        #Adding early L1, late L1, L2, L3, as extra tickmarks 
        ax2.set_yticks(list(all_yaxis) + extraticks)
        old_labels = list(ax2.get_yticks())
        new_labels = []
        for label in old_labels:
            if float(label) in extraticks:
                new_labels.append('')
            else:
                new_labels.append(label)
        
        ax2.set_yticklabels(new_labels)
        ax2.legend(bbox_to_anchor=(1, 1), loc='upper left')
    
    else:
        ax3 = sns.stripplot(
            x = 'Normalization', 
            y = 'Neural Age',
            edgecolor='black',
            hue = 'slope',
            
            linewidth= 1, 
            data = inside, 
            size = 4, 
            zorder = 5,
            dodge=False,
            ax =axcurr,
            alpha=0.7)

        subplot_median(inside,ax3,'black')
        
        #Adding early L1, late L1, L2, L3, as extra tickmarks 
        ax3.set_yticks(list(nondauer_yaxis)+ extraticks)
        old_labels = list(ax3.get_yticks())
        new_labels = []
        for label in old_labels:
            if float(label) in extraticks:
                new_labels.append('')
            else:
                new_labels.append(label)
        
        ax3.set_yticklabels(new_labels)
    
    
    plt.tight_layout()




plt.figure()
ax4 = plt.figsize=(8, 6)
# sns.stripplot(
#     x = 'Normalization', 
#     y = 'Neural Age',
#     edgecolor='black',
#     palette = ['r'],
#     linewidth= 1, 
#     data = inside, 
#     size = 4, 
#     zorder = 5,
#     jitter= True,
#     dodge=False,
#     alpha=0.7)

#plot_median(df, 'grey')
plot_median(inside, 'black')
#plot_median(inb, 'red')

# sns.stripplot(
#     x = 'Normalization', 
#     y = 'Neural Age', 
#     hue = 'classification', 
#     data = outside, 
#     size = 4, 
#     marker = 'X',
#     palette=palette,
#     zorder = 5)

sns.stripplot(
    x = 'Normalization', 
    y = 'Neural Age', 
    hue = 'classification', 
    data = between, 
    size = 4, 
    zorder = 5)

# extraticks = [4.3, 16, 23, 27, 50]
# plt.yticks(list(plt.yticks()[0]) + extraticks)

plt.legend(bbox_to_anchor=(1.05, 1.0, 0.3, 0.2), loc='upper left')
plt.tight_layout()

plt.show()