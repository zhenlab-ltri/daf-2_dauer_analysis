# import scipy as sc
# import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt

# from scipy import stats
# from matplotlib_venn import venn3

# # timepoints = [4.3, 16, 23, 27, 50, 50]
# # values = [0,2,4.4,6,8,10]

# # pearsons = sc.stats.pearsonr(timepoints, values)
# # print(pearsons)

# # v = np.array((0,0,0,5))
# # v = np.append(v, [5.6, 4.6])
# # print(v)

# # mad = sc.stats.median_abs_deviation(values)
# # print(mad)


# plt.figure(figsize=(4,4))
# input = set([1,4,5,0,7])
# output = set([9,4,6,2,7])
# total = set([0,3,4,3,9,5])

# df.loc[df['column_name'].isin(some_values)]



# venn3([input, output, total], ('Input', 'Output', 'Total'))
# print((input - (output | total)) )
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create toy data 
x = np.linspace(0, 10, 20)
y = x + (np.random.rand(len(x)) * 10)

# Extend x data to contain another row vector of 1s
X = np.vstack([x, np.ones(len(x))]).T
print(X)

plt.figure(figsize=(12,8))
for i in range(0, 1):
    sample_index = np.random.choice(range(0, len(y)), len(y))

    X_samples = X[sample_index]
    y_samples = y[sample_index]  
    print(X_samples.shape)
    print(y_samples.shape)  

    lr = LinearRegression()
    lr.fit(X_samples, y_samples)
    plt.plot(x, lr.predict(X), color='grey', alpha=0.2, zorder=1)

plt.scatter(x,y, marker='o', color='orange', zorder=4)

lr = LinearRegression()
lr.fit(X, y)
plt.plot(x, lr.predict(X), color='red', zorder=5)




