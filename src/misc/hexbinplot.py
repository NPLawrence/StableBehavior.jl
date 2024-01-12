import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)

df = pd.read_csv("./experiments/KEEP-discrete-PID-batch-2023_08_08_21_38_52/PID_param_project.csv")
df2 = pd.read_csv("./experiments/KEEP-discrete-PID-batch-2023_08_08_21_38_52/PID_param.csv")
K = pd.read_csv("./experiments/KEEP-discrete-PID-batch-2023_08_08_21_38_52/boundary.csv")

# print(K.head())

# df.reset_index(drop=True, inplace=True)
# df2.reset_index(drop=True, inplace=True)
# df.rename(columns={ df.columns[0]: "kp", df.columns[1]: "ki" }, inplace = True)

# df['mode'] = np.ones(1010)
# print(df.head())



# plot.hexbin(x='kp', y='ki', C=np.ones(1010),
#                     reduce_C_function=np.sum,
#                     gridsize=10,
#                     cmap="viridis")

# ax = df2.plot.hexbin(x=0, y=1, C=2*np.ones(1010))

dfplot = pd.DataFrame({
    'coord_x': pd.concat([df.iloc[0], df2.iloc[0]]), 
    'coord_y': pd.concat([df.iloc[1], df2.iloc[1]]),
    'observations': np.concatenate((np.ones(len(df.iloc[0])), -1*np.ones(len(df2.iloc[0]))))
    })


def my_C_function(C):
    s = np.sum(C)
    return np.sign(s)*np.log1p(np.abs(s))


fig, ax = plt.subplots(1, 1)

dfplot.plot.hexbin(x='coord_x',
                    y='coord_y',
                    C='observations',
                    gridsize=20,
                    reduce_C_function=my_C_function,
                    cmap="RdBu",
                    xlim=(-1.0, 1.6),
                    ylim=(-0.2, 0.75),
                    colorbar=False,
                    ax=ax)

K.plot(x='x1', y='x2', color='gray', kind='line', legend=False, xlabel=r'$k_p$', ylabel=r'$k_i$', ax=ax)
plt.hlines(0, -1.1, 2.1, color='gray')
ax.set_axisbelow(True)
ax.grid(color='gray', alpha=0.15)

plt.rcParams.update({'font.size': 30})

# ax.annotate("Stable", (1.0, -0.125))
# ax.annotate("Unstable", (-0.5, 0.5))

    # plt = plot(annotations=[(-0.5, 0.5, "Unstable"), (1.0, -0.1, "Stable")], 

# import tikzplotlib
# tikzplotlib.save('./experiments/KEEP-discrete-PID-batch-2023_08_08_21_38_52/figures/figure.pgf')
plt.savefig('./experiments/KEEP-discrete-PID-batch-2023_08_08_21_38_52/figures/figure.pdf', backend='pgf')

plt.show()