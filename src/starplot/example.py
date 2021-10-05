from starplot import StarPlot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame({'value': np.log([0.000001,1,100,0.00005,2,25,0.0000015,4,150,0.0000073,5,110,151,56,45,152,566500,455,175,47,789,454,45,4852,7120,0.001,0.05,3500,0.01,0.01,4205,0.05,0.5,452,0.7,0.4]),
                    'compound': np.repeat([['A','B','C'],], 12, axis = 0).flatten(), 
                    'group': np.repeat(['control', 'g1', 'g2'], 12)})

bp = sns.boxplot(x = 'compound', y = 'value', hue = 'group', data = df)
plt.show()

for l in bp.lines:
    print(l.get_xydata()[:,1])

"""
    *   ---6
   ___  ---4
    |   ---2
   _|_  
  |_ _| ---5
  |_ _| 
    |   ---1
   _|_  ---3
    *   ---6
"""

df['id'] = [i//3 for i in range(len(df['group']))]
dfo = df.set_index(['id','compound','group']).unstack(-2).reset_index(0).iloc[:,[1,2,3]]  
dfo.columns = pd.Index(['A','B','C'], name = 'compound') 

sp = StarPlot.star(dfo, stats_cen = 'mannwhitneyu', starsize = 10)

sp = StarPlot.base(dfo, stats_cen = 'mannwhitneyu')
sp.drawlines(footing = [12,15,12], starsize = 15)
sp.plot(footing = [12,15,12], starsize = 12, noffset = [15, 15])
sp.ax.set_ylabel('log values')
sp.fig

sp = StarPlot.star(dfo+20)