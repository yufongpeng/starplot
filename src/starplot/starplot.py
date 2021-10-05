from scipy.stats import ttest_ind, ttest_rel, median_test, wilcoxon, bartlett, levene, fligner
from scipy.stats import f as ftest
from scipy.stats.mstats import mannwhitneyu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

class StarPlot:
    @classmethod
    def stats(cls, data, groupby = None, control = None, plottype = None, mode = 'circular', **kwargs):
        sp = cls(data, groupby = groupby, control = control, plottype = plottype, mode = mode, **kwargs)
        # pval_cen, pval_var, level
        sp.test()
        return sp

    @classmethod
    def base(cls, data, groupby = None, control = None, plottype = None, mode = 'circular', **kwargs):
        sp = cls(data, groupby = groupby, control = control, plottype = plottype, mode = mode, **kwargs)
        sp.test()
        sp.baseplot()
        return sp

    @classmethod
    def star(cls, data, groupby = None, control = None, plottype = None, mode = 'circular', **kwargs):
        sp = cls(data, groupby = groupby, control = control, plottype = plottype, mode = mode, **kwargs)
        sp.test()
        sp.plot()
        return sp

    def __init__(self, data, groupby = None, control = None, plottype = None, mode = 'circular', **kwargs):
        include = kwargs.pop('include', {})
        exclude = kwargs.pop('exclude', {})
        self.mode = mode
        self.plottype = plottype
        self.stats_kw = dict(stats_cen = 'independent t test', test_var = True, stats_var = 'f test', 
                    crit_cen = [0.05,0.01,0.001,0.0001], crit_var = 0.05) 
        self.plot_kw = dict(errorbar = True, stars = ['*','**','***','****'], rotate = 0, linewidth = 1, elinewidth = 0.5, 
                            fontsize = 14, capsize = 4, starsize = 10, cutoff = None, footing = None, 
                            noffset = [10, 10], noffset_blank = 2, noffset_border = [10, 35], noffset_star = 1)
        for k, v in kwargs.items():
            if k in self.stats_kw:
                self.stats_kw[k] = v
            else:
                self.plot_kw[k] = v


        if not self.plottype:
            if self.stats_kw['stats_cen'] in ['median test', 'mannwhitneyu', 'wilcoxon']:
                self.plottype = 'box'
            else:
                self.plottype = 'bar'

        self.data = data
        self.groupby = groupby
        self.control = control
        # groups, features, size
        self._extract(include, exclude)
        # center, error
        self._set_cen_err()
    

    def _extract(self, include, exclude):
        # exclude group column, splice groups
        if self.groupby:
            self.data = self.data.set_index(self.groupby)
        if include.get('feature', {}):
            # If specific features are included
            features = [feature for feature in self.data.columns if feature in include['feature']]
        else:
            if exclude.get('feature',{}):
                # Exclude specific features
                features = [feature for feature in self.data.columns if feature not in exclude['feature']]
            else:
                features = self.data.columns
        self.features = pd.Index(features)
        self.features.name = self.data.columns.name
        # Same logic for group
        if include.get('group',{}):
            groups = [group for group in self.data.index.unique() if group in include['group']]
        else:
            if exclude.get('group',{}):
                groups = [group for group in self.data.index.unique() if group not in exclude['group']]
            else:
                groups = self.data.index.unique()
        self.groups = pd.Index(groups)
        self.groups.name = self.data.index.name
        self.size = (len(self.features), len(self.groups))
    
    def _set_cen_err(self):
        # stats
        error = np.ones(self.size)
        center = np.ones(self.size)
        # Try vectorizing ?
        if self.stats_kw['stats_cen'] in ['median test', 'mannwhitneyu', 'wilcoxon']:
            # Use median and iqr for nonparametric test
            for igroup, group in enumerate(self.groups):
                error[:, igroup] = self.data.loc[group, self.features].quantile(0.75) - self.data.loc[group, self.features].quantile(0.25)
                center[:, igroup] = self.data.loc[group, self.features].median()
        else:
            for igroup, group in enumerate(self.groups):
                error[:, igroup] = self.data.loc[group, self.features].std()
                center[:, igroup] = self.data.loc[group, self.features].mean()
        self.center = pd.DataFrame(center, columns=self.groups, index=self.features)
        self.error = pd.DataFrame(error, columns=self.groups, index=self.features)
        if self.control:
            div = self.center[self.control].copy()
            for group in self.groups:
                self.center[group] = self.center[group]/div
                self.error[group] = self.error[group]/div
                
    @staticmethod
    def _test_var(stats, group1, group2):
        if stats == 'f test':
            return 0.5-abs(0.5-ftest.sf(group1.var()/group2.var(), group1.shape[0]-1, group2.shape[0]-1))
        else:
            if stats == 'bartlett':
                return [bartlett(group1.iloc[:, ifeature], group2.iloc[:, ifeature])[1] for ifeature in range(group1.shape[1])]
            elif stats == 'levene':
                return [levene(group1.iloc[:, ifeature], group2.iloc[:, ifeature])[1] for ifeature in range(group1.shape[1])]
            elif stats == 'fligner':
                return [fligner(group1.iloc[:, ifeature], group2.iloc[:, ifeature])[1] for ifeature in range(group1.shape[1])]

    @staticmethod
    def _test_cen(stats, group1, group2, equal_var):
        if stats == 'independent t test':
            return [ttest_ind(group1.iloc[:, ifeature], group2.iloc[:, ifeature], equal_var = equal_var[ifeature])[1] for ifeature in range(group1.shape[1])]
        elif stats == 'paired t test':
            return ttest_rel(group1, group2)[1]
        elif stats == 'median test':
            return [median_test(group1.iloc[:, ifeature], group2.iloc[:, ifeature])[1] for ifeature in range(group1.shape[1])]
        elif stats == 'mannwhitneyu':
            return [mannwhitneyu(group1.iloc[:, ifeature], group2.iloc[:, ifeature])[1] for ifeature in range(group1.shape[1])]
        elif stats == 'wilcoxon':
            return [wilcoxon(group1.iloc[:, ifeature], group2.iloc[:, ifeature])[1] for ifeature in range(group1.shape[1])]
        
    def test(self, **kwargs):    
        for k, v in kwargs.items():
            self.stats_kw[k] = v
        if self.mode == 'circular':
            pval_cen = np.ones(self.size).transpose()
            pval_var = np.ones(self.size).transpose()
            level = np.ones(self.size).transpose()
            id = []
            layer = np.ones(self.size[1])
            for igroup, group in enumerate(self.groups):
                nextgroup = (igroup+1)%self.size[1]
                id.append((igroup, nextgroup))
                if self.stats_kw['test_var']:
                    pval_var[igroup, :] = self._test_var(self.stats_kw['stats_var'], self.data.loc[group, self.features], self.data.loc[self.groups[nextgroup], self.features])
                    equal_var = self.stats_kw['crit_var'] < pval_var[:, igroup]
                else:
                    equal_var = [True for i in range(self.size[0])]
                pval_cen[igroup, :] = self._test_cen(self.stats_kw['stats_cen'], self.data.loc[group, self.features], self.data.loc[self.groups[nextgroup], self.features], equal_var)
                crit = np.array(self.stats_kw['crit_cen']) 
                level[igroup, :] = [len(crit) - len(crit.compress(ip > crit)) for ip in pval_cen[igroup, :]] # 0 == nonsignificant
                layer[igroup] = (igroup + 1)//self.size[1]
            self.pval_cen = pd.DataFrame(pval_cen, columns=self.features, index=id)
            self.pval_var = pd.DataFrame(pval_var, columns=self.features, index=id)
            self.level = pd.DataFrame(level, columns=self.features, index=id)
            self.level['layer'] = layer

    def new_plot(self):
        plt.rcParams['font.family'] = 'Times New Roman'
        self.fig, self.ax = plt.subplots()

    def drawpatches(self, **kwargs):
        self._parse_plot_kw(**kwargs)
        if self.plottype == 'bar':
            if self.plot_kw['errorbar']:
                self.center.plot.bar(yerr = self.error, ax = self.ax, rot = self.plot_kw['rotate'], capsize = self.plot_kw['capsize'], 
                                    error_kw = dict(elinewidth = self.plot_kw['elinewidth']), fontsize = self.plot_kw['fontsize'])
                max_val = (self.center+self.error).values
                min_val = (self.center-self.error).values
            else:
                self.center.plot.bar(ax = self.ax, rot = self.plot_kw['rotate'], capsize = self.plot_kw['capsize'],
                                    fontsize = self.plot_kw['fontsize'])
                max_val = self.center.values
                min_val = self.center.values
            yaxismin = min(0, min_val.min())
        elif self.plottype == 'box':
            df = self.data.stack().reset_index() 
            gn, cn, vn = df.columns       
            sns.boxplot(x = cn, y = vn, hue = gn, data = df)
            temp = [np.concatenate([l.get_ydata() for l in self.ax.lines[(6*i):(6*i+6)]]).flatten() for i in range(len(self.ax.lines)//6)]
            max_val = np.array([d.max() for d in temp]).reshape(self.size)
            min_val = np.array([d.min() for d in temp]).reshape(self.size) 
            yaxismin = min_val.min()

        offset = (max_val.max() - yaxismin)/100 
        self._patches_data = (offset, max_val, yaxismin)
        return self.fig

    def drawlines(self, **kwargs):
        self._parse_plot_kw(**kwargs)
        offset, max_val, yaxismin = self._patches_data
        noffset = self.plot_kw['noffset']
        x = self._find_x()
        for ifeature in range(self.size[0]):
            ilayer = 0
            while ilayer <= self.level['layer'].max():
                df = self.level.loc[self.level.layer == ilayer].iloc[:,  [ifeature]]
                df = df.loc[df.iloc[:, 0] > 0]
                df.insert(0, 'max_val', [tuple(np.sort(max_val[ifeature, groups])) for groups in df.index])
                df = df.sort_values(by = 'max_val')
                if self.plot_kw['footing'] is not None:
                    max_val[ifeature, :] = [max(self.plot_kw['footing'][ifeature], i) for i in max_val[ifeature, :]]
                for groups in df.index:
                    # draw
                    ymin = max_val[ifeature, groups] + offset*self.plot_kw['noffset_blank']
                    ymax = max_val[ifeature, min(groups):(max(groups)+1)].max() + noffset[ilayer]*offset
                    if self.plot_kw['cutoff'] is not None:
                        ymin = [max(i, self.plot_kw['cutoff'][ifeature]) for i in ymin]
                    xs = x[ifeature,  groups]
                    self.ax.vlines(x = xs[0], ymin = ymin[0], ymax = ymax, lw = self.plot_kw['linewidth'])
                    self.ax.vlines(x = xs[1], ymin = ymin[1], ymax = ymax, lw = self.plot_kw['linewidth'])
                    self.ax.hlines(xmin = xs.min(), xmax = xs.max(), y = ymax, lw = self.plot_kw['linewidth'])
                    self.ax.annotate(self.plot_kw['stars'][int(df.loc[df.index == groups].iloc[:, 1]-1)], xy = (xs.mean(), ymax+offset*self.plot_kw['noffset_star']), ha = 'center', size = self.plot_kw['starsize'])
                    # update max_val
                    max_val[ifeature, groups] = ymax
                ilayer += 1
        self.ax.set_ylim(yaxismin - self.plot_kw['noffset_border'][0]*offset, max_val.max() + self.plot_kw['noffset_border'][1]*offset)
        return self.fig
    
    def _find_x(self):
        if self.plottype == "bar":
            return np.array([p.get_x() + p.get_width()/2 for p in self.ax.patches]).reshape(self.size).transpose()
        elif self.plottype == "box":
            return np.array([p.get_xdata()[0] for p in self.ax.lines[0::6]]).reshape(self.size)

    @staticmethod
    def _scaler2vector(obj, n):
        if obj is None or isinstance(obj, list):
            return obj
        else:
            return np.repeat([obj], n)

    def _parse_plot_kw(self, **kwargs):
        for k, v in kwargs.items():
            self.plot_kw[k] = v
        self.plot_kw['footing'] = self._scaler2vector(self.plot_kw['footing'], self.size[0])
        self.plot_kw['cutoff'] = self._scaler2vector(self.plot_kw['cutoff'], self.size[0])

    def baseplot(self, **kwargs):
        self.new_plot()
        self.drawpatches(**kwargs)

    def plot(self, **kwargs):
        self.new_plot()
        self.drawpatches(**kwargs)
        self.drawlines()
        
