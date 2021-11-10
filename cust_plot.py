import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def _cust_plot(ax, x, y, title=None, xlab=None, ylab=None, labels=None, off=0, avg=None, loc='best'):
    inst = lambda M: any(isinstance(M, j) for j in [list, np.ndarray])
    get = lambda M: (lambda i: M[i]) if inst(M[0])  else (lambda i: M)
    len2d = lambda M: len(M) if inst(M[0]) else 1
    getx = get(x)
    gety = get(y)
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    for i in range(len2d(y)):
        yi = gety(i)[off:]
        xi = getx(i)[off:]
        label = labels[i] if labels else None
        lines = ax.plot(xi, yi, label=label)
        if i >= 10:
            lines[0].set_linestyle('dashed')
        if avg:
             x_avg, y_avg = move_avg(yi, avg)
             ax.plot(x_avg + xi[0], y_avg, label=f'{label} {avg} point avg')
    if labels:
        ax.legend(loc=loc)
    return ax


def _cust_plot_pdf(ax, mu, sig, title=None, xlab=None, ylab=None, labels=None):
    xmin = (mu - 3 * sig).min()
    xmax = (mu + 3 * sig).max()
    count = abs((xmax - xmin) / 1000)
    x = np.arange(xmin, xmax, count)
    y = [norm.pdf(x, mu[i], sig[i]) for i in range(len(mu))]
    return _cust_plot(ax=ax, x=x, y=y, title=title, xlab=xlab, ylab=ylab, labels=labels, loc='center right')


def move_avg(y, p):
    if len(y) < p:
        return np.array([]), np.array([])
    c = np.cumsum(y)
    y_avg = (c[p:] - c[:-p]) / p
    x_avg = np.arange(len(y_avg)) + p
    return x_avg, y_avg


def plot(plt_data, plt_avg=None, plt_off=0, plt_subtitle='', figsize=(20,24)):
    _, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize, facecolor='w')
    len_rewards = len(plt_data['rewards'])
    x = np.arange(len_rewards) + 1
    fn_labs = plt_data['fns']
    def_plot = lambda ax, y, ylab, title, labels=None, avg=None: _cust_plot(ax, x, y, xlab='Batches', ylab=ylab, labels=labels, off=plt_off, title=f'{title}{plt_subtitle}', avg=avg)
    def_plot(ax[0,0], plt_data['rewards'],   ylab='Navigation efficiency ratio',         title='Rewards vs. batches',                                labels=['rewards'], avg=plt_avg)
    def_plot(ax[0,1], plt_data['success'],   ylab='Success ratio',                       title='Success ratio vs. batches',                          labels=['success'], avg=plt_avg)
    def_plot(ax[1,0], plt_data['mu'],        ylab='Mu',                                  title='Mean criteria weight vs. batches',                   labels=fn_labs)
    def_plot(ax[1,1], plt_data['sig'],       ylab='Sigma',                               title='Standard deviation for criteria weight vs. batches', labels=fn_labs)
    _cust_plot_pdf(ax[2,0], np.array(plt_data['mu'])[:,-1], np.array(plt_data['sig'])[:,-1], xlab='Weight', ylab='Probability', labels=fn_labs,
        title=f"Probability density function for criteria weights{plt_subtitle}")