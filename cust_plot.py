import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def plot(plt_data, plt_avg=None, plt_off=0, plt_subtitle='', figsize=(20,24), loc='best'):
    _, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize, facecolor='w')
    def_plot = lambda ax, fn: fn(ax=ax, plt_data=plt_data, plt_off=plt_off, plt_avg=plt_avg, plt_subtitle=plt_subtitle, loc=loc)
    def_plot(ax[0,0], plot_rewards)
    def_plot(ax[0,1], plot_success)
    def_plot(ax[1,0], plot_mu)
    def_plot(ax[1,1], plot_sig)
    plot_pdf(ax[2,0], plt_data, plt_subtitle=plt_subtitle)


def plot_rewards(ax, plt_data, plt_avg=None, plt_off=0, plt_subtitle='', loc='best', **kwargs):
    y = plt_data['rewards']
    plt_title = 'Rewards vs. batches'
    ylab = 'Navigation efficiency ratio'
    _default_cust_plot(ax=ax, plt_data=plt_data, y=y, plt_off=plt_off, plt_avg=plt_avg, plt_title=plt_title, plt_subtitle=plt_subtitle, ylab=ylab, loc=loc)


def plot_success(ax, plt_data, plt_avg=None, plt_off=0, plt_subtitle='', loc='best', **kwargs):
    y = plt_data['success']
    plt_title = 'Success ratio vs. batches'
    ylab = 'Success ratio'
    _default_cust_plot(ax=ax, plt_data=plt_data, y=y, plt_off=plt_off, plt_avg=plt_avg, plt_title=plt_title, plt_subtitle=plt_subtitle, ylab=ylab, loc=loc)


def plot_mu(ax, plt_data, plt_off=0, plt_subtitle='', loc='best', **kwargs):
    y = plt_data['mu']
    plt_title = 'Mean criteria weight vs. batches'
    ylab = 'Mu'
    _default_cust_plot(ax=ax, plt_data=plt_data, y=y, plt_off=plt_off, plt_title=plt_title, plt_subtitle=plt_subtitle, ylab=ylab, loc=loc)


def plot_sig(ax, plt_data, plt_off=0, plt_subtitle='', loc='best', **kwargs):
    y = plt_data['sig']
    plt_title = 'Standard deviation for criteria weight vs. batches'
    ylab = 'Sigma'
    _default_cust_plot(ax=ax, plt_data=plt_data, y=y, plt_off=plt_off, plt_title=plt_title, plt_subtitle=plt_subtitle, ylab=ylab, loc=loc)


def plot_pdf(ax, plt_data, plt_subtitle='', loc='center right', **kwargs):
    _cust_plot_pdf(ax, np.array(plt_data['mu'])[:,-1], np.array(plt_data['sig'])[:,-1], xlab='Weight', ylab='Probability', labels=plt_data['fns'],
        title=f"Probability density function for criteria weights{plt_subtitle}", loc=loc)


def _default_cust_plot(ax, plt_data, y, plt_off=0, plt_avg=None, plt_title='', plt_subtitle='', xlab='Batches', ylab='', loc='best'):
    x = np.arange(len(plt_data['rewards'])) + 1
    labels = plt_data['fns']
    _cust_plot(ax=ax, x=x, y=y, title=f'{plt_title}{plt_subtitle}', xlab=xlab, ylab=ylab, labels=labels, off=plt_off, avg=plt_avg, loc=loc)


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
             x_avg, y_avg = _move_avg(yi, avg)
             ax.plot(x_avg + xi[0], y_avg, label=f'{label} {avg} point avg')
    if labels:
        ax.legend(loc=loc)
    return ax


def _cust_plot_pdf(ax, mu, sig, title=None, xlab=None, ylab=None, labels=None, loc='center right'):
    xmin = (mu - 3 * sig).min()
    xmax = (mu + 3 * sig).max()
    count = abs((xmax - xmin) / 1000)
    x = np.arange(xmin, xmax, count)
    y = [norm.pdf(x, mu[i], sig[i]) for i in range(len(mu))]
    return _cust_plot(ax=ax, x=x, y=y, title=title, xlab=xlab, ylab=ylab, labels=labels, loc=loc)


def _move_avg(y, p):
    if len(y) < p:
        return np.array([]), np.array([])
    c = np.cumsum(y)
    y_avg = (c[p:] - c[:-p]) / p
    x_avg = np.arange(len(y_avg)) + p
    return x_avg, y_avg