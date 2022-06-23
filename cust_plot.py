import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, List
import torch
from utils import device


def plot(plt_data: dict, plt_avg: int = None, plt_off: int = 0, plt_subtitle: str = '', figsize: Tuple[int,int] = (20,24), loc: str = None, scaled: bool = True, zero_line: bool = False) -> None:
    """
    Plot a summary of the rewards, success ratio, mu, sig and each criteria weight's probability density function in training

    Parameters
    ----------
    plt_data : dict
        Dictionary to store data as the network progresses.
        Requires the keys: 'rewards', 'success', 'mu' and 'sig' with values being a list.
        The values for 'mu' and 'sig' should be a 2D list, with a row for each criteria in the model
    plt_avg : int, optional
        Number of points to plot for a moving average, by default None
        If None, no moving average is plotted
    plt_off : int, optional
        Start index of the plot, by default 0
    plt_subtitle : str, optional
        Optional comment to append to each axes title, by default ''
    figsize : Tuple[int,int], optional
        Size of the plot, by default (20,24)
    loc : str, optional
        Position of the legend, by default None
        See 'matplotlib.pyplot.legend'
    scaled : bool, optional
        Whether to scale data so that criteria weights represent their percentage influence instead of raw values, by default True
    zero_line : bool, optional
        Whether or not to include a line to separate at mu = 0, by default False
    """

    # Create plot
    _, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize, facecolor='w')
    def_plot = lambda ax, fn, scaled, **kwargs: fn(ax=ax, plt_data=plt_data, plt_off=plt_off, plt_avg=plt_avg, plt_subtitle=plt_subtitle, scaled=scaled, **kwargs)
    if loc is not None:
        def_plot = lambda ax, fn, scaled: def_plot(ax=ax, fn=fn, scaled=scaled, loc=loc)

    # Add rewards, success, mu, sigma and pdf
    def_plot(ax[0,0], plot_rewards, False)
    def_plot(ax[0,1], plot_success, False)
    def_plot(ax[1,0], plot_mu, scaled, zero_line=zero_line)
    def_plot(ax[1,1], plot_sig, scaled)
    plot_pdf(ax[2,0], plt_data, plt_subtitle=plt_subtitle, scaled=scaled, zero_line=zero_line)


def plot_multi_experiments(paths: List[str], plt_title: str = None, plt_avg: int = None, plt_off: int = 0, loc: str = None, scaled: bool = True, zero_line: bool = False, figsize: Tuple[int,int] = None, save_path: str = None) -> None:
    """
    Plots the training evolution of multiple experiments side by side

    Parameters
    ----------
    paths : List[str]
        Path to the data for each experiment
    plt_title : str, optional
        Header text for the plot, by default None
    plt_avg : int, optional
        Moving average size, by default None
        If None, no moving average is plotted
    plt_off : int, optional
        Start index of the plot, by default 0
    loc : str, optional
        _description_, by default None
    scaled : bool, optional
        Whether or not to scale criteria weights so that they represent their percentage contribution instead of raw values, by default True
    zero_line : bool, optional
        Whether or not to include a line to separate at mu = 0, by default False
    figsize : Tuple[int,int], optional
        Size of the plot, by default None
    save_path : str, optional
        Where to save the plot, by default None
        If None, plot will not be saved
    """

    # Create figure / axes
    if figsize is None:
        figsize = (len(paths)*10,40)
    fig, ax = plt.subplots(5, len(paths), figsize=figsize)

    # Fixed Plotting parameters
    kwargs = {'plt_avg':plt_avg, 'plt_off':plt_off, 'scaled':scaled, 'zero_line':zero_line}
    if loc is not None:
        kwargs['loc'] = loc

    # Plot each experiment
    for col, path in enumerate(paths):
        plt_data = torch.load(path, map_location=device)
        kwargs['plt_subtitle'] = f' (experiment {col+1}/{len(paths)})'
        plot_rewards(ax[0,col], plt_data=plt_data, **kwargs)
        plot_success(ax[1,col], plt_data=plt_data, **kwargs)
        plot_mu(ax[2,col], plt_data=plt_data, **kwargs)
        plot_sig(ax[3,col], plt_data=plt_data, **kwargs)
        plot_pdf(ax[4,col], plt_data=plt_data, **kwargs)
    plt.tight_layout()

    # Add title
    if plt_title is not None:
        fig.subplots_adjust(top=0.96)
        fig.suptitle(plt_title)

    # Save
    if save_path is not None:
        plt.savefig(save_path, dpi=300)


def plot_summary(ax: matplotlib.axes.Axes, paths: List[str], scaled: bool = True, alpha: float = 0.7, zero_line: bool = False, plt_subtitle: str = '') -> None:
    """
    Plot a dot-plot summary of the final criteria weights mu after a number of experiments

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    paths : List[str]
        Path to the data for each experiment
    scaled : bool, optional
        Whether or not to scale criteria weights so that they represent their percentage contribution instead of raw values, by default True
    alpha : float, optional
        Opacity of the dots in the plot, by default 0.7
    zero_line : bool, optional
        Whether or not to include a line to separate at mu = 0, by default False
    """

    # Parameters to be set during file reading
    mu = None
    fn_len = None
    fns = None

    # Extract exeriments
    for i, path in enumerate(paths):
        plt_data = torch.load(path, map_location=device)

        # Get number of functions and setup mu
        if mu is None:
            fns = plt_data['fns']
            fn_len = len(fns)
            mu = np.zeros((fn_len, len(paths)))

        # Extract final mu with fixed order across experiments
        exp_mu = [data[-1] for data in plt_data['mu']]
        for j, fn in enumerate(plt_data['fns']):
            k = fns.index(fn)
            mu[k,i] = exp_mu[j]

    # Rescale mu
    scale = _get_mu_scale(mu, scaled)
    mu *= scale

    # Plot summary of experiments
    for i in range(fn_len):
        ax.scatter(mu[i], -np.ones(len(paths)) * i, alpha=alpha, edgecolors=None, linewidths=0, linewidth=0, marker='o')
    if zero_line:
        ax.axvline(x=0, color='black', ls='--', lw=1)

    # Plot title and labels
    ax.yaxis.set_ticks(np.arange(fn_len) * -1)
    ax.set_yticks(np.arange(fn_len) * -1, plt_data['fns'])
    ax.set_xlabel('Mu')
    ax.set_title(f'Final mean criteria weights after {len(paths)} experiments' + plt_subtitle)


def plot_multi_summary(paths: List[List[str]], scaled: bool = True, alpha: float = 0.7, zero_line: bool = False, plt_title: str = None, plt_subtitle: List[str] = None, figsize: Tuple[int,int] = None, save_path: str = None):
    """
    Plot a dot-plot summary of the final criteria weights mu after a number of experiments for multiple subjects

    Parameters
    ----------
    paths : List[List[str]]
        2D list, where each row is a list of paths to the data for each experiment related to that subject
    scaled : bool, optional
        Whether or not to scale criteria weights so that they represent their percentage contribution instead of raw values, by default True
    alpha : float, optional
        Opacity of the dots in the plot, by default 0.7
    zero_line : bool, optional
        Whether or not to include a line to separate at mu = 0, by default False
    plt_title : str, optional
        Header text for the plot, by default None
    plt_subtitle : str, optional
        Text to add to the end of each individual axes title, by default None
    figsize : Tuple[int,int], optional
        Size of the plot, by default None
    save_path : str, optional
        Where to save the plot, by default None
        If None, plot will not be saved
    """

    # Create figure / axes
    if figsize is None:
        figsize = (len(paths) * 10, 8)
    fig, ax = plt.subplots(1, len(paths), figsize=figsize)

    # Plot each experiment for each subject
    for i, subj_paths in enumerate(paths):
        subj_subtitle = '' if plt_subtitle is None else plt_subtitle[i]
        plot_summary(ax=ax[i], paths=subj_paths, scaled=scaled, alpha=alpha, zero_line=zero_line, plt_subtitle=subj_subtitle)

    # Align plots and set title
    fig.subplots_adjust(wspace=.5)
    fig.subplots_adjust(top=0.93)
    fig.suptitle(plt_title)
    plt.tight_layout()

    # Save
    if save_path is not None:
        plt.savefig(save_path, dpi=300)


def plot_rewards(ax: matplotlib.axes.Axes, plt_data: dict, plt_avg: int = None, plt_off: int = 0, plt_subtitle: str = '', loc: str = 'lower right', **kwargs) -> None:
    """
    Plot the evolution of the rewards (navigation efficiency) in training

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    See 'cust_plot.plot' for more information on parameters
    """

    y = plt_data['rewards']
    plt_title = 'Rewards vs. batches'
    ylab = 'Navigation efficiency ratio'
    labels = ['rewards']
    _default_cust_plot(ax=ax, plt_data=plt_data, y=y, plt_off=plt_off, plt_avg=plt_avg, plt_title=plt_title, plt_subtitle=plt_subtitle, ylab=ylab, loc=loc, labels=labels)


def plot_success(ax: matplotlib.axes.Axes, plt_data: dict, plt_avg: int = None, plt_off: int = 0, plt_subtitle: str = '', loc: str = 'lower right', **kwargs) -> None:
    """
    Plot the evolution of the success ratio in training

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    See 'cust_plot.plot' for more information on parameters
    """

    y = plt_data['success']
    plt_title = 'Success ratio vs. batches'
    ylab = 'Success ratio'
    labels = ['success']
    _default_cust_plot(ax=ax, plt_data=plt_data, y=y, plt_off=plt_off, plt_avg=plt_avg, plt_title=plt_title, plt_subtitle=plt_subtitle, ylab=ylab, loc=loc, labels=labels)


def plot_mu(ax: matplotlib.axes.Axes, plt_data: dict, plt_off: int = 0, plt_subtitle: str = '', loc: str = 'lower left', scaled: bool = True, zero_line: bool = False, **kwargs) -> None:
    """
    Plot the evolution of mean criteria weights mu in training

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    zero_line : bool, optional
        Whether or not to include a line to separate at mu = 0, by default False
    See 'cust_plot.plot' for more information on parameters
    """

    scale = _get_mu_scale(plt_data['mu'], scaled)
    y = np.array(plt_data['mu']) * scale
    plt_title = 'Mean criteria weight vs. batches'
    ylab = 'Mu'
    _default_cust_plot(ax=ax, plt_data=plt_data, y=y, plt_off=plt_off, plt_title=plt_title, plt_subtitle=plt_subtitle, ylab=ylab, loc=loc, yzero_line=zero_line)


def plot_sig(ax: matplotlib.axes.Axes, plt_data: dict, plt_off: int = 0, plt_subtitle: str = '', loc: str = 'upper right', scaled: bool = True, **kwargs) -> None:
    """
    Plot the evolution of criteria weight standard deviations sigma in training

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    See 'cust_plot.plot' for more information on parameters
    """

    scale = _get_mu_scale(plt_data['mu'], scaled)
    y = np.array(plt_data['sig']) * scale
    plt_title = 'Standard deviation for criteria weight vs. batches'
    ylab = 'Sigma'
    _default_cust_plot(ax=ax, plt_data=plt_data, y=y, plt_off=plt_off, plt_title=plt_title, plt_subtitle=plt_subtitle, ylab=ylab, loc=loc)


def plot_pdf(ax: matplotlib.axes.Axes, plt_data: dict, plt_subtitle: str = '', loc: str = 'center right', scaled: bool = True, zero_line: bool = False, **kwargs) -> None:
    """
    Plot the proability density function of the criteria weights at the end of training

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    zero_line : bool, optional
        Whether or not to include a line to separate at mu = 0, by default False
    See 'cust_plot.plot' for more information on parameters
    """

    # Get distribution
    mu = np.array(plt_data['mu'])[:,-1]
    sig = np.array(plt_data['sig'])[:,-1]

    # Re-scale for percentages
    scale = _get_mu_scale(mu, scaled)
    mu *= scale
    sig *= scale

    # Draw
    xmax = (mu + 3 * sig).max()
    xmin = (mu - 3 * sig).min()
    step = (xmax - xmin) / 1000
    x = np.arange(xmin, xmax, step)
    y = [norm.pdf(x, mu[i], sig[i]) for i in range(len(mu))]
    _cust_plot(ax=ax, x=x, y=y, xlab='Weight', ylab='Probability Density', labels=plt_data['fns'], loc=loc,
        plt_title=f"Probability density function for criteria weights{plt_subtitle}", xzero_line=zero_line)


def _get_mu_scale(mu: List[List[float]], scaled: bool) -> float:
    """
    Returns the multiplier that scales criteria weights in each training step to represent each criteria's percentage influence

    Parameters
    ----------
    mu : List[List[float]]
        Mean weights of each criteria in each training step as a number of features x number of batches matrix
    scaled : bool
        Whether or not to scale data so that criteria weights represent their percentage influence instead of raw values

    Returns
    -------
    float
        Scale multiplier
    """

    return 1 / abs(np.array(mu)).sum(axis=0) if scaled else 1


def _default_cust_plot(ax: matplotlib.axes.Axes, plt_data: dict, y: 'List or np.ndarray', plt_off: int = 0, plt_avg: int = None, plt_title: str = '', plt_subtitle: str = '', xlab: str = 'Batches', ylab: str = '', loc: str = 'best', labels: List[str] = None, scaled: bool = False, xzero_line: bool = False, yzero_line: bool = False) -> None:
    """
    Used as a wrapper for plot_rewards, plot_success, plot_mu, plot_sigma and plot_pdf
    See 'cust_plot._cust_plot' for more information on parameters
    """

    x = np.arange(len(plt_data['rewards'])) + 1
    if labels is None:
        labels = plt_data['fns']
    _cust_plot(ax=ax, x=x, y=y, plt_title=f'{plt_title}{plt_subtitle}', xlab=xlab, ylab=ylab, labels=labels, off=plt_off, avg=plt_avg, loc=loc, xzero_line=xzero_line, yzero_line=yzero_line)


def _cust_plot(ax: matplotlib.axes.Axes, x: 'List or np.ndarray', y: 'List or np.ndarray', plt_title: str = None, xlab: str = None, ylab: str = None, labels: List[str] = None, off: int = 0, avg: int = None, loc: str = 'best', xzero_line: bool = False, yzero_line: bool = False) -> None:
    """
    Used to an x vs. y plot on a matplotlib axis

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    x, y : List or np.ndarray
        x and y-axis data
        If 2D, a line will be plotted for each row
    plt_title : str, optional
        Title text of the plot, by default None
    xlab : str, optional
        x-axis label, by default None
    ylab : str, optional
        y-axis label, by default None
    labels : List[str], optional
        Legend labels, by default None
    off : int, optional
        Start index of the plot, by default 0
    avg : int, optional
        Number of points to plot for a moving average, by default None
        If None, no moving average is plotted
    loc : str, optional
        Position of the legend, by default 'best'
        See 'matplotlib.pyplot.legend'
    xzero_line : bool, optional
        Whether or not to include a vertical line at x=0, by default False
    yzero_line : bool, optional
        Whether or not to include a horizontal line at y=0, by default False
    """

    # Data extraction methods
    inst = lambda M: any(isinstance(M, j) for j in [list, np.ndarray])
    get = lambda M: (lambda i: M[i]) if inst(M[0])  else (lambda i: M)
    len2d = lambda M: len(M) if inst(M[0]) else 1
    getx = get(x)
    gety = get(y)

    # Label the plot
    ax.set_title(plt_title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    # Plot multiple lines
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

    # Plot zero lines
    if xzero_line:
        ax.axvline(x=0, color='black', ls='--', lw=1)
    if yzero_line:
        ax.axhline(y=0, color='black', ls='--', lw=1)

    # Draw legend
    if labels:
        ax.legend(loc=loc)


def _move_avg(y: List[float], p: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the 'p' point moving average x and y coordinates from a sequence of values

    Parameters
    ----------
    y : List[float]
        Sequence of values
    p : int
        Number of points in the moving average

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        x and y coordinates of the moving average
    """

    if len(y) < p:
        return np.array([]), np.array([])
    c = np.cumsum(y)
    y_avg = (c[p:] - c[:-p]) / p
    x_avg = np.arange(len(y_avg)) + p
    return x_avg, y_avg