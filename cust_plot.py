import numpy as np
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, List


def plot(plt_data: dict, plt_avg: int = None, plt_off: int = 0, plt_subtitle: str = '', figsize: Tuple[int,int] = (20,24), loc: str = None, scaled: bool = True) -> None:
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
    """

    # Create plot
    _, ax = plt.subplots(nrows=3, ncols=2, figsize=figsize, facecolor='w')
    def_plot = lambda ax, fn, scaled, **kwargs: fn(ax=ax, plt_data=plt_data, plt_off=plt_off, plt_avg=plt_avg, plt_subtitle=plt_subtitle, scaled=scaled, **kwargs)
    if loc is not None:
        def_plot = lambda ax, fn, scaled: def_plot(ax=ax, fn=fn, scaled=scaled, loc=loc)

    # Add rewards, success, mu, sigma and pdf
    def_plot(ax[0,0], plot_rewards, False)
    def_plot(ax[0,1], plot_success, False)
    def_plot(ax[1,0], plot_mu, scaled)
    def_plot(ax[1,1], plot_sig, scaled)
    plot_pdf(ax[2,0], plt_data, plt_subtitle=plt_subtitle, scaled=scaled)


def plot_rewards(ax, plt_data, plt_avg=None, plt_off=0, plt_subtitle='', loc='lower right', **kwargs) -> None:
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


def plot_mu(ax: matplotlib.axes.Axes, plt_data: dict, plt_off: int = 0, plt_subtitle: str = '', loc: str = 'lower left', scaled: bool = True, **kwargs) -> None:
    """
    Plot the evolution of mean criteria weights mu in training

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    See 'cust_plot.plot' for more information on parameters
    """

    scale = _get_mu_scale(plt_data['mu'], scaled)
    y = np.array(plt_data['mu']) * scale
    plt_title = 'Mean criteria weight vs. batches'
    ylab = 'Mu'
    _default_cust_plot(ax=ax, plt_data=plt_data, y=y, plt_off=plt_off, plt_title=plt_title, plt_subtitle=plt_subtitle, ylab=ylab, loc=loc)


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


def plot_pdf(ax: matplotlib.axes.Axes, plt_data: dict, plt_subtitle: str = '', loc: str = 'center right', scaled: bool = True, **kwargs) -> None:
    """
    Plot the proability density function of the criteria weights at the end of training

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
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
    xmax = max(abs((mu + 3 * sig).max()), abs((mu - 3 * sig).min()))
    xmin = -xmax
    step = 2 * xmax / 1000
    x = np.arange(xmin, xmax, step)
    y = [norm.pdf(x, mu[i], sig[i]) for i in range(len(mu))]
    _cust_plot(ax=ax, x=x, y=y, xlab='Weight', ylab='Probability Density', labels=plt_data['fns'], loc=loc,
        plt_title=f"Probability density function for criteria weights{plt_subtitle}")


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


def _default_cust_plot(ax: matplotlib.axes.Axes, plt_data: dict, y: 'List or np.ndarray', plt_off: int = 0, plt_avg: int = None, plt_title: str = '', plt_subtitle: str = '', xlab: str = 'Batches', ylab: str = '', loc: str = 'best', labels: List[str] = None, scaled: bool = False) -> None:
    """
    Used as a wrapper for plot_rewards, plot_success, plot_mu, plot_sigma and plot_pdf
    See 'cust_plot._cust_plot' for more information on parameters
    """

    x = np.arange(len(plt_data['rewards'])) + 1
    if labels is None:
        labels = plt_data['fns']
    _cust_plot(ax=ax, x=x, y=y, plt_title=f'{plt_title}{plt_subtitle}', xlab=xlab, ylab=ylab, labels=labels, off=plt_off, avg=plt_avg, loc=loc)


def _cust_plot(ax: matplotlib.axes.Axes, x: 'List or np.ndarray', y: 'List or np.ndarray', plt_title: str = None, xlab: str = None, ylab: str = None, labels: List[str] = None, off: int = 0, avg: int = None, loc: str = 'best') -> None:
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