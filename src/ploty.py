import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18, 'font.family': 'sans'})
plt.style.use('ggplot')


def scatter_plot(ax, x, y, title, xlab, ylab, color, zorder=1):
    '''Create a scatter plot

    Parameters
    ----------
    ax: plot axis
    x: list in the x-axis
    y: list in the y-axis
    title: str
    xlab: str
    ylab: str
    color: str
    zorder: int, default set to 1

    Returns
    -------
    None
    '''
    ax.scatter(x, y, alpha= 0.5, color=color, s=50, zorder=1)
    ax.set_title(title, fontsize=35)
    ax.set_ylabel(xlab, fontsize=20)
    ax.set_xlabel(ylab, fontsize=20)


def line_plot(ax, x, y, color, label):
    '''Create a line plot

    Parameters
    ----------
    ax: plot axis
    x: list in the x-axis
    y: list in the y-axis
    label: str
    color: str

    Returns
    -------
    None
    '''
    ax.plot(x, y, linewidth=2, color=color, label=label)


def bar_plot(ax, data, label):
    '''Create a bar plot

    Parameters
    ----------
    ax: plot axis
    data: list of ints
    label: string
    Returns
    -------
    None
    '''
    ax.bar(label, data, label=label)
