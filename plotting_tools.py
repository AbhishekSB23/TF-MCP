'''

UTILITY TOOLS FOR NeuSamp

AUTHOR: ABIJITH J. KAMATH
abijithj@iisc.ac.in

Cite:
[1] 

'''

# %% LOAD LIBRARIES

from email.policy import default
import os
import pywt

import numpy as np

from numpy import matlib
from matplotlib import pyplot as plt
# from celluloid import Camera

import matplotlib.ticker as ticker

# %% PLOT TOOLS

def plot_diracs(tk, ak, ax=None, plot_colour='blue', alpha=1,
    line_width=2, marker_style='o', marker_size=4, line_style='-',
    legend_show=True, legend_loc='lower left', legend_label=None, ncols=2,
    title_text=None, xaxis_label=None, yaxis_label=None, xlimits=[0,1],
    ylimits=[-1,1], show=False, save=None):
    ''' Plots Diracs at tk, ak '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    markerline, stemlines, baseline = plt.stem(tk, ak, label=legend_label,
        linefmt=line_style)
    plt.setp(stemlines, linewidth=line_width, color=plot_colour, alpha=alpha)
    plt.setp(markerline, marker=marker_style, linewidth=line_width, alpha=alpha,
        markersize=marker_size, markerfacecolor=plot_colour, mec=plot_colour)
    plt.setp(baseline, linewidth=0)

    if legend_label and legend_show:
        plt.legend(ncol=ncols, loc=legend_loc, frameon=True, framealpha=0.8,
            facecolor='white')

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_signal(x, y, ax=None, plot_colour='blue', alpha=1, xaxis_label=None,
    yaxis_label=None, title_text=None, legend_label=None, legend_show=True,
    legend_loc='lower left', n_col=2, line_style='-', line_width=None,
    xlimits=[-2,2], ylimits=[-2,2], axis_formatter='%0.1f',
    show=False, save=None, annotates = False, annotation = None, 
    pos = None, marker = None, markersize = 10, grid = None):

    '''
    Plots signal with abscissa in x and ordinates in y 

    '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.plot(x, y, linestyle=line_style, linewidth=line_width,
        color=plot_colour, label=legend_label, zorder=0, alpha=alpha, 
        marker = marker, markersize = markersize)
    if legend_label and legend_show:
        plt.legend(ncol=n_col, loc=legend_loc, frameon=False, framealpha=0.8,
            facecolor='white')
    
    if grid:
        plt.grid()
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)
    
    if annotates:
        plt.annotate(annotation, xy=pos, color=plot_colour)

    if axis_formatter:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(axis_formatter))

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_points(xpoints, ypoints, ax=None, line_width=None,  xaxis_label=None,
    yaxis_label=None, point_colour='black', size = None,
    alpha=1, legend_label=None, legend_show=True, marker_style='o',
    legend_loc='lower left', title_text=None, show=False,
    xlimits=[0,1], ylimits=[-1,1], save=None, xticks = None):
    '''Scatter plot of points'''

    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.scatter(xpoints, ypoints, s = size,  zorder=10, marker=marker_style, alpha=alpha,
        color=point_colour, linewidth=line_width, label=legend_label)
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    if xticks:
        plt.xticks(xpoints, xticks)

    if legend_label and legend_show:
        plt.legend(ncol=2, loc=legend_loc, frameon=False, framealpha=0.8, facecolor='white')
        # plt.legend(ncol=2, loc=legend_loc, frameon=False, facecolor='white',fancybox=True, framealpha=0.5 )

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.title(title_text)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_points_on_xaxis(points, ax=None, line_width=None, point_colour='black',
    alpha=1, legend_label=None, legend_show=True, legend_loc='lower left',
    title_text=None, show=False, xlimits=[0,1], ylimits=[-1,1], save=None):
    '''
    Plots points on x-axis

    '''
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()    

    zerovec = np.zeros(len(points))
    plt.scatter(points, zerovec, zorder=10, marker='o', alpha=alpha,
        color=point_colour, linewidth=line_width, label=legend_label)
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)

    if legend_label and legend_show:
        plt.legend(ncol=2, loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.title(title_text)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_hline(level=0, ax=None, line_colour='black', line_style='-',
    alpha=1, line_width=0.5, annotation=None, pos=(1,1)):
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.axhline(level, color=line_colour, linestyle=line_style,
        linewidth=line_width, alpha=alpha)
    if annotation:
        plt.annotate(annotation, xy=pos, color=line_colour)

def plot_dirac_clouds(tk, ak, ax=None, plot_colour='blue', alpha=1,
    line_width=2, marker_style='*', marker_size=4, legend_show=True,
    legend_loc='lower left', legend_label=None, title_text=None,
    xaxis_label=None, yaxis_label=None, xlimits=[0,1], ylimits=[-1,1],
    show=False, save=None):
    ''' Scatter plots of Diracs at tk, ak '''

    plt.scatter(tk, ak, color=plot_colour, s=marker_size, alpha=alpha,
        marker=marker_style, edgecolors=plot_colour, label=legend_label)

    if legend_label and legend_show:
        plt.legend(ncol=2, loc=legend_loc, frameon=True, framealpha=0.8,
            facecolor='white')

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_mcerrors(x, y, ax=None, plot_colour='blue', line_width=2,
    marker_style='o', marker_size=4, line_style='-', legend_label=None,
    legend_loc='lower left', legend_show=True, title_text=None, dev_alpha=0.5,
    xaxis_label=None, yaxis_label=None, xlimits=[-30,30], ylimits=[1e-4, 1e2],
    show=False, save=None):
    ''' Plot x,y on loglog '''

    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    # means = np.mean(y, axis=1)
    # devs = np.std(y, axis=1)
    median = np.median(y, axis=1)
    q1 = np.quantile(y, 0.25, axis=1)
    q3 = np.quantile(y, 0.75, axis=1)
    plt.loglog(x, median, linestyle=line_style, linewidth=line_width,
        color=plot_colour, marker=marker_style, markersize=marker_size,
        label=legend_label)
    plt.fill_between(x, q1, q3, color=plot_colour,
        linewidth=0, alpha=dev_alpha)

    if legend_label and legend_show:
        plt.legend(loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_loss(x, y, ax=None, plot_colour='blue', line_width=2,
    marker_style='o', marker_size=4, line_style='-', legend_label=None,
    legend_loc='lower left', legend_show=True, title_text=None, dev_alpha=0.5,
    xaxis_label=None, yaxis_label=None, xlimits=[-30,30], ylimits=[1e-4, 1e2],
    show=False, save=None):
    ''' Plot loss function vs. parameter '''

    if ax is None:
        fig = plt.figure(figsize=(12,6))
        ax = plt.gca()

    plt.semilogy(x, y, linestyle=line_style, linewidth=line_width,
        color=plot_colour, marker=marker_style, markersize=marker_size,
        label=legend_label)

    if legend_label and legend_show:
        plt.legend(loc=legend_loc, frameon=True, framealpha=0.8, facecolor='white')

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

def plot_position_estimates(tk, tk_estimate, jitter_idx, ax=None,
    plot_colour='blue', marker_style='o', marker_size=4, legend_label=None,
    legend_loc='lower right', legend_show=True, title_text=None,
    xaxis_label=None, yaxis_label=None, xlimits=[0,1], ylimits=[0,1],
    plot_line=False, show=True, save=None):
    ''' 2D scatter plot for true position vs estimated position '''

    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.gca()

    estimate = np.mean(tk_estimate[jitter_idx,:,:], axis=0)
    devs = np.std(tk_estimate[jitter_idx,:,:], axis=0)
    # plt.scatter(estimate, tk, marker=marker_style, color=plot_colour,
    #     s=marker_size, linewidth=line_width, label=legend_label)
    plt.errorbar(estimate, tk, xerr=devs, fmt=marker_style, color=plot_colour,
        ms=marker_size, capsize=10, alpha=0.8, label=legend_label)

    if plot_line:
        t = np.linspace(xlimits[0],xlimits[1])
        plt.plot(t, t, color='black', linestyle='-', zorder=0)

    if legend_label and legend_show:
        plt.legend(loc=legend_loc, frameon=True, framealpha=0.8,
            facecolor='white')

    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title_text)

    if save:
        plt.savefig(save + '.pdf', format='pdf')

    if show:
        plt.show()

    return

# %% VIDEO RENDERING

def sampling_simulation(t, signal, reference, mode, tk, samples=None,
    threshold=None, xlimits=[0,1], ylimits=[-0.6,0.9], save=None):
    '''
    Makes video with signals and samples

    '''
    
    fig = plt.figure(figsize=(12,6))
    ax = plt.gca()
    camera = Camera(fig)

    if mode == 'ctem':
        num_points = signal.shape[0]
        for i in tqdm(range(num_points)):
            plot_hline(0, ax=ax)
            ax.plot(t, signal, c='blue', linewidth=4, linestyle='-')
            ax.plot(t[:i], reference[:i], c='red', linewidth=1, linestyle='--')
            camera.snap()

        plot_signal(t, signal, ax=ax, plot_colour='blue', line_width=4,
            line_style='-', legend_label=r"$y(t)$")
        plot_signal(t, reference, ax=ax, plot_colour='red', line_width=1,
            line_style='--', legend_label=r"$r(t)$")
        plot_diracs(tk, samples, ax=ax, plot_colour='red', marker_size=10,
            line_width=4, legend_label=r"($t'_n, y(t'_n))$")
        camera.snap()

    elif mode == 'iftem':
        num_points = signal.shape[0]
        for i in tqdm(range(num_points)):
            plot_hline(threshold, ax=ax, annotation=r'$\gamma$',
                pos=(0.975,0.15))
            plot_hline(0, ax=ax)
            ax.plot(t, signal, c='blue', linewidth=4, linestyle='-')
            ax.plot(t[:i], reference[:i], c='red', linewidth=1, linestyle='--')
            camera.snap()

        plot_hline(0, ax=ax)
        plot_hline(threshold, ax=ax, annotation=r'$\gamma$',
            pos=(0.975,0.15))
        plot_signal(t, signal, ax=ax, plot_colour='blue', line_width=4,
            line_style='-', legend_label=r"$y(t)$")
        plot_signal(t, reference, ax=ax, plot_colour='red', line_width=1,
            line_style='--', legend_label=r"$v(t)$")
        plot_points_on_xaxis(tk, ax=ax,
            legend_label=r"$t'_n$", point_colour='red')
        camera.snap()

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

    plt.xlim(xlimits)
    plt.ylim(ylimits)

    animation = camera.animate()
    animation.save(save+'.mp4')

    return
