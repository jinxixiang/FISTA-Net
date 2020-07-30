# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 09:55:25 2020

@author: XIANG
"""
import matplotlib.pyplot as plt
import torch
from skimage import transform
import numpy as np  
from matplotlib import patches

# helper function to show ample data
def show_batch(data_loader, fig_name, idx):    
    """
    Show one batch of EIT sample data.
    idx, batch numvber to be shown;
    """
    for i_batch, (input_batch, target_batch) in enumerate(data_loader):
        #print('batch num: #', i_batch)
        if i_batch == idx:
            #print('batch num: #', i_batch, input_batch.size(), target_batch.size())

            nrows = 2  # number of sample to be displayed
            ncols = 2
            figsize = 4
            
            fig, ax = plt.subplots(
                nrows, ncols, sharex=False, sharey=False,
                figsize=(ncols * figsize, figsize * nrows))

            for i, ax_row in enumerate(ax):
                ax_row[0].imshow(input_batch[i],cmap=plt.cm.Greys_r,
                    extent=(0, 180, 0, input_batch[i].shape[0]), aspect='auto')
                ax_row[1].imshow(target_batch[i], cmap=plt.cm.Greys_r)

            plt.show()
            fig.savefig(fig_name)



def show_image_matrix(fig_name, image_batches, titles=None, indices=None, **kwargs):
    """Visualize a 2D set of images arranged in a grid.
    Parameters
    ----------
    image_batches : sequence of `Tensor` or `Variable`
        List containing batches of images that should be displayed.
        Each tensor should have the same shape after squeezing, except
        for the batch axis.
    titles : sequence of str, optional
        Titles for the colums in the plot. By default, titles are empty.
    indices : sequence of int, optional
    kwargs :
        Further keyword arguments that are passed on to the Matplotlib
        ``imshow`` function.
    """

    if indices is None:
        displayed_batches = image_batches
    else:
        displayed_batches = [batch[indices] for batch in image_batches]

    nrows = len(displayed_batches[0])
    ncols = len(displayed_batches)
    
    print('nrows:', nrows, 'ncols:', ncols)
    
    if titles is None:
        titles = [''] * ncols

    figsize = 10
    fig, rows = plt.subplots(
        nrows, ncols, sharex=False, sharey=False,
        figsize=(ncols * figsize, figsize * nrows))

    if nrows == 1:
        rows = [rows]

    for i, row in enumerate(rows):
        if ncols == 1:
            row = [row]
        for name, batch, ax in zip(titles, displayed_batches, row):
            if i == 0:
                ax.set_title(name)
            pcm = ax.imshow(batch[i].squeeze(), cmap=plt.cm.Greys_r, vmin=0, vmax=0.6)
            fig.colorbar(pcm, ax=ax)
            ax.set_axis_off()
    plt.show()
    fig.savefig(fig_name)
    