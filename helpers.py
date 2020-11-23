# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 09:55:25 2020

@author: XIANG
"""
import matplotlib.pyplot as plt
import torch
from skimage import transform
import numpy as np  
import matplotlib
matplotlib.use("Agg")
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
            print('batch num: #', i_batch, input_batch.size(), target_batch.size())

            nrows = 3  # number of sample to be displayed
            ncols = 2
            figsize = 4
            
            fig, ax = plt.subplots(
                nrows, ncols, sharex=False, sharey=False,
                figsize=(ncols * figsize, figsize * nrows))

            for i, ax_row in enumerate(ax):
                ax_row[0].plot(input_batch[i])
                ax_row[1].imshow(target_batch[i])

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
            pcm = ax.imshow(batch[i].squeeze(), cmap='jet', vmin=0, vmax=1)
            fig.colorbar(pcm, ax=ax)
            ax.set_axis_off()
    plt.show()
    fig.savefig(fig_name)
    


def test_rescale(vect_in):
    # vect_in: tensor (batch_size, 1, 104, 1)
    # from 1d vector to 2d EIM
    # output (batch_size, 1, 64, 64)
    vect_in = torch.squeeze(vect_in, 1)
    vect_in = torch.squeeze(vect_in, 2)
    vect_in = vect_in.detach().numpy() # (batch_size, 104)
    
    rescale_mat = np.zeros((vect_in.shape[0],16, 16))
    vect_in = np.insert(vect_in, 13, 0, axis=1)
    k = 0
    for i in range(14):
        rescale_mat[:, i, 2+i:] = vect_in[:, k:k+14-i]
        k = k + 14 - i

    rescale_mat = rescale_mat + rescale_mat.transpose((0, 2, 1))
    arr_out = transform.resize(rescale_mat, (vect_in.shape[0], 64, 64))
    ts_out = torch.from_numpy(arr_out)
    ts_out = torch.unsqueeze(ts_out, 1)
    return ts_out


def gen_gaussian_noise(signal, SNR):
    """
    signal dimension: (batch_size, vector_dim)
    """
    noise  = np.random.randn(*signal.shape)
    noise = noise - np.mean(noise)
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))
    noise_var = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_var) / np.std(noise)) * noise
    return noise