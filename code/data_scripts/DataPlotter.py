import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def PlotTrainingValidationLoss(accumulatedTrainingLosses, accumulatedValidationLosses, title, savePath, defaultBatchIndexSpacing=50):
    """
    Saves a plot of the training loss and validation loss for a single run of training.
    """
    numberOfFolds, numberOfSteps = accumulatedTrainingLosses.shape

    batchX = np.linspace(0, (numberOfSteps - 1) * defaultBatchIndexSpacing, numberOfSteps)

    plt.title(title)
    plt.subplot(1, 2, 1)
    plt.xlabel('Batch Index')
    plt.ylabel('Training Loss')
    for k in range(numberOfFolds):
        plt.scatter(batchX, accumulatedTrainingLoss[k, :], c=(0.5 * k/numberOfFolds, 0.7 * k / numberOfFolds, 1.0))

    plt.subplot(1, 2, 2)
    plt.xlabel('Batch Index')
    plt.ylabel('Validation Loss')
    for k in range(numberOfFolds):
        plt.scatter(batchX, accumulatedValidationLoss[k, :], c=(1.0 * k / numberOfFolds, 0.0, 0.0))

    plt.savefig(savePath, bbox_inches='tight')


def plotHist(X, saveName='', title='Histogram of X'):
    fig, ax = plt.subplots()
    counts, bins, patches = ax.hist(X, facecolor='mediumseagreen', edgecolor='black')

    # Set the ticks to be at the edges of the bins.
    ax.set_xticks(bins)
    # Set the xaxis's tick labels to be formatted with 1 decimal place...
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    # Change the colors of bars at the edges...
    twentyfifth, seventyfifth = np.percentile(X, [25, 75])
    for patch, rightside, leftside in zip(patches, bins[1:], bins[:-1]):
        if rightside < twentyfifth:
            patch.set_facecolor('mediumaquamarine')
        elif leftside > seventyfifth:
            patch.set_facecolor('mediumturquoise')

    # Label the raw counts and the percentages below the x-axis...
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        # Label the raw counts
        ax.annotate(str(count), xy=(x, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -18), textcoords='offset points', va='top', ha='center')

        # Label the percentages
        percent = '%0.0f%%' % (100 * float(count) / counts.sum())
        ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -32), textcoords='offset points', va='top', ha='center')

    # Give ourselves some more room at the bottom of the plot
    plt.subplots_adjust(bottom=0.15)
    plt.ylabel('Frequency in counts')
    plt.title(title)
    if saveName == '':
        plt.show()
    else:
        plt.savefig(saveName, bbox_inches='tight')
