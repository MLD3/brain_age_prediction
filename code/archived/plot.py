import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
def plotValdPerformances():
    x_vals      = np.array([10, 15, 20, 30])
    batch       = np.array([4.964, 4.656, 4.199, 4.284])
    batch_err   = np.array([0.633, 0.306, 0.151, 0.250])
    depth       = np.array([4.531, 5.113, 5.341, 4.961])
    depth_err   = np.array([0.664, 0.564, 0.656, 0.338])
    halfway     = np.array([4.327, 4.533, 4.339, 4.575])
    halfway_err = np.array([0.228, 0.161, 0.222, 0.263])
    baseline_x    = np.array([31])
    baseline_y    = np.array([4.320])
    baseline_err  = np.array([0.226])

    batch_line = plt.errorbar(x_vals, batch, yerr=batch_err, fmt='-bo', capsize=4)
    depth_line = plt.errorbar(x_vals + 0.2, depth, yerr=depth_err, fmt='-ro', capsize=4)
    halfway_line = plt.errorbar(x_vals + 0.4, halfway, yerr=halfway_err, fmt='-go', capsize=4)
    base_line  = plt.errorbar(baseline_x, baseline_y, yerr=baseline_err, fmt='-mo', capsize=4)

    batch_line.set_label('Batch Concatenation')
    depth_line.set_label('Depth Concatenation')
    halfway_line.set_label('Halfway Concatenation')
    base_line.set_label('Baseline 3D CNN')
    ax = plt.gca()
    ax.legend()
    plt.title('Comparison of 3D models on MRI imaging data')
    plt.xlabel('Stride Size')
    plt.ylabel('Mean Square Error')
    plt.show()

def plotTestPerformances():
    y_pos=np.arange(4)
    names=np.array(['Batch_Stride20', 'Depth_Stride10', 'Halfway_Stride10', 'Baseline'])
    title='Test Performance of best 3D models'
    p=np.array([4.290, 4.700, 4.006, 4.295])
    err = np.array([1.046, 0.432, 0.390, 0.714])
    plt.xticks(y_pos, names, rotation=45)
    plt.title(title)
    plt.ylabel('Mean Square Error')
    plt.bar(y_pos, p, align='center', alpha=0.5, yerr=err, capsize=30)
    plt.show()

def plotTrainingTime():
    x_vals      = np.array([10, 15, 20, 30])
    batch       = np.array([240, 75, 37, 18])
    depth       = np.array([3.5, 3.083, 3.25, 3.83])
    halfway     = np.array([160, 53, 29, 16])
    baseline    = np.array([12.567] * 4)

    [batch_line]      = plt.plot(x_vals, batch, '-bo')
    [depth_line]      = plt.plot(x_vals, depth, '-ro')
    [halfway_line]    = plt.plot(x_vals, halfway, '-go')
    [base_line]       = plt.plot(x_vals, baseline, '-mo')

    batch_line.set_label('Batch Concatenation')
    depth_line.set_label('Depth Concatenation')
    halfway_line.set_label('Halfway Concatenation')
    base_line.set_label('Baseline 3D CNN')
    ax = plt.gca()
    ax.legend()

    for xy in zip(x_vals, batch):
        ax.annotate('{}'.format(xy[1]), xy=(xy[0]-0.1, xy[1]+5), textcoords='data')
    for xy in zip(x_vals, depth):
        ax.annotate('{}'.format(xy[1]), xy=(xy[0]+0.2, xy[1]+0.8), textcoords='data')
    for xy in zip(x_vals, halfway):
        ax.annotate('{}'.format(xy[1]), xy=(xy[0]+0.2, xy[1]+3), textcoords='data')
    for xy in zip(x_vals, baseline):
        ax.annotate('{}'.format(xy[1]), xy=(xy[0]+0.2, xy[1]+0.8), textcoords='data')

    plt.title('Comparison of time taken by 3D models on MRI imaging data')
    plt.xlabel('Stride Size')
    plt.ylabel('Time (minutes) / 10k iterations')
    plt.show()

if __name__ == '__main__':
    plotTestPerformances()
