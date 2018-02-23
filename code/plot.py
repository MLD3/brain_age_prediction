import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
def plotValdPerformances():
    x_vals      = np.array([6, 9, 13, 18, 27, 60])
    constant    = np.array([2.976, 3.334, 3.408, 3.150, 3.159, 3.397])
    constant_err = np.array([0.212, 0.170, 0.368, 0.222, 0.421, 0.179])
    depth       = np.array([3.080, 3.025, 3.265, 3.366, 3.321, 3.320])
    depth_err   = np.array([0.495, 0.363, 0.397, 0.530, 0.147, 0.277])
    reverse     = np.array([3.462, 3.141, 2.897, 2.902, 2.797, 3.196])
    reverse_err = np.array([0.462, 0.429, 0.131, 0.180, 0.260, 0.128])


    const_line = plt.errorbar(x_vals, constant, yerr=constant_err, fmt='-bo', capsize=4)
    depth_line = plt.errorbar(x_vals + 0.5, depth, yerr=depth_err, fmt='-ro', capsize=4)
    rever_line = plt.errorbar(x_vals + 1.0, reverse, yerr=reverse_err, fmt='-go', capsize=4)

    const_line.set_label('Constant Filters')
    depth_line.set_label('Normal Filters')
    rever_line.set_label('Reverse Filters')
    ax = plt.gca()
    ax.legend()
    plt.title('Comparison of 3D models on MRI imaging data')
    plt.xlabel('Stride Size')
    plt.ylabel('Mean Square Error')
    plt.show()

def plotTestPerformances():
    y_pos=np.arange(6)
    names=np.array(['depth_stride9', 'constant_stride6', 'reverse_stride_27', 'baseline', 'r_baseline', 'c_baseline'])
    title='Test Performance of best 3D models'
    p=np.array([4.257, 4.758, 3.652, 4.150, 4.137, 4.481])
    err = np.array([0.635, 1.602, 0.328, 0.407, 0.339, 0.815])
    plt.xticks(y_pos, names, rotation=45)
    plt.title(title)
    plt.ylabel('Mean Square Error')
    plt.bar(y_pos, p, align='center', alpha=0.5, yerr=err, capsize=30)
    plt.show()

def plotTrainingTime():
    x_vals      = np.array([6, 9, 13, 18, 27, 60])
    constant    = np.array([37, 23.5, 23, 26, 43, 210])
    depth       = np.array([31, 21, 20, 20, 23, 64])
    reverse     = np.array([37, 23.5, 23, 25.5, 40, 189])

    [constant_line]   = plt.plot(x_vals, constant, '-bo')
    [depth_line]      = plt.plot(x_vals, depth, '-ro')
    [reverse_line]    = plt.plot(x_vals, reverse, '-go')

    constant_line.set_label('Constant Filters')
    depth_line.set_label('Normal Filters')
    reverse_line.set_label('Reverse Filters')
    ax = plt.gca()
    ax.legend()

    for xy in zip(x_vals, constant):
        ax.annotate('{}'.format(xy[1]), xy=(xy[0]-1, xy[1]+5), textcoords='data')
    for xy in zip(x_vals, depth):
        ax.annotate('{}'.format(xy[1]), xy=(xy[0], xy[1]-6), textcoords='data')
    for xy in zip(x_vals, reverse):
        ax.annotate('{}'.format(xy[1]), xy=(xy[0]-1, xy[1]+1), textcoords='data')

    plt.title('Comparison of time taken by 3D models on MRI imaging data')
    plt.xlabel('Stride Size')
    plt.ylabel('Time (minutes) / 10k iterations')
    plt.show()

def plotNumberParams():
    x_vals      = np.array([1,2,3,4,5,6])
    constant_fc     = np.array([16640, 16640, 131328, 131328, 442624, 1310976])
    constant_conv   = np.array([2313792, 1209600, 912384, 820800, 787968, 775782])
    depth_fc        = np.array([16640, 16640, 131328, 131328, 442624, 1310976])
    depth_conv      = np.array([411912, 273888, 236736, 225228, 221184, 219672])
    reverse_fc      = np.array([2304, 2304, 16640, 16640, 55552, 164096])
    reverse_conv    = np.array([1759104, 654912, 357696, 266112, 233280, 221184])

    constant_plot_bot = plt.bar(x_vals-0.25, constant_conv, 0.25)
    constant_plot_top = plt.bar(x_vals-0.25, constant_fc, 0.25, bottom=constant_conv)
    depth_plot_bot = plt.bar(x_vals, depth_conv, 0.25)
    depth_plot_top = plt.bar(x_vals, depth_fc, 0.25, bottom=depth_conv)
    reverse_plot_bot = plt.bar(x_vals+0.25, reverse_conv, 0.25)
    reverse_plot_top = plt.bar(x_vals+0.25, reverse_fc, 0.25, bottom=reverse_conv)
    plt.legend((constant_plot_bot, constant_plot_top, depth_plot_bot, depth_plot_top, reverse_plot_bot, reverse_plot_top),
               ('Constant: Conv.', 'Constant: FC', 'Depth: Conv.', 'Depth: FC', 'Reverse: Conv.', 'Reverse: FC'))
    tick_ind = []
    tick_names = []
    strides = [6, 9, 13, 18, 27, '_baseline']
    for i in range(6):
        tick_ind.append(i + 1 - 0.25)
        tick_ind.append(i + 1)
        tick_ind.append(i + 1 + 0.25)
        tick_names.append('constant{}'.format(strides[i]))
        tick_names.append('depth{}'.format(strides[i]))
        tick_names.append('reverse{}'.format(strides[i]))
    plt.xticks(tick_ind, tick_names, rotation=60)

    plt.title('Comparison of number of parameters of 3D models on MRI imaging data')
    plt.xlabel('Stride Size')
    plt.ylabel('Number of Parameters')
    plt.show()

def scatterParamPerformance():
    constant_fc     = np.array([16640, 16640, 131328, 131328, 442624, 1310976])
    constant_conv   = np.array([2313792, 1209600, 912384, 820800, 787968, 775782])
    depth_fc        = np.array([16640, 16640, 131328, 131328, 442624, 1310976])
    depth_conv      = np.array([411912, 273888, 236736, 225228, 221184, 219672])
    reverse_fc      = np.array([2304, 2304, 16640, 16640, 55552, 164096])
    reverse_conv    = np.array([1759104, 654912, 357696, 266112, 233280, 221184])
    constant_test   = np.array([4.758, 5.057, 5.439, 3.910, 3.983, 4.481])
    depth_test      = np.array([4.276, 4.257, 3.902, 4.312, 4.058, 4.150])
    reverse_test    = np.array([8.591, 4.023, 3.865, 3.609, 3.652, 4.137])
    constant_err    = np.array([1.602, 0.977, 1.942, 0.399, 0.490, 0.815])
    depth_err       = np.array([0.084, 0.635, 0.496, 0.452, 0.317, 0.407])
    reverse_err     = np.array([1.539, 0.640, 0.736, 0.614, 0.328, 0.339])
    const_line = plt.errorbar(constant_conv + constant_fc, constant_test, fmt='bo', capsize=4)
    depth_line = plt.errorbar(depth_conv + depth_fc, depth_test, fmt='ro', capsize=4)
    rever_line = plt.errorbar(reverse_conv + reverse_fc, reverse_test, fmt='go', capsize=4)
    const_line.set_label('Constant Filters')
    depth_line.set_label('Normal Filters')
    rever_line.set_label('Reverse Filters')
    ax = plt.gca()
    ax.legend()
    plt.title('Comparison of 3D models on MRI imaging data')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Mean Square Error')
    plt.show()

if __name__ == '__main__':
    scatterParamPerformance()
