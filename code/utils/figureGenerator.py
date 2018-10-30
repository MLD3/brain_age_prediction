import numpy as np
import matplotlib.pyplot as plt

setSizeMax = [[4.026662, 3.6936357, 3.0700812, 3.292542, 3.3384645], [3.5657804, 3.5086155, 3.132291, 2.8430943, 3.3452682], 
              [3.7519553, 3.6189108, 3.2368956, 3.228827, 3.4537094], [3.9054072, 3.6797917, 3.1281686, 3.4469864, 3.3653097],
              [3.9648356, 3.2223673, 3.1712408, 2.9377477, 3.5153627], [3.6443756, 3.3611243, 3.1863413, 3.1622417, 3.4042056]]
setSize100 = [[5.198996, 4.5753193, 4.625415, 4.4554176, 4.4561825], [5.1806836, 4.5786624, 4.6436014, 4.61493, 4.854835],
              [5.236398, 4.8247604, 4.4500566, 4.5757008, 4.990748], [5.359669, 4.9746337, 4.437601, 4.810336, 4.8332353],
              [5.2469606, 4.895724, 4.629127, 4.670203, 4.53984], [5.128481, 4.5420375, 4.5636864, 4.845483, 4.491019]]
setSize200 = [[4.8781123, 3.7757328, 3.8954775, 3.8441284, 3.9132178], [4.761964, 3.9234145, 4.0152464, 3.942052, 3.7950258],
              [4.6629977, 3.6903708, 3.6849656, 3.5651596, 3.8488114], [4.9443703, 3.8234732, 3.8973026, 3.624601, 3.9894443],
              [4.6528516, 4.192685, 3.8941238, 3.8305833, 3.7537909], [5.0757504, 4.009447, 3.7111387, 3.665534, 3.9744802]]
setSize300 = [[4.322419, 3.6598663, 2.970005, 3.4505024, 3.5190794], [4.1831474, 3.6429536, 3.2843113, 3.2719688, 3.5503485],
              [4.184121, 3.6398535, 3.447017, 3.5356555, 3.56401], [4.4305043, 3.537839, 3.1223776, 3.2703493, 3.5969524],
              [4.2838726, 3.477313, 3.5414884, 3.3749697, 3.9233503], [4.104877, 3.5391066, 3.2835493, 3.199413, 3.4291399]]
'''
setSizeMax = np.array([np.mean(i) for i in setSizeMax])
setSize100 = np.array([np.mean(i) for i in setSize100])
setSize200 = np.array([np.mean(i) for i in setSize200])
setSize300 = np.array([np.mean(i) for i in setSize300])
x = [1, 2, 3, 4, 5, 6]
aug_rate = ['0(Baseline)', '0.25', '0.5', '1', '2', '3']
fig, ax = plt.subplots()
ax.set_title('Validation Performance for Different Ratio of Combine Augmentation')
ax.set_xlabel('Ratio of Augmented Image Added')
ax.set_ylabel('MSE on Validation Set')
ax.plot(x, setSize100, 'r', label='Training Set Size 100', marker='o')
ax.plot(x, setSize200, 'b', label='Training Set Size 200', marker='o')
ax.plot(x, setSize300, 'g', label='Training Set Size 300', marker='o')
ax.plot(x, setSizeMax, 'y', label='Training Set Size 435 (Max)', marker='o')
ax2 = ax.twinx()
plt.xticks(x, aug_rate, rotation=0)
plt.legend(loc='best')
plt.grid()
plt.show()
'''
setSizeMaxPrime = [[3.7712011, 3.5206826, 3.0489833, 3.1438515, 3.4459448], [3.7805753, 3.439979, 3.0599868, 3.1597443, 3.4454644],
                   [3.6563258, 3.7810953, 3.012351, 3.4448557, 3.1203332], [3.718237, 3.7966473, 3.1685765, 3.1563003, 3.5926354],
                   [3.7432508, 3.5170836, 3.1815531, 3.4896865, 3.276381]]
setSize100Prime = [[5.19147, 4.280733, 5.1098, 4.8000183, 5.6077733], [5.1743927, 4.4417105, 4.638224, 4.296157, 5.5589294],
                   [5.3328676, 4.377469, 4.8432336, 4.4286995, 5.676351], [5.3219247, 4.242214, 4.9739842, 4.5192533, 5.585831],
                   [4.8365793, 4.1677003, 4.7135353, 4.621968, 5.4021535]]
setSize200Prime = [[4.368521, 4.035285, 3.537452, 3.6039917, 3.860363], [4.2749777, 4.0978265, 3.9835057, 3.5335777, 3.9777613],
                   [4.363634, 3.685268, 3.8746233, 3.7890735, 3.9389577], [4.440342, 4.389369, 3.7450044, 3.520944, 4.1532],
                   [4.6433415, 4.1851707, 3.3530242, 3.6587255, 3.7910657]]
setSize300Prime = [[4.0141993, 3.803925, 3.215949, 3.7609594, 3.785941], [4.1543455, 3.816741, 3.3380601, 3.838153, 3.8036542],
                   [4.0206, 3.8547723, 3.356816, 3.863352, 3.825958], [3.966241, 4.159071, 3.043993, 3.6277225, 3.5512514],
                   [3.9821718, 4.18363, 3.5879092, 3.7373502, 3.8310058]]
setSizeMax = np.array([np.mean(setSizeMax[i]) for i in range(1, 6)])
setSize100 = np.array([np.mean(setSize100[i]) for i in range(1, 6)])
setSize200 = np.array([np.mean(setSize200[i]) for i in range(1, 6)])
setSize300 = np.array([np.mean(setSize300[i]) for i in range(1, 6)])
setSizeMaxPrime = np.array([np.mean(i) for i in setSizeMaxPrime])
setSize100Prime = np.array([np.mean(i) for i in setSize100Prime])
setSize200Prime = np.array([np.mean(i) for i in setSize200Prime])
setSize300Prime = np.array([np.mean(i) for i in setSize300Prime])
x = [1, 2, 3, 4, 5]
aug_rate = ['0.25', '0.5', '1', '2', '3']
f, axarr = plt.subplots(4, sharex=True)
axarr[0].plot(x, setSizeMaxPrime, 'salmon', label='Training Set Size Max without Combination', marker='o')
axarr[0].plot(x, setSizeMax, 'crimson', label='Training Set Size Max with Combination', marker='o')
axarr[1].plot(x, setSize100Prime, 'orchid', label='Training Set Size 100 without Combination', marker='o')
axarr[1].plot(x, setSize100, 'darkviolet', label='Training Set Size 100 with Combination', marker='o')
axarr[2].plot(x, setSize200Prime, 'limegreen', label='Training Set Size 200 without Combination', marker='o')
axarr[2].plot(x, setSize200, 'seagreen', label='Training Set Size 200 with Combination', marker='o')
axarr[3].plot(x, setSize300Prime, 'skyblue', label='Training Set Size 300 without Combination', marker='o')
axarr[3].plot(x, setSize300, 'blue', label='Training Set Size 300 with Combination', marker='o')
axarr[0].set_title('Difference between Performance after Replacement')
axarr[3].set_xlabel('Ratio of Augmented Image Added')
for i in range(4):
    axarr[i].set_ylabel('MSE on Validation Set')
    axarr[i].legend(loc='best')
plt.xticks(x, aug_rate, rotation=0)
plt.grid()
plt.show()