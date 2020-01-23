
import matplotlib
matplotlib.get_backend()
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pickle

filename = 'salience_variance_data-20191202-143836.dict'
all_data = pickle.load(open(filename, 'rb'))

#first figure out how many axes we want
sigmas = all_data['salience_means'].keys()
sigmas = [x[0] for x in sigmas]
sigmas = set(sigmas)
sigmas = list(sigmas)
sigmas.sort()

mps = all_data['salience_means'].keys()
mps = [x[1] for x in mps]
mps = set(mps)
mps = list(mps)
mps.sort()

fig1, fig1_axes = plt.subplots(1,len(sigmas),sharey=True)
fig2, fig2_axes = plt.subplots(len(sigmas),len(mps),sharey=True)
#top row will be reserved for blending values

##Figure 1: blend accuracy
for n in range(len(fig1_axes)):
    blend_x_values = []
    blend_std_values = []
    blend_y_values = []

    sigma = sigmas[n]
    for param in [(sigma,x) for x in mps]:
        blend_y_values.append(all_data['blend_means'][param])
        blend_x_values.append(param[1])
        blend_std_values.append(all_data['blend_std'][param][0])
    # fig1_axes[n].plot(blend_x_values,blend_y_values,label='sigma: ' + repr(sigma))
    fig1_axes[n].errorbar(blend_x_values, blend_y_values, yerr=blend_std_values, fmt='--o', label='mean diff. sigma: ' + repr(sigma))
    fig1_axes[n].legend()

##Figure 2: Saliences.  Sigmas vertical, mp horizontal
for n in range(len(sigmas)):
    ax_row = fig2_axes[n]
    for i in range(len(mps)):
        ax = ax_row[i]
        x_values = ['f' + repr(x) for x in range(1,len(all_data['salience_means'][(sigmas[n],mps[i])][0].tolist())+1)]
        ax.errorbar(x_values,all_data['salience_means'][(sigmas[n],mps[i])][0].tolist(), yerr=all_data['salience_means'][(sigmas[n],mps[i])][0].tolist(), fmt='o', label='sigma: ' + repr(sigmas[n]) + ' mp: ' + repr(mps[i]))
        ax.legend()

# params = [(x,y) for x in sigmas for y in mps]
# for n in range(len(params)):
#     ax = fig2_axes[n]
#     x_values = range(1,len(all_data['salience_means'][params[n]]))
#
#     ax.errorbar(x_values,all_data['salience_means'][params][n][0].tolist(), yerr=all_data['salience_means'][params][n], fmt='--o', label='sigma: ' + repr(params[0]) + ' mp: ' + repr(params[1]))
#     ax.legend()
















print('data open...')

plt.show()

#     ax1_x = range(1, len(blend_results)+1)
#     ax1.errorbar(mismatch_penalties, blend_results, yerr=blend_sd_results, fmt='--o', label='mean diff. sigma:' + repr(sigma))
#     ax1.legend()
#     ax1.set_title("mean diff. blend vs ground truth.")
#
#     for n in range(len(salience_mean_results)):#mean_set in salience_mean_results:
#         mean_set = salience_mean_results[n]
#         sd_set = salience_sd_results[n]
#         ax2_x = observation_slots#range(1,len(mean_set)+1)
#         MP = mismatch_penalties[n]
#         # ax2.scatter(ax2_x, mean_set, label='MP'+repr(i+1))#s=[sd * 1000 for sd in sd_set],
#         ax2.errorbar(ax2_x, mean_set, yerr=sd_set ,label='sigma: ' + repr(sigma) + ' MP: '+repr(MP),fmt='--o')
#     ax2.legend()
#     ax2.set_title("mean salience")
#
# plt.show()