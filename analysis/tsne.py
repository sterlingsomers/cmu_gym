from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle


'''Get the data'''
# pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/All_maps_random_500.tj','rb')
pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/All_maps_20x20_500_images.tj','rb')
obs = pickle.load(pickle_in)
dict = {}

t = 0
''' Collect observations '''
# for epis in range(len(obs)):
#     print('Episode:',epis)
#     # Get position of the flag
#     indx = np.nonzero(obs[epis]['flag'])[0][0] # find the timestep where you drop
#     print('Drop happened at timestep=',indx)
#     traj_length = obs[epis]['flag'].__len__()-1 # We take out the last obs as the drone has dropped
#     print('trajectory contains in total',traj_length, 'timesteps')
#     if (traj_length - indx >= 5): # collect trajectories that have multiple steps (5 or more) before dropping
#         # for i in range(traj_length-indx):
#         for i in range(traj_length):
#             sub_dict = {}
#             print('iter:',i,'indx+i=', indx+i)
#             # sub_dict['obs'] = obs[epis]['observations'][indx+i] # first i=0
#             sub_dict['fc'] = obs[epis]['fc'][indx + i]
#             sub_dict['actions'] = obs[epis]['actions'][indx + i] # actions are from 0 to 15
#
#             if epis==499:
#                 sub_dict['colorc'] = 'green'
#             else:
#                 sub_dict['colorc'] = 'blue'
#
#             if (indx+i)== (traj_length-1): # if obs at current timestep is a drop (YOU CAN DO THAT WITH THE ACTION=15)
#                 sub_dict['color'] = 'green'
#                 sub_dict['colorb'] = 'green'
#                 sub_dict['target'] = 1
#             else:
#                 sub_dict['color'] = 'blue'
#                 sub_dict['target'] = 0
#                 if sub_dict['actions'] == 0:
#                     sub_dict['colorb'] = 'red'
#                 elif sub_dict['actions'] == 14:
#                     sub_dict['colorb'] = 'cyan'
#                 elif sub_dict['actions'] == 13:
#                     sub_dict['colorb'] = 'magenta'
#                 elif sub_dict['actions'] == 12:
#                     sub_dict['colorb'] = 'yellow'
#                 elif sub_dict['actions'] == 11:
#                     sub_dict['colorb'] = 'black'
#                 elif sub_dict['actions'] == 10:
#                     sub_dict['colorb'] = '#eeefff'
#                 elif sub_dict['actions'] == 9:
#                     sub_dict['colorb'] = '#7db731'
#                 elif sub_dict['actions'] == 8:
#                     sub_dict['colorb'] = '#ba7332'
#                 elif sub_dict['actions'] == 7:
#                     sub_dict['colorb'] = '#91278b'
#                 elif sub_dict['actions'] == 6:
#                     sub_dict['colorb'] = '#264789'
#                 elif sub_dict['actions'] == 5:
#                     sub_dict['colorb'] = '#247e87'
#                 elif sub_dict['actions'] == 4:
#                     sub_dict['colorb'] = '#c3e2e5'
#                 elif sub_dict['actions'] == 3:
#                     sub_dict['colorb'] = '#f94736'
#                 elif sub_dict['actions'] == 2:
#                     sub_dict['colorb'] = '#3a0500'
#                 elif sub_dict['actions'] == 1:
#                     sub_dict['colorb'] = '#c4b942'
#             dict[t] = sub_dict
#             t = t + 1

# pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/All_maps_random_500_drop_traj.tj','wb')
# pickle.dump(dict,pickle_in)

for epis in range(len(obs)):
    print('Episode:',epis)
    # Get position of the flag
    indx = np.nonzero(obs[epis]['flag'])[0][0] # find the timestep where you switch to DRPPING AGENT !!!
    print('Switch to drop_agent happened at timestep=',indx)
    traj_length = obs[epis]['flag'].__len__()-1 # We take out the last obs as the drone has dropped
    print('full trajectory contains in total',traj_length, 'timesteps')
    print('dropping trajectory contains in total', traj_length-indx, 'timesteps')
    if (traj_length - indx >= 5): # collect drop trajectories that have multiple steps (5 or more) before dropping
        for i in range(traj_length-indx):
            sub_dict = {}
            print('iter:',i,'indx+i=', indx+i)
            # sub_dict['obs'] = obs[epis]['observations'][indx+i] # first i=0
            sub_dict['fc'] = obs[epis]['fc'][indx + i]
            sub_dict['images'] = obs[epis]['observations'][indx + i]
            sub_dict['actions'] = obs[epis]['actions'][indx + i] # actions are from 0 to 15
            sub_dict['values'] = obs[epis]['values'][indx + i]
            sub_dict['episode'] = epis
            sub_dict['timestep'] = i

            if sub_dict['actions'] == 15:
                sub_dict['action_label'] = 'drop'
            elif sub_dict['actions'] == 14:
                sub_dict['action_label'] = 'up-right'
            elif sub_dict['actions'] == 13:
                sub_dict['action_label'] = 'up-diag-right'
            elif sub_dict['actions'] == 12:
                sub_dict['action_label'] = 'up-forward'
            elif sub_dict['actions'] == 11:
                sub_dict['action_label'] = 'up-diag-left'
            elif sub_dict['actions'] == 10:
                sub_dict['action_label'] = 'up-left'
            elif sub_dict['actions'] == 9:
                sub_dict['action_label'] = 'right'
            elif sub_dict['actions'] == 8:
                sub_dict['action_label'] = 'diag-right'
            elif sub_dict['actions'] == 7:
                sub_dict['action_label'] = 'forward'
            elif sub_dict['actions'] == 6:
                sub_dict['action_label'] = 'diag-left'
            elif sub_dict['actions'] == 5:
                sub_dict['action_label'] = 'left'
            elif sub_dict['actions'] == 4:
                sub_dict['action_label'] = 'down-right'
            elif sub_dict['actions'] == 3:
                sub_dict['action_label'] = 'down-diag-right'
            elif sub_dict['actions'] == 2:
                sub_dict['action_label'] = 'down-forward'
            elif sub_dict['actions'] == 1:
                sub_dict['action_label'] = 'down-diag-left'
            elif sub_dict['actions'] == 0:
                sub_dict['action_label'] = 'down-left'
            
            # if epis==0:
            #     sub_dict['colorc'] = 'green'
            # else:
            #     sub_dict['colorc'] = 'blue'

            if (indx+i)== (traj_length-1): # if obs at current timestep is a drop (YOU CAN DO THAT WITH THE ACTION=15)
                sub_dict['color_drop'] = 'green'
                sub_dict['color_action'] = 'green'
                sub_dict['target'] = 1
            else:
                sub_dict['color_drop'] = 'blue'
                sub_dict['target'] = 0
                if sub_dict['actions'] == 0:
                    sub_dict['color_action'] = 'red'
                elif sub_dict['actions'] == 14:
                    sub_dict['color_action'] = 'cyan'
                elif sub_dict['actions'] == 13:
                    sub_dict['color_action'] = 'magenta'
                elif sub_dict['actions'] == 12:
                    sub_dict['color_action'] = 'yellow'
                elif sub_dict['actions'] == 11:
                    sub_dict['color_action'] = 'black'
                elif sub_dict['actions'] == 10:
                    sub_dict['color_action'] = '#eeefff'
                elif sub_dict['actions'] == 9:
                    sub_dict['color_action'] = '#7db731'
                elif sub_dict['actions'] == 8:
                    sub_dict['color_action'] = '#ba7332'
                elif sub_dict['actions'] == 7:
                    sub_dict['color_action'] = '#91278b'
                elif sub_dict['actions'] == 6:
                    sub_dict['color_action'] = '#264789'
                elif sub_dict['actions'] == 5:
                    sub_dict['color_action'] = '#247e87'
                elif sub_dict['actions'] == 4:
                    sub_dict['color_action'] = '#c3e2e5'
                elif sub_dict['actions'] == 3:
                    sub_dict['color_action'] = '#f94736'
                elif sub_dict['actions'] == 2:
                    sub_dict['color_action'] = '#3a0500'
                elif sub_dict['actions'] == 1:
                    sub_dict['color_action'] = '#c4b942'
            dict[t] = sub_dict # So actually here you store all drop agent timesteps from all episodes
            t = t + 1
    else:
        print("DROPPING TRAJECTORY OMITTED AS TOO SHORT")

print("Saving....")
pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/selected_drop_traj_images.tj','wb')
pickle.dump(dict,pickle_in)
print("...saved")

dims = (len(dict),256)
fc = np.zeros(dims)
for x in range(0,len(dict)):
    fc[x] = dict[x]['fc']

colors = []
for i in range(len(dict)):
    colors.append(dict[i]['colorb']) # Change the color to "color" or "colorb"
print('====>>>> tSNE...')
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=100)
X_tsne = tsne.fit_transform(fc)
plt.scatter(X_tsne[:,0],X_tsne[:,1],c=colors,alpha=0.2)
# plt.tick_params(
#     axis='both',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom='off',      # ticks along the bottom edge are off
#     top='off',         # ticks along the top edge are off
#     labelbottom='off',
#     left='off',
#     right='off',
#     labelleft='off')
# plt.title('t-SNE of fc layers representations (256neurons)')
#
# #Manually add legend (CAREFUL IF YOU CHANGE ORDER OR SMTH AS LABELS MIGHT NOT BE CONSISTENT)
# classes = ['blocking','orange+green','green','orange']
# class_colours = ['red','pink','green','orange']
# recs = []
# for i in range(0,len(class_colours)):
#     recs.append(mpatches.Circle((0,0),radius=5,fc=class_colours[i]))
# plt.legend(recs,classes,loc=4)
# plt.savefig('t-SNE_plus_orange.png',dpi=300)
plt.show()