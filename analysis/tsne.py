from sklearn import manifold
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle


'''Get the data'''
pickle_in = open('/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/data/230_70_static_500.tj','rb')
obs = pickle.load(pickle_in)

for i in range(len(obs)):


tsne = manifold.TSNE(n_components=2, init='pca', random_state=0,perplexity=100)
X_tsne = tsne.fit_transform(fc)
plt.scatter(X_tsne[:,0],X_tsne[:,1],c=colors,alpha=0.2)
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off',
    left='off',
    right='off',
    labelleft='off')
plt.title('t-SNE of fc layers representations (256neurons)')

#Manually add legend (CAREFUL IF YOU CHANGE ORDER OR SMTH AS LABELS MIGHT NOT BE CONSISTENT)
classes = ['blocking','orange+green','green','orange']
class_colours = ['red','pink','green','orange']
recs = []
for i in range(0,len(class_colours)):
    recs.append(mpatches.Circle((0,0),radius=5,fc=class_colours[i]))
plt.legend(recs,classes,loc=4)
plt.savefig('t-SNE_plus_orange.png',dpi=300)
plt.show()