import pickle

import fnmatch
import os

from scipy.stats import entropy
from scipy.spatial import distance
datas = []
for file in os.listdir('.'):
    if fnmatch.fnmatch(file, 'MODEL_*FC*'):
        datas.append(pickle.load(open(file,'rb')))


# data = pickle.load(open('MODEL_TRACE_eBEHAVE_FC_noise030_MP3_26-35.tj','rb'))

kls = []
JSs = []
for data in datas:
    for map in data:
        for mission in data[map]:
            if type(mission) == str:
                continue
            for dist_net,dist_actr in zip(data[map][mission]['action_probs'],data[map][mission]['actr_actions']):
                dist_actr[dist_actr == 0] = 0.00000001
                dist_net[dist_net == 0] = 0.00000001
                single_kl = entropy(dist_net[0],dist_actr[0])
                single_js = distance.jensenshannon(dist_actr[0],dist_net[0])

                kls.append(single_kl)
                JSs.append(single_js)


k = len(kls)
avg_divergence = (1/(k*(k-1))) * sum(kls)




print("done")