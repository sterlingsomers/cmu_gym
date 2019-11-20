from pyactup import *
import random
import inspect
import math
from collections import OrderedDict
import scipy.stats as stats
import numpy as np

f_low, f_high = 0, 1
f_mu = 0
f_sigma = 2
f_dist = stats.truncnorm((f_low-f_mu)/f_sigma, (f_high-f_mu)/f_sigma, loc=f_mu, scale=f_sigma)

def compute_S(probe,feature_list,history,Vk,MP,t):
    chunk_names = []

    PjxdSims = {}
    for feature in feature_list:
        Fk = probe[feature]
        for chunk in history:
            dSim = None
            vjk = None
            for attribute in chunk['attributes']:
                if attribute[0] == feature:
                    vjk = attribute[1]
                    break

            if Fk == vjk:
                dSim = 0.0
            else:
                dSim = (vjk - Fk) / abs(Fk - vjk)
            # if Fk == vjk:
            #     dSim = 0
            # else:
            #     dSim = -1 * ((Fk-vjk) / math.sqrt((Fk - vjk)**2))

            Pj = chunk['retrieval_probability']
            if not feature in PjxdSims:
                PjxdSims[feature] = []
            PjxdSims[feature].append(Pj*dSim)
            pass

    # vio is the value of the output slot
    fullsum = {}
    result = {}  # dictionary to track feature
    Fk = None
    for feature in feature_list:
        Fk = probe[feature]
        if not feature in fullsum:
            fullsum[feature] = []
        inner_quantity = None
        Pi = None
        vio = None
        dSim = None
        vik = None
        for chunk in history:
            Pi = chunk['retrieval_probability']
            for attribute in chunk['attributes']:
                if attribute[0] == Vk:
                    vio = attribute[1]

            for attribute in chunk['attributes']:
                if attribute[0] == feature:
                    vik = attribute[1]
            # if Fk > vik:
            #     dSim = -1
            # elif Fk == vik:
            #     dSim = 0
            # else:
            #     dSim = 1
            # dSim = (Fk - vjk) / sqrt(((Fk - vjk) ** 2) + 10 ** -10)
            if Fk == vik:
                dSim = 0.0
            else:
                dSim = (vik - Fk) / abs(Fk - vik)
            #
            # if Fk == vik:
            #     dSim = 0
            # else:
            #     dSim = -1 * ((Fk-vik) / math.sqrt((Fk - vik)**2))

            inner_quantity = dSim - sum(PjxdSims[feature])
            fullsum[feature].append(Pi * inner_quantity * vio)

        result[feature] = sum(fullsum[feature])

    # sorted_results = sorted(result.items(), key=lambda kv: kv[1])
    return result



def create_curated_chunk(f=1,a=1):
    chunk = {}

    possible_values = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,
                       0.35,0.4,0.45,0.5,0.55,0.6,0.65,
                       0.7,0.75,0.8,0.85,0.9,0.95,1.0]
    for i,j in zip(range(f),range(a)):
        feature_name = 'f' + repr(i)
        feature_value = random.choice(possible_values)
        action_name = 'a' + repr(j)
        action_value = feature_value - random.randint(0,40*i**2)/100
        chunk[feature_name] = feature_value
        chunk[action_name] = action_value

    return chunk


def curate_chunks(n=100,features=1,targets=1):
    chunks = []

    for i in range(n):
        chunk = create_curated_chunk(f=features,a=targets)
        chunks.append(chunk)
    return chunks

def curate_linear_chunks(n=100,weights=[1.0],values=[0,1],vector=0,targets=1):
    # possible_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    #                    0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
    #                    0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

    chunks = []
    for i in range(n):
        chunk = {}
        feature_names = []
        for v in range(len(weights)):
            feature_name = 'f' + repr(v)
            feature_names.append(feature_name)
            feature_value = random.choice(values)
            chunk[feature_name] = feature_value
        target_value = 0.0
        for feat,weight in zip(feature_names,weights):
            target_value += chunk[feat] * weight
        chunk['t0'] = target_value
        chunks.append(chunk)
    return chunks


def curate_function_chunks(n=100,func=lambda f1, f2, f3: f1 + f2 + f3, vector=0,values=0):
    chunks = []
    vector_chunks = []
    vector_targets = []
    for i in range(n):
        chunk = OrderedDict()
        params = list(inspect.signature(func).parameters)
        for factor in params:
            if not values:
                factor_value = random.random()#random.choice(values)
            else:
                factor_value = random.choice(values)
            chunk[factor] = factor_value
        target_value = func(**chunk)
        if type(target_value) == list:
            raise ValueError("Cannot unpack targets yet...")
        chunk['t0'] = target_value
        chunks.append(chunk)
        if vector:
            if vector == 1:
                vector_chunks.append(list(chunk.values()))
            elif vector == 2:
                feats = [x[1] for x in list(chunk.items()) if 't' not in x[0]]
                vals = [x[1] for x in list(chunk.items()) if 't' in x[0]]
                vector_chunks.append(feats)
                vector_targets.append(vals)
    if not vector:
        return chunks, None, None
    if vector == 1:
        return chunks, vector_chunks, None
    if vector == 2:
        return chunks, vector_chunks, vector_targets



def non_linear_salience(probe,feature_list,history,Vk,Vo,MP,t):
    #first we will collect Pj * dSim(fk,vjk) into a list to sum later
    result_by_feature = {}
    Pj_x_dSimFK_vjk = {}
    for feature in feature_list:
        Fk = probe[feature]
        for chunk in history:
            dSim = None
            vjk = None
            for attribute in chunk['attributes']:
                if attribute[0] == feature:
                    vjk = attribute[1]
                    break

            if Fk == vjk:
                dSim = 0.0
            else:
                dSim = (vjk - Fk) / abs(Fk - vjk)
            # if Fk == vjk:
            #     dSim = 0
            # else:
            #     dSim = -1 * ((Fk-vjk) / math.sqrt((Fk - vjk)**2))

            Pj = chunk['retrieval_probability']
            if not feature in Pj_x_dSimFK_vjk:
                Pj_x_dSimFK_vjk[feature] = []
            Pj_x_dSimFK_vjk[feature].append(Pj * dSim)
    #Now, for every feature, Pj_x_dSimFk_vjk has a list that needs to be summed.

    numerators_by_feature = {}
    denomenators_by_feautre = {}

    for feature in feature_list:
        Fk = probe[feature]
        SimVo_vio = None
        dSimVo_vio = None
        Pi = None

        for chunk in history:
            vio = None
            vik = None
            dSimFk_vik = None
            Pi = chunk['retrieval_probability']
            for attribute in chunk['attributes']:
                if attribute[0] == Vk: #Vk is the blend slot, vio is the value of the output slot
                    vio = attribute[1]
                if attribute[0] == feature:
                    vik = attribute[1]
            SimVo_vio = custom_similarity(Vo,vio)
            if SimVo_vio < 0:
                print("--")
            if Vo == vio:
                dSimVo_vio = 0.0
            else:
                dSimVo_vio = (vio - Vo) / abs(Vo - vio)

            #dSimFk_vik
            if Fk == vik:
                dSimFk_vik = 0.0
            else:
                dSimFk_vik = (vik - Fk) / abs(Fk - vik)

            if not feature in numerators_by_feature:
                numerators_by_feature[feature] = []
                denomenators_by_feautre[feature] = []


            numerators_by_feature[feature].append(Pi * (dSimFk_vik - sum(Pj_x_dSimFK_vjk[feature]) * SimVo_vio * dSimVo_vio))
            denomenators_by_feautre[feature].append(Pi * (dSimVo_vio)**2) #second derivative is 0

        result_by_feature[feature] = -MP/t * (sum(numerators_by_feature[feature]) / sum(denomenators_by_feautre[feature]))

    return result_by_feature




def custom_similarity(x,y):
    # return math.sqrt((x-y)**2) * -1
    result = 1 - abs(x - y) # 1 - x == y
    return result

observation_slots = ['f0', 'f1', 'f2']
set_similarity_function(custom_similarity, *observation_slots)

m = Memory()
MP = 1
t = 1
random.seed(35)
m = Memory(noise=0.0, decay=0.0, temperature=t, threshold=-100.0, mismatch=MP,optimized_learning=False)

#chunks = curate_chunks(n=10000,features=3,targets=3)
# chunks = [
#     {'f0':1.0, 'f1':0.0,'f2':0.0,'a0':1.0,'a1':0.0,'a2':0.0},
#     {'f0':1.0, 'f1':1.0,'f2':0.0,'a0':1.0,'a1':1.0,'a2':0.0},
#     {'f0':1.0, 'f1':1.0,'f2':1.0,'a0':1.0,'a1':1.0,'a2':1.0},
#     {'f0':1.0, 'f1':0.0,'f2':1.0,'a0':1.0,'a1':0.0,'a2':1.0},
#     {'f0':0.0, 'f1':1.0,'f2':0.0,'a0':0.0,'a1':1.0,'a2':0.0},
#     {'f0':0.0, 'f1':1.0,'f2':1.0,'a0':0.0,'a1':1.0,'a2':1.0},
#     {'f0':0.0, 'f1':0.0,'f2':1.0,'a0':0.0,'a1':0.0,'a2':1.0},
#     {'f0':0.0, 'f1':0.0,'f2':0.0,'a0':0.0,'a1':0.0,'a2':0.0},
#     {'f0':0.5, 'f1':0.0,'f2':0.0,'a0':0.5,'a1':0.0,'a2':0.0},
#     {'f0':0.5, 'f1':0.5,'f2':0.0,'a0':0.5,'a1':0.5,'a2':0.0},
#     {'f0':0.5, 'f1':0.5,'f2':0.5,'a0':0.5,'a1':0.5,'a2':0.5},
#     {'f0':0.5, 'f1':0.0,'f2':0.5,'a0':0.5,'a1':0.0,'a2':0.5},
#     {'f0':0.0, 'f1':0.5,'f2':0.0,'a0':0.0,'a1':0.5,'a2':0.0},
#     {'f0':0.0, 'f1':0.5,'f2':0.5,'a0':0.0,'a1':0.5,'a2':0.5},
# ]
#constantinos example
# chunks = [{'f0':1.0, 'f1':0.0, 'a0':0.0, 'a1':1.0},
#          {'f0':0.0, 'f1':1.0, 'a0':1.0, 'a1':0.0},
#           {'f0':1.0, 'f1':0.0, 'a0':0.0, 'a1':1.0}
# ]

#linear function chunks
# chunks = curate_linear_chunks(n=10000,weights=[0.25,0.5,1.0],values=[0.0,0.1,
#                                                                     0.05,0.15,0.2,0.25,0.3,0.35,
#                                                                     0.4,0.45,0.5,0.55,0.6,
#                                                                     0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0])
funct = lambda f0, f1, f2: f0 + f1 + f2
chunks, vector_features, vector_targets = curate_function_chunks(n=10000, func=funct,values=[0,1], vector=2)

# chunks = [{'f0':0,'f1':0, 't0':0},
#           {'f0':0,'f1':1, 't0':1},
#           {'f0':1,'f1':0, 't0':1},
#           {'f0':1,'f1':1, 't0':2}]

# random.shuffle(chunks)
for chunk in chunks:
    m.learn(**chunk)

m.advance(0.01)
m.activation_history = []

#generate probe
# probe_chunk = curate_function_chunks(n=1, func=funct, values=[0.0,0.1,0.2,0.3,0.4,0.5,
#                                                             0.6,0.7,0.8,0.9,1.0])[0]
# del probe_chunk['t0']

#manual probe :)
probe_chunk = {'f0':0.1, 'f1':0.1, 'f2':0.1, 'f3':0.1, 'f4':0.1, 'f5':0.1, 'f6':0.1, 'f7':0.1}
#probe_chunk = {'f0':1.0, 'f1':1.0}#, 'f2':1.0}
#constantinos example
# probe_chunk = {'f0':1.0, 'f1':0.0}


blend_slot = 't0'

###ACTUP
blend_value = m.blend(blend_slot, **probe_chunk)

# activations = [chunk['retrieval_probability'] for chunk in m.activation_history]
salience = compute_S(probe_chunk, [x for x in list(probe_chunk.keys()) if not x == blend_slot],m.activation_history,blend_slot,MP,t)
# salience2 = non_linear_salience(probe_chunk, [x for x in list(probe_chunk.keys()) if not x == blend_slot],m.activation_history,blend_slot,blend_value,MP,t)

t0 = m.blend('t0', **probe_chunk)
# a1 = m.blend('a1', **probe_chunk)
# a2 = m.blend('a2', **probe_chunk)

print('probe')
print(probe_chunk)
print('blend')
print(t0)
print('salience')
for t in salience:
    print(t,salience[t])
# for t in salience2:
#     print(t,salience2[t])
###END ACTUP


#vectored version
b = list(probe_chunk.values())

dec = np.array(vector_targets)
M = np.array(vector_features)

match = np.sum(np.abs(b-M),axis=1)

Ps = np.exp(-match) / sum(np.exp(-match))

dM = np.sign(b-M)

P = Ps
P = P.reshape((P.size,1))

Pu = P*dec

PdM = P * dM

sumPdM = np.sum(PdM,axis=0)

dM_sumPdM = dM - sumPdM

print("VECTOR IMPLEMENTATION")
print("vector salience", np.dot(Pu[:,0],dM_sumPdM))
print('blend', np.matmul(P.reshape(len(chunks)),dec))




