from pyactup import *
import random
import math

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

        result[feature] = (MP / t) * sum(fullsum[feature])

    # sorted_results = sorted(result.items(), key=lambda kv: kv[1])
    return result



def create_curated_chunk(f=1,a=1):
    chunk = {}


    for i,j in zip(range(f),range(a)):
        feature_name = 'f' + repr(i)
        feature_value = random.random()
        action_name = 'a' + repr(j)
        action_value = feature_value
        chunk[feature_name] = feature_value
        chunk[action_name] = action_value

    return chunk


def curate_chunks(n=100,features=1,targets=1):
    chunks = []
    for i in range(n):
        chunk = create_curated_chunk(f=features,a=targets)
        chunks.append(chunk)
    return chunks

def custom_similarity(x,y):
    # return math.sqrt((x-y)**2) * -1
    return abs(x - y) * - 1

observation_slots = ['f0', 'f1', 'f2']
set_similarity_function(custom_similarity, *observation_slots)

m = Memory()
MP = 3
t = 1
m = Memory(noise=0.0, decay=0.0, temperature=t, threshold=-100.0, mismatch=MP,optimized_learning=False)

chunks = curate_chunks(n=10,features=3,targets=3)
chunks = [
    {'f0':1.0, 'f1':0.0,'f2':0.0,'a0':1.0,'a1':0.0,'a2':0.0},
    {'f0':1.0, 'f1':1.0,'f2':0.0,'a0':1.0,'a1':1.0,'a2':0.0},
    {'f0':1.0, 'f1':1.0,'f2':1.0,'a0':1.0,'a1':1.0,'a2':1.0},
    {'f0':1.0, 'f1':0.0,'f2':1.0,'a0':1.0,'a1':0.0,'a2':1.0},
    {'f0':0.0, 'f1':1.0,'f2':0.0,'a0':0.0,'a1':1.0,'a2':0.0},
    {'f0':0.0, 'f1':1.0,'f2':1.0,'a0':0.0,'a1':1.0,'a2':1.0},
    {'f0':0.0, 'f1':0.0,'f2':1.0,'a0':0.0,'a1':0.0,'a2':1.0},
    {'f0':0.0, 'f1':0.0,'f2':0.0,'a0':0.0,'a1':0.0,'a2':0.0},
]

# random.shuffle(chunks)
for chunk in chunks:
    m.learn(**chunk)

m.advance(0.0)
m.activation_history = []
probe_chunk = {'f0':0.0, 'f1':1.0, 'f2':0.0}
blend_slot = 'a1'
blend_value = m.blend(blend_slot, **probe_chunk)

# activations = [chunk['retrieval_probability'] for chunk in m.activation_history]

salience = compute_S(probe_chunk, ['f0','f1','f2'],m.activation_history,blend_slot,MP,t)

print("here")




