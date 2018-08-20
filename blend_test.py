import json
from blending_and_salience import actr
import numpy as np
import random
import itertools

#example from whiteboard start

not_dicts = ['FILLED-SLOTS','EMPTY-SLOTS','BLENDED-SLOTS','IGNORED-SLOTS','RESULT-CHUNK','MATCHED-CHUNK-VALUES',
             'MATCHED-VALUES-MAGNITUDES']

similarities_by_time = {}

def choose(items,chances):
    p = chances[0]
    x = random.random()
    i = 0
    while x > p :
        i = i + 1
        p = p + chances[i]
    return items[i]

def access_by_key(key, list):
    '''Assumes key,vallue pairs and returns the value'''
    if not key in list:
        raise KeyError("Key not in list")

    return list[list.index(key)+1]


def y_value(*args):
    print("y_value", args)
    return a*args[0] + b*args[1]# + c*args[2]


def new_blend_request(args):
    '''Add a new time key for similarities'''
    similarities_by_time[actr.get_time()] = []

def similarity(val1, val2):
    '''Linear tranformation, abslute difference'''
    if val1 == None:
        return None
    max_val = max(map(max, zip(*feature_sets)))
    min_val = min(map(min, zip(*feature_sets)))
    print("max,min,val1,val2",max_val,min_val,val1,val2)
    val1_t = (((val1 - min_val) * (0 + 1)) / (max_val - min_val)) + 0
    val2_t = (((val2 - min_val) * (0 + 1)) / (max_val - min_val)) + 0
    #print("val1_t,val2_t", val1_t, val2_t)
    #print("sim returning", abs(val1_t - val2_t) * -1)
    #print("sim returning", ((val1_t - val2_t)**2) * - 1)
    #return float(((val1_t - val2_t)**2) * - 1)
    #return abs(val1_t - val2_t) * - 1
    #return 0
    #print("sim returning", abs(val1_t - val2_t) * - 1)
    #return abs(val1_t - val2_t) * -1
    print("sim returning", (abs(val1 - val2) * - 1)/max_val)
    return (abs(val1 - val2) * - 1)/max_val

    print("sim returning", abs(val1 - val2) / (max_val - min_val) * - 1)
    return abs(val1 - val2) / (max_val - min_val) * - 1


actr.add_command('similarity_function',similarity)
#actr.add_command('new_blend_request', new_blend_request)
actr.load_act_r_model("/Users/paulsomers/blendingtests/salience-test.lisp")
actr.record_history("blending-trace")


#make up some chunks and add them to memory
#y = af + bf + cf
a, b, c = 2, 1, 1

#get 3 fs from normal
mu = 5
sigma = 0.5
feature_sets = np.random.normal(mu,sigma,(10,3))

feature_sets = [[0,0],
                [0,1],
                [1,0],
                [1,1]]

def generate_module2(radio, food, fa, water, bays=4):
    provisions_needed = [radio, food, fa, water]
    provisions = {'radio': 0, 'food':0, 'firstaid':0, 'water':0}
    while bays:
        print(radio, food, fa, water, bays, provisions_needed)
        if provisions_needed[0]:
            provisions['radio'] = provisions['radio'] + 1
            provisions_needed[0] = provisions_needed[0] - 1
            bays = bays - 1
            if not bays: break
        if provisions_needed[1]:
            provisions['food'] = provisions['food'] + 1
            provisions_needed[1] = provisions_needed[1] - 1
            bays = bays - 1
            if not bays: break
        if provisions_needed[2]:
            provisions['firstaid'] = provisions['firstaid'] + 1
            provisions_needed[2] = provisions_needed[2] - 1
            bays = bays - 1
            if not bays: break
        if provisions_needed[3]:
            provisions['water'] = provisions['water'] + 1
            provisions_needed[3] = provisions_needed[3] - 1
            bays = bays - 1
            if not bays: break

        if not radio or not food or not fa or not water:
            return ['isa', 'decision', 'needsRadio', radio,
                    'needsFood', food,
                    'needsFirstaid', fa,
                    'needsWater', water,
                    'radio', provisions['radio'],
                    'food', provisions['food'],
                    'firstaid', provisions['firstaid'],
                    'water', provisions['water']]

    return ['isa', 'decision', 'needsRadio', int(radio),
            'needsFood', int(food),
            'needsFirstaid', int(fa),
            'needsWater', int(water),
            'radio',int(provisions['radio']),
            'food', int(provisions['food']),
            'firstaid', int(provisions['firstaid']),
            'water', int(provisions['water'])]

chks = []
#print(generate_module2(0,0,0,1))
for i in range(100):
    chks.append(generate_module2(*np.random.randint(3,size=4)))

#mark examples
#chks = []

#chks.append(['isa', 'decision', 'needsFood', 0, 'needsWater', 0, 'food', 0, 'water', 0])
#chks.append(['isa', 'decision', 'needsFood', 0, 'needsWater', 1, 'food', 0, 'water', 2])
#chks.append(['isa', 'decision', 'needsFood', 1, 'needsWater', 0, 'food', 2, 'water', 0])
#chks.append(['isa', 'decision', 'needsFood', 1, 'needsWater', 1, 'food', 1, 'water', 1])

# chks.append(['isa', 'decision', 'needsRadio', 0, 'needsFood', 0, 'needsFirstaid', 0, 'needsWater', 0, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 0, 'needsFood', 0, 'needsFirstaid', 0, 'needsWater', 1, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 0, 'needsFood', 0, 'needsFirstaid', 1, 'needsWater', 0, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 0, 'needsFood', 0, 'needsFirstaid', 1, 'needsWater', 1, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 1, 'needsFood', 1, 'needsFirstaid', 1, 'needsWater', 1, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 0, 'needsFood', 1, 'needsFirstaid', 0, 'needsWater', 0, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 0, 'needsFood', 1, 'needsFirstaid', 0, 'needsWater', 1, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 0, 'needsFood', 1, 'needsFirstaid', 1, 'needsWater', 1, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 1, 'needsFood', 0, 'needsFirstaid', 0, 'needsWater', 0, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 1, 'needsFood', 0, 'needsFirstaid', 0, 'needsWater', 1, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 1, 'needsFood', 0, 'needsFirstaid', 1, 'needsWater', 0, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 1, 'needsFood', 0, 'needsFirstaid', 1, 'needsWater', 1, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 1, 'needsFood', 1, 'needsFirstaid', 0, 'needsWater', 0, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 0, 'needsFood', 0, 'needsFirstaid', 0, 'needsWater', 0, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 1, 'needsFood', 1, 'needsFirstaid', 0, 'needsWater', 1, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])
# chks.append(['isa', 'decision', 'needsRadio', 1, 'needsFood', 1, 'needsFirstaid', 1, 'needsWater', 0, 'radio', 1, 'food', 1, 'firstaid', 0, 'water', 0])


chks = ['isa', 'decision', 'needsRadio', ['needsRadio', 1], 'needsFirstaid', ['needsFirstaid', 1],
        'needsFood', ['needsFood', 1], 'needsWater', ['needsWater', 1],
        'food', ['food', 1], 'water', ['water', 1], 'firstaid', ['firstaid', 1], 'radio', ['radio', 1]]
actr.add_dm(chks)
#for x in chks:
#    x = [int(n) if isinstance(n, np.int64) else n for n in x]
#    actr.add_dm(x)

# feature_sets = [[0,0,0],
#                 [0,0,1],
#                 [0,1,0],
#                 [0,1,1],
#                 [1,0,0],
#                 [1,0,1],
#                 [1,1,0],
#                 [1,1,1]]

# choices = [1,0]
#
# def choose(items,chances):
#     import random
#     p = chances[0]
#     x = random.random()
#     i = 0
#     while x > p :
#         i = i + 1
#         p = p + chances[i]
#     return items[i]
#
# chosen = []
# for n in range(10):
#     tmp = []
#
#     item = choose(choices,[0.75,0.25])
#     tmp.append(item)
#     tmp.extend([y for y in choices if y not in tmp])
#     chosen.append(tmp)
# print("chosen", chosen)

#feature_sets = chosen
#a,b,c = 1,1,1

chunks = []
#random value vesion
# #create chuns from thsoe random feature sets
# for fset in feature_sets:
#     actr.add_dm(['isa', 'decision', 'f1', fset[0], 'f2', fset[1],# 'f3', fset[2],
#                  #   'value',y_value(fset[0],fset[1],fset[2])])
#                  'value', y_value(fset[0], fset[1])])
#
# chk = ['isa', 'observation', 'f1', 0.8, 'f2', 0.8, 'actual', y_value(0.8,0.8)]





#two terms
#actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 0, 'value', 0])

#three terms
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 0, 'f3', 0, 'value', 0])
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 0, 'f3', 1, 'value', 1])
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 1, 'f3', 0, 'value', 1])
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 1, 'f3', 1, 'value', 2])
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 0, 'f3', 0, 'value', 0])
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 0, 'f3', 1, 'value', 1])
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 1, 'f3', 0, 'value', 1])
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 1, 'f3', 1, 'value', 2])

#need a random set to use
#and push it into imaginal
# feature_example = np.random.normal(mu,sigma-0.1,3)
# chk = ['isa', 'observation', 'f1', feature_example[0],
#        'f2', feature_example[1],
#        'f3', feature_example[2],
#        'actual',y_value(feature_example[0],feature_example[1],feature_example[2])]


# #probability version
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 0, 'value', 0])
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 0, 'value', 0])
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 0, 'value', 0])
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 0, 'value', 0])
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 0, 'value', 0])
#
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 0, 'value', 1])
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 0, 'value', 1])
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 0, 'value', 1])
#
#
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 0, 'value', 0])
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 0, 'value', 1])
#
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 1, 'value', 1])
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 1, 'value', 1])
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 1, 'value', 1])
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 1, 'value', 1])
# actr.add_dm(['isa', 'decision', 'f1', 1, 'f2', 1, 'value', 1])
#
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 1, 'value', 0])
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 1, 'value', 0])
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 1, 'value', 0])
#
#
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 1, 'value', 1])
# actr.add_dm(['isa', 'decision', 'f1', 0, 'f2', 1, 'value', 0])
#
# chk = ['isa', 'observation', 'f1', 0.5, 'f2', 0.5, 'actual', 1]#, 'f3', 0, 'actual', 1]





#for mark
chk = ['isa', 'observation', 'needsRadio', 1, 'needsFood', 1, 'needsFirstaid', 1, 'needsWater', 1, 'actual', 1]
#chk = ['isa', 'observation', 'needsFood',1, 'needsWater', 1, 'actual', 1]
chunk = actr.define_chunks(chk)
actr.schedule_simple_event_now("set-buffer-chunk",
                               ['imaginal', chunk[0]])
actr.run(10)

d = actr.get_history_data("blending-trace")
d = json.loads(d)

asdf = actr.get_history_data("blending-times")


def compute_S(blend_trace, keys_list):
    '''For blend_trace @ time'''
    #probablities
    probs = [x[3] for x in access_by_key('MAGNITUDES',access_by_key('SLOT-DETAILS',blend_trace[0][1])[0][1])]
    #feature values in probe
    FKs = [access_by_key(key.upper(),access_by_key('RESULT-CHUNK',blend_trace[0][1])) for key in keys_list]
    chunk_names = [x[0] for x in access_by_key('CHUNKS', blend_trace[0][1])]

    #Fs is all the F values (may or may not be needed for tss)
    #They are organized by chunk, same order as probs
    vjks = []
    for name in chunk_names:
        chunk_fs = []
        for key in keys_list:
            chunk_fs.append(actr.chunk_slot_value(name,key))
        vjks.append(chunk_fs)

    #tss is a list of all the to_sum
    #each to_sum is Pj x dSim(Fs,vjk)/dFk
    #therefore, will depend on your similarity equation
    #in this case, we need max/min of the features because we use them to normalize
    max_val = max(map(max, zip(*feature_sets)))
    min_val = min(map(min, zip(*feature_sets)))
    n = max_val - min_val
    n = max_val
    #n = 1
    #this case the derivative is:
    #           Fk > vjk -> -1/n
    #           Fk = vjk -> 0
    #           Fk < vjk -> 1/n
    #compute Tss
    #there should be one for each feature
    #you subtract the sum of each according to (7)
    tss = {}
    ts2 = []
    for i in range(len(FKs)):
        if not i in tss:
            tss[i] = []
        for j in range(len(probs)):
            if FKs[i] > vjks[j][i]:
                dSim = -1/n
            elif FKs[i] == vjks[j][i]:
                dSim = 0
            else:
                dSim = 1/n
            tss[i].append(probs[j] * dSim)
        ts2.append(sum(tss[i]))

    #vios
    viosList = []
    viosList.append([actr.chunk_slot_value(x,'radio') for x in chunk_names])
    viosList.append([actr.chunk_slot_value(x,'food') for x in chunk_names])
    viosList.append([actr.chunk_slot_value(x, 'water') for x in chunk_names])
    viosList.append([actr.chunk_slot_value(x, 'firstaid') for x in chunk_names])
    #compute (7)
    rturn = []
    for vios in viosList:
        results = []
        for i in range(len(FKs)):
            tmp = 0
            sub = []
            for j in range(len(probs)):
                if FKs[i] > vjks[j][i]:
                    dSim = -1/n
                elif FKs[i] == vjks[j][i]:
                    dSim = 0
                else:
                    dSim = 1/n
                tmp = probs[j] * (dSim - ts2[i]) * vios[j]#sum(tss[i])) * vios[j]
                sub.append(tmp)
            results.append(sub)

        print("compute S complete")
        rturn.append(results)
    return rturn


   #print("compute_S complete")





#compute S(Vo,Fk)
#Sum(P * der/wrt-k)

#get the probabilities from the trace
#Assumes the right structure from the list (may need to find another way)
#list of chunks with probabilities ['CHUNK', chunk_name, 'probability', p_value, 'value', value, 'increment', increment']

#for chunk_trace in access_by_key('MAGNITUDES',access_by_key('SLOT-DETAILS',d[0][1])[0][1]):
    #need the derivative of the similarity equation
    #similarity is (roughly) |x - y|
    # x - y
    #-------
    #|y - x|    #wolframalpha
#    pass
# #ordered probablities
# probs = [x[3] for x in access_by_key('MAGNITUDES',access_by_key('SLOT-DETAILS',d[0][1])[0][1])]
# #can get the feature values from the result-chunk as it will be the same as the probe chunk
# F1K = access_by_key('F1',access_by_key('RESULT-CHUNK',d[0][1]))
# #have to get the f-values for each chunk manually, by name
# #chunk_names has the same order as probs. zip them if needed.
# chunk_names = [x[0] for x in access_by_key('CHUNKS',d[0][1])]
# #now get the f1 values for each chunks
# f1s = [actr.chunk_slot_value(x,'f1') for x in chunk_names]
# #Pj * dSim(Fk,vjk) 'Pj'
# to_sum = [p*((F1K - y)/abs(F1K - y)) for p,y in zip(probs,f1s)]
# sumPj = sum(to_sum)
# #get MP
MP = actr.get_parameter_value(':mp')
# #get t
t = access_by_key('TEMPERATURE',d[0][1])
# #the values
# vs = [actr.chunk_slot_value(x,'value') for x in chunk_names]
#
factors = ['needsRadio','needsFood', 'needsWater', 'needsFirstaid']
#factors = ['needsFood', 'needsWater']
result_factors = ['radio','food','water','firstaid']
#result_factors = ['food','water']
results = compute_S(d, factors)#,'f3'])
for sums,result_factor in zip(results,result_factors):
    print("For", result_factor)
    for s,factor in zip(sums,factors):
        print(factor, MP/t * sum(s))

print("actual value is", actr.chunk_slot_value('OBSERVATION0','ACTUAL'))

print("done")


