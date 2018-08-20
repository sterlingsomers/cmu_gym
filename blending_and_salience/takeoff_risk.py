import numpy as np
import itertools
import actr
import random


feature_values = {} #the content of chunks stored as feature:[val1,val2,...,valn]



def generate_module1(needsRadio=0,needsFood=0,needsFirstaid=0,needsWater=0,bays=4):
    '''Generates chunks for take off module 1, as described in:
     https://www.dropbox.com/s/rzhdtkd8svdcpkb/3%20COGLE%20Competencies%20vs%20Situation%20Risks%20and%20Rewards.docx?dl=0#pageContainer4.
     Each provision has a success.'''
    provisions = {'radio': ['radio', 1], 'food': ['food', 1], 'firstaid': ['firstaid', 0], 'water': ['water', 0]}
    return ['isa', 'decision', 'needsRadio', int(needsRadio),
                    'needsFood', int(needsFood),
                    'needsFirstaid', int(needsFirstaid),
                    'needsWater', int(needsWater),
                    'radio', provisions['radio'],'radiosuccess',['radiosuccess',int(not(abs(needsRadio-provisions['radio'][1])))],
                    'food', provisions['food'],'foodsuccess',['foodsuccess',int(not(abs(needsFood-provisions['food'][1])))],
                    'firstaid', provisions['firstaid'],'firstaidsuccess',['firstaidsuccess',int(not(abs(needsFirstaid-provisions['firstaid'][1])))],
                    'water', provisions['water'],'watersuccess',['watersuccess',int(not(abs(needsWater-provisions['water'][1])))]]






def generate_module2(needsRadio=0,needsFood=0,needsFirstaid=0,needsWater=0,bays=4):
    '''Generates a chunk for takeoff module 2, as described in
    https://www.dropbox.com/s/rzhdtkd8svdcpkb/3%20COGLE%20Competencies%20vs%20Situation%20Risks%20and%20Rewards.docx?dl=0#pageContainer4.
    Each provision has a success'''
    provisions_needed = [needsRadio, needsFood, needsFirstaid, needsWater]
    provisions = {'radio':['radio',0],'food':['food',0],'firstaid':['firstaid',0],'water':['water',0]}

    while bays:
        #print(radio, food, fa, water, bays, provisions_needed)
        if provisions_needed[0]:
            provisions['radio'][1] = provisions['radio'][1] + 1
            provisions_needed[0] = provisions_needed[0] - 1
            bays = bays - 1
            if not bays: break
        if provisions_needed[1]:
            provisions['food'][1] = provisions['food'][1] + 1
            provisions_needed[1] = provisions_needed[1] - 1
            bays = bays - 1
            if not bays: break
        if provisions_needed[2]:
            provisions['firstaid'][1] = provisions['firstaid'][1] + 1
            provisions_needed[2] = provisions_needed[2] - 1
            bays = bays - 1
            if not bays: break
        if provisions_needed[3]:
            provisions['water'][1] = provisions['water'][1] + 1
            provisions_needed[3] = provisions_needed[3] - 1
            bays = bays - 1
            if not bays: break

        if not needsRadio or not needsFood or not needsFirstaid or not needsWater:
            return ['isa', 'decision', 'needsRadio', int(needsRadio),
                    'needsFood', int(needsFood),
                    'needsFirstaid', int(needsFirstaid),
                    'needsWater', int(needsWater),
                    'radio', provisions['radio'],'radiosuccess',['radiosuccess',1],
                    'food', provisions['food'],'foodsuccess',['foodsuccess',1],
                    'firstaid', provisions['firstaid'],'firstaidsuccess',['firstaidsuccess',1],
                    'water', provisions['water'],'watersuccess',['watersuccess',1]]

    return ['isa', 'decision', 'needsRadio', int(needsRadio),
            'needsFood', int(needsFood),
            'needsFirstaid', int(needsFirstaid),
            'needsWater', int(needsWater),
            'radio', provisions['radio'],'radiosuccess',['radiosuccess',1],
            'food', provisions['food'],'foodsuccess',['foodsuccess',1],
            'firstaid', provisions['firstaid'],'firstaidsuccess',['firstaidsuccess',1],
            'water', provisions['water'],'watersuccess',['watersuccess',1]]





#ACT-R Functions
def similarity(val1, val2):
    '''Linear tranformation, abslute difference'''
    print("similarity", val1, val2)
    slot = val2[0].lower()
    val2 = val2[1]
    print('val1',val1,'val2',val2)
    if val1 == None:
        return None
    if val1 == 0:
        print('val1 is zero')
    if val2 == 0:
        print('val2 is zero')
    #max_val = max(map(max, zip(*feature_sets)))
    #min_val = min(map(min, zip(*feature_sets)))
    #again, max and min are more complicated
    values = [x[1] for x in feature_values[slot]]
    min_val = min(values)
    max_val = max(values)
    max_val += EPS
    print('min', min_val, 'max', max_val)
    #print("max,min,val1,val2",max_val,min_val,val1,val2)
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




#generate chunks
#can substitute module1 for module2
allcombinations = []
lst = list(itertools.product([0,1],repeat=4))
for item in lst:
    allcombinations.append(generate_module1(*item))


for chk in allcombinations:
    print(chk)


# now go through those, and make the feature_values
# need to go through each chunk by 2s (slot, value)
# from https://stackoverflow.com/questions/2990121/how-do-i-loop-through-a-list-by-twos
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


# chunks = [['isa','decision','value1',['value1',1.5],'output',['output',1],'value2',['value2',1]],
#         ['isa','decision','value1',['value1',0],'output',['output',0],'value2',['value2',0]],
#          ['isa', 'decision', 'value1', ['value1', 1], 'output', ['output', 1], 'value2', ['value2', 1]]]
# chunks = chunks[0:5]
for chunk in allcombinations:
    #print(chunk)
    for slot, value in grouper(2, chunk):
        #print('slot', slot, 'value', value, 'features', feature_values)
        if not slot in feature_values:
            feature_values[slot] = []
        feature_values[slot].append(value)

        # do the chunks need to change format?
        # they should be slot:[slot, value]
#print("FV", feature_values)


#setup ACT-R
actr.add_command('similarity_function',similarity)



#load an ACT-R model
actr.load_act_r_model("/Users/paulsomers/blendingtests/takeoff-risk.lisp")
actr.record_history("blending-trace")



#load the chunks into actr
for chk in allcombinations:
    actr.add_dm(chk)


#generate a probe
probe = random.choice(allcombinations)
probe[1] = 'observation'
probe.append('actual')
probe.append(35)
print("probe", probe)



#push the probe into the imaginal buffer
probe = actr.define_chunks(probe)
actr.schedule_simple_event_now("set-buffer-chunk", ['imaginal',probe[0]])


#run ACT-R
actr.run(10)