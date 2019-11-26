import scipy.stats as stats
from collections import OrderedDict
import inspect
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import Pool
from functools import partial
import dill as pickle

from pyactup import *

#The aim here is to create a random data set
#then create some random chunks to test the blend
#and test the variance on the salience (which we should know from ground truth)
#what we want to test is how sensitive it is to diversion of the probes to the mean
#so, we can chose a MU at the (known) mean (because the mu of the data set is known)
#and change the SD of the

def create_truncnorm_distribution(low, high, mu, sigma, size):
    distribution = stats.truncnorm((low - mu) / sigma, (high - mu) / sigma, loc=mu, scale=sigma)
    values = distribution.rvs(size)
    return values

def curate_function_chunks(n=0, func=0, distribution='uniform', distribution_args={},chunks=True, vectors=0):
    #Create a truncated normal distribution around mu and sigma
    chunk_data = []
    distribution_map = {'uniform':np.random.uniform,
                        'truncnorm':create_truncnorm_distribution}

    #values = distribution_map[distribution](**distribution_args)
    #feature_distribution = stats.truncnorm((min - mu) / sigma, (max - mu) / sigma, loc=mu, scale=sigma)
    # plot_values = feature_distribution.rvs(1000)
    # fig, ax = plt.subplots(1,1)
    # x = np.linspace(stats.truncnorm.ppf(0.01, min, max), stats.truncnorm.ppf(0.99, min, max), 100)
    # ax.plot(x, feature_distribution.pdf(x), 'k-', lw=2, label='frozen')
    # vals = stats.truncnorm.ppf([0.001, 0.5, 0.999], min, max)
    # np.allclose([0.001, 0.5, 0.999], stats.truncnorm.cdf(vals, min, max))
    # r = feature_distribution.rvs(5000)
    # ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    # plt.show()
    # params = list(inspect.signature(func).parameters)
    # factor_values = {}
    # for factor in params:
    #     factor_values[factor] = feature_distribution.rvs(n).tolist()
    # print("here")
    #HOW MANY factors are there?
    params = list(inspect.signature(func).parameters)
    n_factors = len(params)
    distribution_args['size'] = (n,n_factors)
    factor_values = distribution_map[distribution](**distribution_args)
    if chunks:
        for i in range(n):
            chunk = dict(zip(params,factor_values[i]))
            chunk['t'] = func(*factor_values[i].tolist())
            chunk_data.append(chunk)
    return chunk_data

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

def curate_probes(n=1000, func=lambda f1, f2, f3: f1 + f2 + f3,distribution='truncnorm', distribution_args={'low':-1, 'high': 1, 'mu':0,'sigma':0.1,}, chunks=True):
    chunk_data = []
    params = list(inspect.signature(func).parameters)
    n_factors = len(params)
    distribution_map = {'uniform': np.random.uniform,
                        'truncnorm': create_truncnorm_distribution}
    distribution_args['size'] = (n,n_factors)
    factor_values = distribution_map[distribution](**distribution_args)
    if chunks:
        for i in range(n):
            chunk = dict(zip(params, factor_values[i]))
            chunk_data.append(chunk)
    return chunk_data

def multi_blends(chunk, memory, slots ):
    '''Use this for multiprocessing, calling the blending function for each chunk it its pool'''
    return [memory.blend(slot,**chunk) for slot in slots]

def multi_blends_salience(probe, observation_slots, action_slots, chunks, noise, decay, temperature, threshold, mismatch):
    m = Memory(noise=noise, decay=decay, temperature=temperature, threshold=threshold, mismatch=mismatch, optimized_learning=False)
    for chunk in chunks:
        m.learn(**chunk)
    m.advance()
    m.activation_history = []

    # observation_slots = list(inspect.signature(funct).parameters)
    # action_slots = [x for x in chunks[0] if not x in observation_slots]
    print("probe", probe)
    blend_values = []
    salience_values = []
    funct = lambda f0, f1, f2: 2 * f0 + f1 + f2
    GT = []
    for slot in action_slots:
        blend_value = m.blend(slot, **probe)
        print('blend', blend_value)
        blend_values.append(blend_value)
        salience_values = compute_S(probe, [x for x in list(probe.keys()) if not x == slot],m.activation_history,slot,mismatch,temperature)
        GT.append(funct(**probe))
    print(blend_values,GT,salience_values)
    return [blend_values,GT,salience_values]



def custom_similarity(x,y):
    return 1 - abs(x - y)

if __name__ == "__main__":
    funct = lambda f0, f1, f2: 2*f0 + f1 + f2 #the function I want the data to represent
    ACT_params = {'MP':1, 't':1}

    data = curate_function_chunks(n=10000, func=funct,distribution='uniform', distribution_args={'low':0, 'high':1},chunks=True)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("2f0 + f1 + f2")
    for sigma in [0.2,0.4,0.5]:
        probes = curate_probes(n=100, func=funct, distribution='truncnorm', distribution_args={'low':0, 'high':1, 'mu':0.5, 'sigma':sigma}, chunks=True)

        observation_slots = list(inspect.signature(funct).parameters)
        action_slots = [x for x in data[0] if not x in observation_slots]
        set_similarity_function(custom_similarity, *observation_slots)


        ###single_processing
        # m = Memory(noise=0.0, decay=0.0, temperature=ACT_params['t'], threshold=-100.0, mismatch=ACT_params['MP'], optimized_learning=False)
        # for chunk in data:
        #     m.learn(**chunk)
        #
        # m.advance()
        # results = []
        # GT = []
        # for probe in probes:
        #     for slot in action_slots:
        #         results.append(m.blend(slot,**probe))
        #         GT.append(funct(**probe))
        blend_results = []
        blend_sd_results = []
        salience_mean_results = []
        salience_sd_results = []
        mismatch_penalties = []
        for i in [1,5,10]:#range(1,6):
        ###multi processing
            p = Pool(processes=8)
            #multi_p = partial(multi_blends, memory=m, slots=action_slots)
            multi_p = partial(multi_blends_salience, observation_slots=observation_slots, action_slots=action_slots, chunks=data, noise=0.0, decay=0.0, temperature=1.0, threshold=-100.0, mismatch=i)
            results = p.map(multi_p, probes)
            mean_difference_blends = np.mean([abs(result[0][0] - result[1][0]) for result in results])
            blend_results.append(mean_difference_blends)
            sd_blends = np.std([abs(result[0][0] - result[1][0]) for result in results])
            blend_sd_results.append(sd_blends)
            salience_dictionaries = [result[2] for result in results]
            salience_values = [sd[os] for sd in salience_dictionaries for os in observation_slots]
            salince_values = np.asarray(salience_values,dtype=np.float32)
            salience_values_array = np.reshape(salience_values, (-1,3))
            salience_mean_results.append(np.mean(salience_values_array,axis=0))
            salience_sd_results.append(np.std(salience_values_array,axis=0))
            mismatch_penalties.append(i)





        # blends = [abs(result[0][0]-result[1][0]) for result in results]
        # x = range(1,len(blends)+1)
        #
        # ax1.plot(x, blends)
        # for mean_blends in blend_results:
        #     ax1_x = range(1, len(mean_blends)+1)
        #     ax1.plot(ax1_x, mean_blends)
        ax1_x = range(1, len(blend_results)+1)
        ax1.errorbar(mismatch_penalties, blend_results, yerr=blend_sd_results, fmt='--o', label='mean diff. sigma:' + repr(sigma))
        ax1.legend()
        ax1.set_title("mean diff. blend vs ground truth.")

        for n in range(len(salience_mean_results)):#mean_set in salience_mean_results:
            mean_set = salience_mean_results[n]
            sd_set = salience_sd_results[n]
            ax2_x = observation_slots#range(1,len(mean_set)+1)
            MP = mismatch_penalties[n]
            # ax2.scatter(ax2_x, mean_set, label='MP'+repr(i+1))#s=[sd * 1000 for sd in sd_set],
            ax2.errorbar(ax2_x, mean_set, yerr=sd_set ,label='sigma: ' + repr(sigma) + ' MP: '+repr(MP),fmt='--o')
        ax2.legend()
        ax2.set_title("mean salience")

    plt.show()


