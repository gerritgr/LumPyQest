import os
import numpy as np
import yaml
from functools import partial

def eval_rate(m, states, rate_func):
    for i, state in enumerate(states):
        exec('{}={}'.format(state, m[i]))
    rate = eval(rate_func)
    return rate

def set_paths(model):
    model['output_dir'] = './output/{}/'.format(model['name'])
    model['output_dir'] = os.path.abspath(model['output_dir'])
    if not os.path.exists(model['output_dir']):
        os.makedirs(model['output_dir'])

def parse_network(network):
    assert('kmax' in network)
    assert('degree_distribution' in network)
    assert(network['kmax'] > 0)
    network['degree_distribution_text'] = network['degree_distribution']
    degree_eval = eval('lambda k: '+network['degree_distribution'])
    degree_probabilities = np.zeros(network['kmax']+1)
    for i, _ in enumerate(degree_probabilities):
        degree_probabilities[i] = degree_eval(i)
        assert(degree_probabilities[i] >= 0)
    degree_probabilities_sum = np.sum(degree_probabilities)
    assert(degree_probabilities_sum > 0)
    degree_probabilities = degree_probabilities/degree_probabilities_sum
    network['degree_distribution'] = degree_probabilities
    return network

def parse_lumping(lumping):
    assert('degree_cluster'in lumping)
    assert('proportionality_cluster' in lumping)
    lumping['degree_cluster'] = int(lumping['degree_cluster'])
    lumping['proportionality_cluster'] = int(lumping['proportionality_cluster'])
    if 'lumping_on' in lumping:
        lumping['lumping_on'] = bool(lumping['lumping_on'])
    else:
        lumping['lumping_on'] = True
    assert(lumping['degree_cluster'] > 0)
    assert(lumping['proportionality_cluster'] > 0)
    return lumping

def parse_rules(rules, states):
    i_rules = list()
    c_rules = list()
    if len(rules) == 0:
        raise ValueError('No rules specified')
    
    rules_new = list()

    for rule in rules:
        key = list(rule.keys())[0]
        rule_consume = key.split('->')[0].strip()
        rule_produce = key.split('->')[1].strip()
        assert(rule_consume != rule_produce)
        assert(rule_consume in states)
        assert(rule_produce in states)
        rule_rate = str(list(rule.values())[0])
        rate_func = partial(eval_rate, states=tuple(states), rate_func=rule_rate)
        rules_new.append((rule_consume, rule_produce, rule_rate, rate_func))
    return rules_new

def clean_model(model):
    # values which must be specified
    essentials = ['horizon', 'rule', 'initial_distribution', 'network']
    for key in essentials:
        if key not in model:
            raise ValueError('Model is missing {}.'.format(key))
    
    if 'eval_points' not in model:
        model['eval_points'] = 101

    # horizon
    model['horizon'] = float(model['horizon'])
    if model['horizon'] <= 0.0:
        raise ValueError('Horizon must be positive.')
    
    # initial distribution
    model['states'] = model['initial_distribution'].keys()
    if len(model['states']) == 0:
        raise ValueError('No states in initial_distribution')
    model['states'] = sorted(list(model['states']))
    
    for s in model['states']:
        if not s.isalpha():
            raise ValueError('State name can only contain alphabetical characters.')
        model['initial_distribution'][s] = float(model['initial_distribution'][s])
        if model['initial_distribution'][s] <= 0.0:
            raise ValueError('Initial distribution must be positive.')        
    
    init_sum = np.sum(list(model['initial_distribution'].values()))
    model['initial_distribution'] = {state:value/init_sum for (state,value) in model['initial_distribution'].items()}

    # rules
    rules  = parse_rules(model['rule'], model['states'])
    model['rules'] = rules
    del model['rule']

    model['lumping'] = parse_lumping(model['lumping'])

    # network
    model['network'] = parse_network(model['network'])

    set_paths(model)

    return model
        

def read_model(filepath):
    if not filepath.endswith('.yml'):
        raise ValueError('Model specification must be .yml file.')
    with open(filepath, 'r') as s:
        try:
            model = yaml.load(s)
        except yaml.YAMLError as exc:
            print(exc)
    model['name'] = filepath.replace('\\',"/").split('/')[-1].replace('.yml','')
    model['modelpath'] = os.path.abspath(filepath)
    return model

def parse_file(filepath):
    m = read_model(filepath)
    return clean_model(m)


if __name__ == '__main__':
    x = parse_file('model/SIR.yml')
    print(x['rules'])
    for rule in x['rules']:
        m = (3,2,3)
        print(rule[3](m))