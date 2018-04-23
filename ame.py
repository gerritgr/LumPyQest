#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
This script generates code which implements AME lumping for a given model.
The generated script (placed in the ./output directory by default) runs independently of the toolset.
Additional output is written into LumpingLog.log (see utilities.py for logging options).

Caution:
The code uses eval/exec, please use with sanitized input only.
Existing files are overwritten without warning.

Example usage and arguments:
python ame.py model/SIR.yml

See the README.md for more optinos.
"""

__author__ = "Gerrit Grossmann"
__copyright__ = "Copyright 2018, Gerrit Grossmann, Group of Modeling and Simulation at Saarland University"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__email__ = "gerrit.grossmann@uni-saarland.de"

#------------------------------------------------------
# Code Starts Here
#------------------------------------------------------

import sys
sys.dont_write_bytecode = True
import scipy
import matplotlib
matplotlib.use('agg')	#run without an X-server
import numpy as np
import sympy
import time
import os
import pickle


from utilities import *
import model_parser 
import cluster_engine
import expr_generator
import visualization


#------------------------------------------------------
# Generate ODE expressions
#------------------------------------------------------

def coefficient_to_sum(coefficient_store):
	beta_sum = '0.0'
	for key, value in coefficient_store.items():
		beta_sum += '+x[{key}]*{value}'.format(key=key, value=value)
	return beta_sum

def beta_sums(model, elem_to_cluster, cluster_to_elems, degree_and_cluster_to_weight, state, s_minus, s_plus, cluster):
	s_minus_index = model['states'].index(s_minus)
	s_plus_index = model['states'].index(s_plus)
	coefficient_store = dict()
	for elem in cluster_to_elems[cluster]:
		elem_new = list(elem)
		if elem_new[s_minus_index] == 0:
			continue
		elem_new[s_minus_index] -= 1
		elem_new[s_plus_index] += 1
		elem_new = tuple(elem_new)
		coefficient = (state, elem_to_cluster[elem_new])
		if coefficient not in coefficient_store:
			coefficient_store[coefficient] = 0.0
		elem_count = elem_new[s_plus_index] #notation is a mess here
		degree = np.sum(elem)
		weight = degree_and_cluster_to_weight[(degree, cluster)]
		coefficient_store[coefficient] += elem_count*weight
	beta_sum = coefficient_to_sum(coefficient_store)
	#debug:
	#if "20," in str(cluster) and state == 'S' and s_minus == 'I' and s_plus == 'R':
	#	print(cluster)
	#	print(beta_sum)
	return beta_sum


def compute_neighbor_sums(model, elem_to_cluster, cluster_to_elems, degree_and_cluster_to_weight):
	states = model['states']
	sum_store = dict()
	for state in states:
		for s_minus in states:
			for s_plus in states:
				if s_minus == s_plus:
					continue
				for cluster in cluster_to_elems:
					s = beta_sums(model, elem_to_cluster, cluster_to_elems, degree_and_cluster_to_weight, state, s_minus, s_plus, cluster)
					sum_store[(state,s_minus,s_plus,cluster)] = s
	return sum_store

def get_rate_sum(model, cluster, s1, s2, elem):
	rate_sum = 0.0
	for rule in model['rules']:
		consume = rule[0]
		product = rule[1]
		if consume == s1 and product == s2:
			rate_sum += rule[3](elem)	
	return rate_sum

def gen_beta_weight_for_states(model, cluster, s, s1, s2, cluster_to_elems, degree_and_cluster_to_weight):
	m_sum = 0.0
	for elem in cluster_to_elems[cluster]:
		w = degree_and_cluster_to_weight[(np.sum(elem), cluster)]
		s_i = model['states'].index(s)
		m_count = elem[s_i]
		rate_sum = get_rate_sum(model, cluster, s1, s2, elem)
		m_sum += (w*m_count*rate_sum)
	return m_sum

def gen_beta_weight(model, cluster_to_elems, degree_and_cluster_to_weight):
	states = model['states']
	model['beta_weight'] = dict()
	for cluster in cluster_to_elems:
		for s in states:
			for s1 in states:
				for s2 in states:
					if s1 == s2:
						continue
					model['beta_weight'][(cluster, s, s1, s2)] = gen_beta_weight_for_states(model, cluster, s, s1, s2, cluster_to_elems, degree_and_cluster_to_weight)
					

#------------------------------------------------------
# Write Data
#------------------------------------------------------

def load_pickle(model):
	m2 = pickle.load(open(model['pickle_path'], 'rb'))
	for key, value in m2.items():
		model[key] = value
	return model

def write_pickle(model):
	time.sleep(0.1)
	pickle.dump(model, open(model['pickle_path'], "wb"))

def set_modelpaths(model):
	# defien where everything is put
	import os
	model['output_dir'] = './output/{}/'.format(model['name'])
	if 'name_extension' not in model:
		model['output_path'] = model['output_dir']+'ame_{}_{}.py'.format(model['name'], model['actual_cluster_number'])
	else:
		model['output_path'] = model['output_dir']+'ame_{}_{}_{}.py'.format(model['name'], model['actual_cluster_number'], model['name_extension'])
	model['output_path'] = os.path.abspath(model['output_path'])
	model['output_dir'] = os.path.dirname(model['output_path'])+'/'
	model['output_name'] = os.path.basename(model['output_path'])
	model['fullname'] = os.path.basename(model['output_path']).replace('.py', '')
	if not os.path.exists(model['output_dir']):
		os.makedirs(model['output_dir'])
	return model

def write_data(odes, beta_exprs, model, cluster_to_centroid):
	#prepare model to output stand-alone script
	ode_str = ''
	for line in beta_exprs:
		ode_str += '\t' + str(line) + '\n'
	sorted_keys = sorted(list(odes.keys()))
	for key in sorted_keys:
		ode = odes[key]
		ode_str += '\t{ode}\n'.format(ode=ode)
	model['ode_text'] = ode_str
	model['sorted_keys'] = sorted_keys
	init = list()
	cluster_init = model['cluster_init']

	for key in sorted_keys:
		init.append(cluster_init[key])
	model['initial_distribution'] = init

	centroid_vectors=list()
	for key in sorted_keys:
		centroid = cluster_to_centroid[key[1]]
		centroid_vectors.append(centroid)
	model['centroid_vectors'] = centroid_vectors
	# actually write data
	return genrate_file_ame(model)


def write_clustering(model, elem_to_cluster, degree_and_cluster_to_weight):
	outpath = model['output_path'].replace('ame_', 'clustering_').replace('.py', '.csv')
	with open(outpath, 'w') as f:
		f.write('sep=;\nm;cluster;weight\n')
		for elem, cluster in elem_to_cluster.items():
			f.write('{};{};{}\n'.format(elem,cluster,degree_and_cluster_to_weight[(np.sum(elem), cluster)]))


#------------------------------------------------------
# Initial Distribution
#------------------------------------------------------

def gen_initial_distribution(model):
	# compute the multinomial initial distribution for each variable
	state_and_m_to_init = dict()
	init_dist_vector = [model['initial_distribution'][state] for state in model['states']]
	for state in model['states']:
		state_scale = model['initial_distribution'][state]
		for elem, cluster in model['elem_to_cluster'].items():
			identifier = (state, elem)
			if identifier not in state_and_m_to_init: state_and_m_to_init[identifier] = 0.0
			elem_scale = multinomial_pmf(elem, init_dist_vector)
			try:
				assert(elem_scale > 0.0)
			except:
				print(elem)
				print(init_dist_vector)
				raise
			degree = np.sum(elem)
			state_and_m_to_init[identifier] += state_scale*elem_scale
	
	sum_for_degree = dict()
	for (state, elem), prob_mass in state_and_m_to_init.items():
		degree = np.sum(elem)
		if degree not in sum_for_degree: sum_for_degree[degree] = 0.0
		sum_for_degree[degree] += prob_mass
	
	state_and_m_to_init_normalized = dict()
	for (state, elem), prob_mass in state_and_m_to_init.items():
		degree = np.sum(elem)
		degree_prob = model['network']['degree_distribution'][degree]
		degree_sum = sum_for_degree[degree]
		state_and_m_to_init_normalized[(state, elem)] = prob_mass/degree_sum * degree_prob
	
	return state_and_m_to_init_normalized

def lump_initial_distribution(model, state_and_m_to_init, cluster_to_elems):
	# lumping the initial distribution is just adding up the init vlaues for all variables inside a cluster
	state_cluster_to_init = dict()
	for state in model['states']:
		for cluster, elems in cluster_to_elems.items():
			identifier = (state, cluster)
			if identifier not in state_cluster_to_init:
				state_cluster_to_init[identifier] = 0.0
			for elem in elems:
				state_cluster_to_init[identifier] += state_and_m_to_init[(state, elem)]
	return state_cluster_to_init

#------------------------------------------------------
# Stropping Heuristic
#------------------------------------------------------

def stopping_heuristic(modelpath, stop_threshold=0.01, multiply_step=1.3, cluster_start=10):
	import pandas as pd
	errors = list()
	cluster_numbers = list()
	old_model = None
	outpath = None
	max_cluster_num = None
	current_cluster_number = cluster_start
	while True:
		model = model_parser.read_model(modelpath)
		model['lumping']['degree_cluster'] = current_cluster_number
		model['lumping']['proportionality_cluster'] = current_cluster_number
		logger.info('Test with '+repr(model['lumping']))
		model = model_parser.clean_model(model)
		generate_and_solve(model, True, False)
		if outpath is None:
			outpath = model['output_path'].replace('ame_', 'heuristic_').replace('.py', '.csv')
		if old_model is not None:
			error = compare_models(model, old_model)
			logger.info('Difference is: '+str(error))
			errors.append(error)
			cluster_numbers.append(old_model['actual_cluster_number'])
			with open(outpath, 'w') as f:
				df = pd.DataFrame({'cluster_num': cluster_numbers, 'errors': errors})
				df.to_csv(outpath)
			if error < stop_threshold:
				logger.info('stop with '+str(model['actual_cluster_number']))
				break
			if old_model['actual_cluster_number'] == model['actual_cluster_number']:
				logger.info('break')
				break
		old_model = model
		current_cluster_number = max(current_cluster_number*multiply_step,current_cluster_number+1) #to make sure we add at least one cluster
		current_cluster_number = int(current_cluster_number+.5)

#------------------------------------------------------
# Solve ODE
#------------------------------------------------------

def solve_ode(model):
	from time import sleep
	logger.info('Start ODE solver.')
	folderpath = model['output_dir']
	filename = model['output_name']
	sys.path.append(folderpath)
	# import generated script
	exec('import {} as odecode'.format(filename[:-3]), globals())
	results, t, time_elapsed = odecode.plot()
	model['trajectories'] = results
	model['time'] = t
	model['time_elapsed'] = time_elapsed
	logger.info('ODE solver done.')

#------------------------------------------------------
# Main
#------------------------------------------------------


def generate_and_solve(model, autorun, unbinned):
	if unbinned:
		# overwrite model spec
		model['lumping']['lumping_on'] = False

	# load model if already solved
	modelid = '{}_{}_{}_{}'.format(model['name'], model['lumping']['degree_cluster'],model['lumping']['proportionality_cluster'], unbinned)
	pickle_path = './output/{}/model_{}.p'.format(model['name'], modelid)
	model['pickle_path'] = pickle_path
	try:
		model = load_pickle(model)
		print('Found pickled model.')
		return model 
	except:
		pass
	
	# otherwise, solve model

	# clustering should be more consistent
	logger.info('Start Clustering.')
	elem_to_cluster, cluster_to_elems, centroids = cluster_engine.generate_clusters(model, unbinned)
	model['elem_to_cluster'] = elem_to_cluster
	degree_and_cluster_to_weight = cluster_engine.compute_cluster_weights(elem_to_cluster, model['network']['degree_distribution'])
	elem_init = gen_initial_distribution(model)
	model['cluster_init'] = lump_initial_distribution(model, elem_init, cluster_to_elems)
	cluster_to_centroid = cluster_engine.find_centroids(elem_to_cluster, degree_and_cluster_to_weight)
	cluster_rate = cluster_engine.find_avg_rates(elem_to_cluster, degree_and_cluster_to_weight, model['rules'])
	model['actual_cluster_number'] = len(cluster_to_centroid) 

	# needs cluster number to define output filepath
	set_modelpaths(model)

	# write plots
	visualization.plot_clustering_2d(model, cluster_to_elems)
	visualization.plot_cluster_init(model, cluster_to_centroid)
	visualization.plot_cluster_init_full(model, cluster_to_centroid, cluster_to_elems, degree_and_cluster_to_weight)
	write_clustering(model, elem_to_cluster, degree_and_cluster_to_weight)

	# compute beta weights
	beta_sum_store = compute_neighbor_sums(model, elem_to_cluster, cluster_to_elems, degree_and_cluster_to_weight)

	# generate ODEs
	logger.info('Generate ODEs.')
	odes, beta_exprs = expr_generator.generate_odes(model, cluster_to_centroid, beta_sum_store, cluster_to_elems, cluster_rate)
	logger.info('Generate ODEs finished.')

	# precompute values inside beta-computation
	gen_beta_weight(model, cluster_to_elems, degree_and_cluster_to_weight)

	outpath = write_data(odes, beta_exprs, model, cluster_to_centroid)
	logger.info('Filepath:\t'+outpath)
	if autorun:
		sol = solve_ode(model)
		model['sol'] = sol
	logger.info('Done.')

	# save model
	write_pickle(model)
	return model

def main(modelpath, autorun, unbinned):
	model = model_parser.parse_file(modelpath)
	return generate_and_solve(model, autorun, unbinned)

if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('model',  help="path to modelfile")
	parser.add_argument('--noautorun', action='store_true', help="generate code without executing it")
	parser.add_argument('--nolumping', action='store_true', help="generate original equations without lumping")
	parser.add_argument('--autolumping', action='store_true', help="generate original equations without lumping")
	args = parser.parse_args()
	if args.autolumping:
		assert(not args.noautorun and not args.nolumping)
		# use heusistic
		stopping_heuristic(args.model)
	else:
		# use predefined cluster number from model file
		main(args.model, not args.noautorun, args.nolumping)
