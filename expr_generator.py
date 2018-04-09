import numpy as np

def generate_odes(model, cluster_to_centroid, beta_sum_store, cluster_to_elems, cluster_rate):
	# generates equations and beta-expressions
	import itertools
	from multiprocessing import Pool, cpu_count

	states = model['states']
	pool = Pool(cpu_count())
	state_combinatinos = itertools.product(states,states,states)
	state_combinatinos = [s_t for s_t in state_combinatinos if s_t[1] != s_t[2]]
	beta_exprs = pool.map(gen_beta, state_combinatinos)
	pool.close()
	pool.join()
	
	rules = model['rules']
	staes = model['states']

	odes = dict()
	for state in model['states']:
		for cluster in cluster_to_elems:
			odes[(state, cluster)] = gen_ame(state, cluster, rules, cluster_to_centroid, beta_sum_store, states, cluster_rate)
	return odes, beta_exprs


def gen_ame(state, cluster, rules, cluster_to_centroid, beta_sum_store, states, cluster_rate):
	c_id = 'x[{}]'.format((state, cluster))
	line = 'dt_x[{}] = 0'.format((state, cluster))

	for rule_i, rule in enumerate(rules):
		consume = rule[0]
		produce = rule[1]
		rate_func = rule[3]
		if produce == state:
			centroid = cluster_to_centroid[cluster]
			rate = cluster_rate[(rule_i, cluster)]
			line += "+({r}*x[('{consume}', '{cluster}')])".format(r=rate, consume=consume, cluster=cluster)

	for rule_i, rule in enumerate(rules):
		consume = rule[0]
		produce = rule[1]
		rate_func = rule[3]
		if consume == state:
			rate = cluster_rate[(rule_i, cluster)]
			line += "-({r}*x[('{consume}', '{cluster}')])".format(r=rate, consume=consume, cluster=cluster)

	for s1 in states:
		for s2 in states:
			if s1 == s2:
				continue
			beta = 'beta_{s}_{s1}_to_{s}_{s2}'.format(s=state, s1=s1,s2=s2)
			beta_sum = beta_sum_store[(state, s2, s1, cluster)] # careful!!!!
			line += "+{beta}*({beta_sum})".format(beta=beta, beta_sum = beta_sum)

	for s1 in states:
		for s2 in states:
			if s1 == s2:
				continue
			beta = 'beta_{s}_{s1}_to_{s}_{s2}'.format(s=state, s1=s1,s2=s2)
			centroid = cluster_to_centroid[cluster]
			centroid_dir = centroid[states.index(s1)]
			line += "-({beta}*x[('{state}', '{cluster}')]*{centroid_dir})".format(beta=beta, state=state, cluster=cluster, centroid_dir=centroid_dir)
	return line



def gen_beta(states_tuple):
	state, s1, s2 = states_tuple
	beta = 'beta_{s}_{s1}_to_{s}_{s2} = compute_beta("{s}","{s1}","{s2}")'
	beta = beta.format(s=state, s1 = s1, s2 = s2)
	return beta

