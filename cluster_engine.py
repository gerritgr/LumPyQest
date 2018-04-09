import utilities

import numpy as np


def gen_degree_clusters(cluster_num, degree_distribution):
	def surrogate_loss(cluster):
		degree_list, prob_list = cluster
		prob_mass = np.sum(prob_list)
		loss = prob_mass**2
		assert(not np.isnan(loss))
		return loss
	
	scale = 0.6

	clustering = [([i],[degree_distribution[i]**scale]) for i in range(len(degree_distribution))]

	while len(clustering) > cluster_num:
		best_costs = 1000**2
		best_i = -1
		best_joint = None
		for i in range(len(clustering)-1):
			c1 = clustering[i]
			c2 = clustering[i+1]
			costs_c1 = surrogate_loss(c1)
			costs_c2 = surrogate_loss(c2)
			cjoin = (c1[0]+c2[0],c1[1]+c2[1])
			costs_cjoin = surrogate_loss(cjoin)
			join_costs = costs_cjoin - (costs_c1 + costs_c2)
			assert(costs_cjoin >= (costs_c1 + costs_c2))
			if join_costs<best_costs:
				best_costs=join_costs
				best_i = i
				best_joint = cjoin
		clustering[best_i] = best_joint
		del clustering[best_i+1]
	indicators = list()

	for i, cluster in enumerate(clustering):
		indicators += [i]*len(cluster[0])

	return indicators

def get_proportionality_cluster(point, intervals):
    point = list(point)

    for i, _ in enumerate(point):
        point[i] += 0.0000001 * i # for trie breaks
    ps = np.sum(point)
    if ps > 0:
        point = [v/float(ps) for v in point]

    #point = [p+0.001*(1/len(point) + p) for p in point]  # for trie breaks
    ps = np.sum(point)
    if ps > 0:
        point = [v/float(ps) for v in point]


    k = intervals
    arange = np.linspace(0+.5*1.0/k,1.0-.5*1.0/k,k)

    new_point = list()
    best_iv = list()
    for pv in point:
    #for pv in point[:-1]:
        best_i = -1
        best_dist = 1000000
        for i, interv in enumerate(arange):
            if np.abs(interv - pv) <= best_dist:
                best_dist = np.abs(interv - pv)
                best_i = i
        new_point.append(best_i)
        best_iv.append(arange[best_i])
    best_sum = np.sum(best_iv)

    cutoff = 1000
    best_iv = [int(v/best_sum*cutoff)/cutoff for v in best_iv]
    new_point = [int(v*cutoff)/cutoff for v in new_point]
    return tuple(new_point)

def get_hexagon_cluster(point, intervals):
    if intervals > 100:
        return tuple(point)
    from scipy.spatial.distance import cosine, euclidean
    def dist(p1, p2):
        p1 = list(p1)
        p2 = list(p2)
        #return euclidean([p/(np.sum(p1)+0.0001) for p in p1], [p/(np.sum(p2)+0.0001) for p in p2])
        return cosine(p1, p2)

    centroid_candidates = utilities.gen_mset_for_degree(intervals, len(point))

    min_dist = 100000
    best_centroid = None

    for centroid in centroid_candidates:
        if best_centroid is None or dist(centroid, point) < min_dist:
            min_dist = dist(centroid, point)
            best_centroid = centroid

    return tuple(best_centroid)


def generate_clusters_box(model, unbinned):
    degree_cluster_num = model['lumping']['degree_cluster']
    degree_distribution = model['network']['degree_distribution']
    if unbinned:
        degree_cluster_num = 100000
    degree_clusters = gen_degree_clusters(degree_cluster_num, degree_distribution)
    elem_to_cluster = dict()
    cluster_to_elems = dict()
    centroids = set()
    mset = utilities.gen_mset(model['network']['kmax'], len(model['states']))

    for point in mset:
        centroid = list()
        for i, v in enumerate(point):
            centroid.append(degree_clusters[v])
        centroid = tuple(centroid)
        degree_cluster = centroid[0] #degree_clusters[np.sum(point)]
        degree_cluster = 0
        cluster = str((degree_cluster,centroid))
        elem_to_cluster[point] = cluster
        if cluster not in cluster_to_elems: cluster_to_elems[cluster] = list()
        cluster_to_elems[cluster].append(point)

    return elem_to_cluster, cluster_to_elems, centroids


def get_box_cluster(point, proportionality_cluster_num, degree_clusters):
    centroid = list()
    for i, v in enumerate(point):
        centroid.append(degree_clusters[v])
    return tuple(centroid)



#def generate_clusters(degree_cluster_num, proportionality_cluster_num, degree_distribution, dim, kmax):
def generate_clusters(model, unbinned):
    degree_cluster_num = model['lumping']['degree_cluster']
    proportionality_cluster_num = model['lumping']['proportionality_cluster']
    degree_distribution = model['network']['degree_distribution']

    if unbinned:
        degree_cluster_num = model['network']['kmax'] + 1 #+1 for zero degree
        proportionality_cluster_num = 1000 # TODO do better
    degree_clusters = gen_degree_clusters(degree_cluster_num, degree_distribution)

    elem_to_cluster = dict()
    cluster_to_elems = dict()
    centroids = set()

    mset = utilities.gen_mset(model['network']['kmax'], len(model['states']))

    for point in mset:
        centroid =  get_proportionality_cluster(point, proportionality_cluster_num)
        degree_cluster = degree_clusters[np.sum(point)] 
        cluster = str((degree_cluster,centroid))

        elem_to_cluster[point] = cluster
        if cluster not in cluster_to_elems: cluster_to_elems[cluster] = list()
        cluster_to_elems[cluster].append(point)
        centroids.add(centroid)

    return elem_to_cluster, cluster_to_elems, centroids

def compute_cluster_weights(elem_to_cluster, degree_distribution):
    cluster_to_normalization = dict()
    degree_and_cluster_to_weight = dict()

    for elem, cluster in elem_to_cluster.items():
        if cluster not in cluster_to_normalization: cluster_to_normalization[cluster] = 0.0
        degree = np.sum(elem)
        dim = len(elem)
        pkm = degree_distribution[degree]
        elems_Mkm = utilities.elemcount_mset_for_degree(degree, dim)
        normalsum = pkm/elems_Mkm
        cluster_to_normalization[cluster] += normalsum

    for elem, cluster in elem_to_cluster.items():
        degree = np.sum(elem)
        dim = len(elem)
        identifer = (degree, cluster)
        Z = cluster_to_normalization[cluster]
        pkm = degree_distribution[degree]
        elems_Mkm = utilities.elemcount_mset_for_degree(degree, dim)
        degree_and_cluster_to_weight[identifer] = (pkm/elems_Mkm)/Z
    
    return degree_and_cluster_to_weight


def compute_cluster_weights_uniform(elem_to_cluster, degree_distribution):
    elems_in_cluster = dict()
    degree_and_cluster_to_weight = dict()

    for elem, cluster in elem_to_cluster.items():
        if cluster not in elems_in_cluster: elems_in_cluster[cluster] = 0.0
        elems_in_cluster[cluster] += 1

    for elem, cluster in elem_to_cluster.items():
        degree = np.sum(elem)
        identifer = (degree, cluster)
        degree_and_cluster_to_weight[identifer] = 1.0/elems_in_cluster[cluster]
    
    return degree_and_cluster_to_weight


def find_centroids(elem_to_cluster, degree_and_cluster_to_weight):
    cluster_to_centroid = dict()
    for elem, cluster in elem_to_cluster.items():
        if cluster not in cluster_to_centroid: cluster_to_centroid[cluster] = [0.0]*len(elem)
        degree = np.sum(elem)
        weight = degree_and_cluster_to_weight[(degree, cluster)]
        vec = cluster_to_centroid[cluster]
        for i, _ in enumerate(vec):
            vec[i] += weight * elem[i]
        cluster_to_centroid[cluster] = vec
    return cluster_to_centroid

def find_avg_rates(elem_to_cluster, degree_and_cluster_to_weight, rules):
    rule_and_cluster_to_rate = dict()
    for elem, cluster in elem_to_cluster.items():
        degree = np.sum(elem)
        weight = degree_and_cluster_to_weight[(degree, cluster)]
        for rule_i, rule in enumerate(rules):
            identifier = (rule_i, cluster)
            if identifier not in rule_and_cluster_to_rate:
                rule_and_cluster_to_rate[identifier] = 0.0
            rate = rule[3](elem) #apply rate function
            rule_and_cluster_to_rate[identifier] += rate * weight
    return rule_and_cluster_to_rate


def approximate_avg_rates_centroid(centroid, rules, min_k, max_k, degree_distribution):
    degree_to_weight = dict()
    dim = len(centroid)
    for k in np.arange(min_k, max_k+1):
        prob = degree_distribution[k]
        #elem_num = utilities.elemcount_mset_for_degree(k, dim)
        #weight = prob/elem_num
        weight = prob # we do not need elem_num here because we anyway only evaluate one elem per degree
        degree_to_weight[k] = weight
    Z = np.sum(list(degree_to_weight.values()))
    degree_to_weight = {k:v/Z for k,v in degree_to_weight.items()}

    diagonal_points = dict()
    for k in np.arange(min_k, max_k+1):
        z = np.sum(centroid)
        centroid_scale = [(p/z*k+0.0) for p in centroid]
        diagonal_points[k] = centroid_scale

    rule_to_rate = dict()
    for rule_i, rule in enumerate(rules):
        rule_rate = 0.0
        for k, diag_point in diagonal_points.items():
            rate = rule[3](diag_point)
            w = degree_to_weight[k]
            rule_rate += w*rate
        rule_to_rate[rule_i] = rule_rate

    cluster_centroid = [0.0]*dim
    for k, diag_point in diagonal_points.items():
        w = degree_to_weight[k]
        for i, p in enumerate(diag_point):
            cluster_centroid[i] += w*p

    
    return rule_to_rate, cluster_centroid


def get_cluster_size(degree, centroids, centroids_2d, dim):
    numer_points = utilities.elemcount_mset_for_degree(degree, dim)
    numer_points_per_cluster = numer_points/(centroids)
    numer_points_2d = utilities.elemcount_mset_for_degree(degree, dim-1)
    numer_points_per_cluster_2d = numer_points_2d/centroids_2d
    return numer_points_per_cluster, numer_points_per_cluster_2d


def compute_neighbors(point, cluster_neighbors, states, proportionality_cluster_num):
    for s1 in states:
        for s2 in states:
            if s1 == s2:
                continue
            s1_i = states.index(s1)
            s2_i = states.index(s2)
            if point[s2_i] == 0:
                continue
            new_point = list(point)
            new_point[s2_i] -= 1
            new_point[s1_i] += 1
            old_cluster = get_proportionality_cluster(point, proportionality_cluster_num)
            new_cluster = get_proportionality_cluster(new_point, proportionality_cluster_num)
            if (old_cluster, s1, s2) not in cluster_neighbors:
                cluster_neighbors[(old_cluster, s1, s2)] = dict()
            old_neighbors = cluster_neighbors[(old_cluster, s1, s2)]
            if new_cluster not in old_neighbors:
                old_neighbors[new_cluster]  = 0
            old_neighbors[new_cluster] += 1 


def clean_cluster_neighbors(cluster_neighbors): # we only have one (valid) neighbor for each change in states + artefacts
    #print('cluster_neighbors', cluster_neighbors.items())

    for cluster, neighbor_dict in cluster_neighbors.items():
        neighbor_dict = {k: v for k, v in neighbor_dict.items() if v > 3} #? to get rid of artefacts
        keys_keep = sorted(list(neighbor_dict.items()), key = lambda x: -x[1])
        for i, item in enumerate(keys_keep):
            if i>1:
                del neighbor_dict[item[0]]
        cluster_neighbors[cluster] = neighbor_dict
    
    neighbors_new = dict()
    for cluster, neighbor_dict in cluster_neighbors.items():
        cluster_id, state_minus, state_plus = cluster
        if cluster_id not in neighbors_new:
            neighbors_new[cluster_id] = dict()
        dir_dict = neighbors_new[cluster_id]
        dir_dict[(state_minus, state_plus)] = list()
        for key in neighbor_dict.keys():
            dir_dict[(state_minus, state_plus)].append(key)
    return neighbors_new

def compute_avg_beta_weights(cluster_dict, states, cluster_info):
    neighbors = cluster_dict['neighbors']
    result_dict = dict()

    in_factor = (cluster_dict['point_num']-cluster_dict['point_border_num'])/cluster_dict['point_num']
    out_factor = cluster_dict['point_border_num']/cluster_dict['point_num']
    in_factor = max(0, in_factor)
    out_factor = max(0, out_factor)
    m = cluster_dict['center']
    d = cluster_dict['degree_cluster']

    for s1, s2 in neighbors.keys():
        s1_index = states.index(s1)
        s2_index = states.index(s2)
        m_new = list(m)
        m_new[s1_index] += 1
        m_new[s2_index] -= 1
        m_count = m_new[s1_index]
        n = neighbors[(s1,s2)]
        if len(n) == 1:
            in_cluster = n[0] 
            kd = cluster_dict['degree_cluster']
            in_cluster = str((kd,in_cluster))
            m_count = max(0,m_count)
            result_dict[(s1,s2)] = "{}*x[('_', '{}')]".format(m_count*in_factor, in_cluster)
        elif len(n) == 2:
            in_cluster = n[0]
            out_cluster = n[1]
            if in_cluster != cluster_dict['proportionality_cluster']:
                in_cluster, out_cluster = out_cluster, in_cluster

            kd = cluster_dict['degree_cluster']
            in_cluster = str((kd,in_cluster))
            out_cluster = str((kd,out_cluster))
            center_out = cluster_info[out_cluster]['center']
            m_count_out = (m_new[s1_index]+center_out[s1_index])/2.0
            m_count_out = max(0,m_count_out)
            result_dict[(s1,s2)] = "{}*x[('_', '{}')]+{}*x[('_', '{}')]".format(m_count*in_factor, in_cluster, m_count*out_factor, out_cluster)
    return result_dict


def compute_avg_clustering(model, unbinned):
    # num_elems
    # direction
    # centroid
    # kmin
    # kmax
    # mumelems
    # elems per degree
    # centroid per degree (diagonal)
    # weight per degree
    # (s',s'') -> neighbor
    if unbinned:
        degree_cluster_num = model['network']['kmax'] + 1 #+1 for zero degree
        proportionality_cluster_num = 1000 # TODO do better

    cluster_info = dict()
    centroid_set = set()
    centroid_set_projection = set()

    degree_cluster_num = model['lumping']['degree_cluster']
    proportionality_cluster_num = model['lumping']['proportionality_cluster']
    degree_distribution = model['network']['degree_distribution']
    degree_clusters = gen_degree_clusters(degree_cluster_num, degree_distribution)
    centroid_to_points = dict()
    centroid_to_direction = dict()
    cluster_neighbors = dict()
    effective_centroids = dict()

    BASELINE_DEGREE = 30

    for point in utilities.gen_mset(BASELINE_DEGREE, len(model['states'])):   #30 arbitrary, should probably depend on clusternum
        centroid =  get_proportionality_cluster(point, proportionality_cluster_num)
        centroid_set.add(centroid)
        if centroid not in centroid_to_points: centroid_to_points[centroid] = list()
        centroid_to_points[centroid].append(point)
        centroid_set_projection.add(get_proportionality_cluster(point[:-1], proportionality_cluster_num))
        if np.sum(point) == BASELINE_DEGREE:
            compute_neighbors(point, cluster_neighbors, model['states'], proportionality_cluster_num)
            effective_centroids[centroid] = effective_centroids.get(centroid,0)+1
    
    cluster_neighbors = clean_cluster_neighbors(cluster_neighbors)

    for centroid, elems in centroid_to_points.items():
        direction = [0.0]*len(centroid)
        n = len(elems)
        for elem in elems:
            for i in range(len(elem)):
                direction[i] += elem[i]/n
        centroid_to_direction[centroid] = list(utilities.create_normalized_np(direction, True))

    for centroid in centroid_set:
        for k, degree_cluster in enumerate(degree_clusters):
            cluster = str((degree_cluster,centroid))
            if cluster not in cluster_info:
                cluster_info[cluster] = {'k_min': 10**10, 'k_max':-1, 'elems': list()}
            cluster_dict = cluster_info[cluster]
            cluster_dict['proportionality_cluster'] = centroid
            cluster_dict['direction'] = centroid_to_direction[centroid] 
            cluster_dict['degree_cluster'] = degree_cluster
            cluster_dict['k_min'] = min(k,cluster_dict['k_min'])
            cluster_dict['k_max'] = max(k,cluster_dict['k_max'])
    
    for cluster, cluster_dict in cluster_info.items():
        rule_to_rate, cluster_centroid = approximate_avg_rates_centroid(cluster_dict['direction'], model['rules'], cluster_dict['k_min'], cluster_dict['k_max'], degree_distribution)
        cluster_dict['center'] = cluster_centroid
        cluster_dict['rule_to_rate'] = rule_to_rate

    for cluster, cluster_dict in cluster_info.items():
        try:
            cluster_dict['neighbors'] = cluster_neighbors[tuple(cluster_dict['proportionality_cluster'])]
        except:
            cluster_dict['neighbors'] = dict() 

    for cluster, cluster_dict in cluster_info.items():
        degree = int(np.sum(cluster_dict['center'])+0.5)
        point, points_border = get_cluster_size(degree, len(effective_centroids), len(centroid_set_projection), len(model['states']))
        cluster_dict['point_num'] = point
        cluster_dict['point_border_num'] = points_border

    for cluster, cluster_dict in cluster_info.items():
        cluster_dict['beta_weights'] = compute_avg_beta_weights(cluster_dict, model['states'], cluster_info)
    return cluster_info

def compute_exact_beta_weights(cluster_dict, states, degree_and_cluster_to_weight, point_to_cluster): 
    betas = dict()
    for s1 in states:
        for s2 in states:
            if s1 == s2:
                continue 
            s1_i = states.index(s1)
            s2_i = states.index(s2)
            term_dict = dict()
            for m in cluster_dict['elems']:
                m_new = list(m)
                if m_new[s2_i] == 0:
                    continue
                m_new[s2_i] -= 1
                m_new[s1_i] += 1
                elem_count = m_new[s1_i]
                c = point_to_cluster[tuple(m_new)]
                k = np.sum(m)
                ident = (k, c)
                if ident not in degree_and_cluster_to_weight:
                    return dict()
                w = degree_and_cluster_to_weight[ident]
                if c not in term_dict:
                    term_dict[c] = 0.0
                term_dict[c] += w*elem_count
            betas[(s1,s2)] = term_dict 
    return betas


def compute_exact_clustering(model, degree_and_cluster_to_weight, unbinned):
    if unbinned:
        degree_cluster_num = model['network']['kmax'] + 1 #+1 for zero degree
        proportionality_cluster_num = 1000 # TODO do better
    cluster_info = model['cluster_info']
    degree_cluster_num = model['lumping']['degree_cluster']
    proportionality_cluster_num = model['lumping']['proportionality_cluster']
    degree_distribution = model['network']['degree_distribution']
    degree_clusters = gen_degree_clusters(degree_cluster_num, degree_distribution)
    cluster_to_elems = dict()
    dim = len(model['states'])
    mset = utilities.gen_mset(model['network']['kmax'], dim)
    point_to_cluster = dict()

    print(cluster_info.keys())
    for point in mset:
        centroid =  get_proportionality_cluster(point, proportionality_cluster_num)
        degree_cluster = degree_clusters[np.sum(point)] 
        cluster = str((degree_cluster,centroid))
        cluster_info[cluster]['elems'].append(point)
        point_to_cluster[point] = cluster

    for cluster, cluster_dict in cluster_info.items():
        if 'rule_to_rate_exact' not in cluster_dict:
            cluster_dict['rule_to_rate_exact'] = {rule_i:0.0 for rule_i in range(len(model['rules']))}
        for elem in cluster_dict['elems']:
            degree = np.sum(elem)
            weight = degree_and_cluster_to_weight[(degree, cluster)] if (degree, cluster) in degree_and_cluster_to_weight else 0
            for rule_i, rule in enumerate(model['rules']):
                rate = rule[3](elem) #apply rate function
                cluster_dict['rule_to_rate_exact'][rule_i] += rate*weight
    
    for cluster, cluster_dict in cluster_info.items():
        if 'direction_exact' not in cluster_dict:
            cluster_dict['direction_exact'] = [0.0]*dim
        for elem in cluster_dict['elems']:
            degree = np.sum(elem)
            w = degree_and_cluster_to_weight[(degree, cluster)] if (degree, cluster) in degree_and_cluster_to_weight else 0
            for i in range(len(elem)):
                elem = utilities.create_normalized_np(elem, True)
                cluster_dict['direction_exact'][i] += elem[i] * w

    for cluster, cluster_dict in cluster_info.items():
        cluster_dict['beta_weights_exact'] = compute_exact_beta_weights(cluster_dict, model['states'], degree_and_cluster_to_weight, point_to_cluster)
        #print(cluster_dict['center'])
        #for key,value in cluster_dict['beta_weights_exact'].items():
        #    print (key, value )
        #print(' ')
        #for key,value in cluster_dict['beta_weights'].items():
        #    print(key, value)
        #print('\n')

    for cluster, cluster_dict in cluster_info.items():
        print(cluster)
        for key, value in cluster_dict.items():
            print(key, value) 
        print('')
    

    # kmax_threshold = 100 if 'kmax_threshold' not in model['lumping'] else model['lumping']['kmax_threshold']
    # kmax_threshold_cluster = degree_clusters[kmax_threshold]
    # kmax_threshold = np.max([k for k in range(len(degree_distribution)+1) if degree_clusters[k] ==  kmax_threshold_cluster])
    # kmax_exact = min(model['network']['kmax'], kmax_threshold)
    # mset = utilities.gen_mset(kmax_exact, len(model['states']))




if __name__ == '__main__':
    compute_clustering(None, None)