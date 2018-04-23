import matplotlib
matplotlib.use('agg')	#run without X-server
import matplotlib.pyplot as plt
#import seaborn as sns 
import random
import numpy as np

def plot_clustering_1d(model, cluster_to_elems):
	return
	plt.clf()
	import seaborn as sns
	clusterlist = [-1] * (model['network']['kmax']+1)

	for i, (cluster, elems) in enumerate(cluster_to_elems.items()):
		for elem in elems:
			if np.sum(elem[1:]) == 0:
				clusterlist[elem[0]] = i

	clusters = [clusterlist]
	ax = sns.heatmap(clusters, cbar=False, yticklabels=False, xticklabels=True, annot=True, fmt="d")
	ax.set_aspect(1.7)
	plt.tight_layout()
	plt.title('Degree Clustering')
	plt.tight_layout()
	plt.savefig(model['output_path'].replace('ame_', 'clustering1D_').replace('.py', '.pdf'))
	sns.reset_orig()
	matplotlib.rcParams.update(matplotlib.rcParamsDefault)

def plot_clustering_2d(model, cluster_to_elems):
	matplotlib.rcParams.update(matplotlib.rcParamsDefault)
	plot_clustering_1d(model, cluster_to_elems)
	plot_clustering_3d(model, cluster_to_elems)
	matplotlib.rcParams.update(matplotlib.rcParamsDefault)
	plt.clf()
	#cp = sns.color_palette('Set1', 10)
	#random.shuffle(cp)
	#sns.set_context('paper', font_scale = 2.3, rc={"lines.linewidth": 5, 'xtick.labelsize':20, 'ytick.labelsize':20})
	#sns.set_style('white')
	ax = plt.gca()
	cluster_key = sorted(cluster_to_elems.keys())
	for i, key in enumerate(cluster_key):
		elems = cluster_to_elems[key]
		elems_projection = set()
		for elem in elems:
			if len(elem) > 2 and np.sum(elem[2:]) > 0:
				continue
			elems_projection.add((elem[0], elem[1]))
		elems = list(elems_projection)
		x = [elem[0] for elem in elems]
		y = [elem[1] for elem in elems]
		ax.scatter(x, y, alpha=0.9, linewidths=0.0)
	ax.set_xlim(xmin=-0.3)
	ax.set_ylim(ymin=-0.3)
	ax.set_aspect(1.0)
	ax.set(xlabel='Neighbors in state {}'.format(model['states'][0]), ylabel='Neighbors in state {}'.format(model['states'][1]))
	plt.tight_layout()
	plt.savefig(model['output_path'].replace('ame_', 'clustering2D_').replace('.py', '.pdf'))
	matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def plot_clustering_3d(model, cluster_to_elems):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import axes3d, Axes3D 
	from matplotlib.colors import Normalize
	import matplotlib.cm as cm
	plt.clf()
	fig = plt.figure()
	#ax = fig.add_subplot(111, projection='3d')
	ax = Axes3D(fig)
	cmap = cm.plasma
	norm = Normalize(vmin=0, vmax=1)
	kmax = model['network']['kmax']

	elem_to_cluster_kmax = dict()

	cluster_to_color=dict()

	for i, (cluster, elems) in enumerate(cluster_to_elems.items()):
		for elem in elems:
			if len(elem) != 3:
				return
			if np.sum(elem) != kmax:
				continue
			elem_to_cluster_kmax[elem] = i
		cluster_to_color[i] = random.random()
	
	for (x,y,z), cluster_i in elem_to_cluster_kmax.items():
		ax.scatter([x], [y], [z], c=cmap(cluster_to_color[cluster_i]))

	matplotlib.rcParams['xtick.labelsize'] = 30
	ax.xaxis.labelpad=25
	ax.yaxis.labelpad=25
	ax.zaxis.labelpad=12
	ax.set_xlabel(model['states'][0], fontsize=30)
	ax.set_ylabel(model['states'][1], fontsize=30)
	ax.set_zlabel(model['states'][2], fontsize=30)
	ax.set_ylim([0,kmax])
	ax.set_xlim([0,kmax])
	ax.set_zlim([0,kmax])
	ax.view_init(35, 45)
	#fig.tight_layout(rect=[0, 0.03, 1, 1])

	plt.savefig(model['output_path'].replace('ame_', 'clustering3D_').replace('.py', '.pdf'))
		

		
def plot_cluster_init(model, cluster_to_centroid):
	plt.clf()

	for i in range(len(model['states'])):
		plt.subplot(1, len(model['states']), i+1)
		import matplotlib.cm as cm
		from matplotlib.colors import Normalize
		import numpy as np

		cluster_init = model['cluster_init'] 
		cmap = cm.plasma
		min_init = np.min(list(cluster_init.values()))
		max_init = np.max(list(cluster_init.values()))
		norm = Normalize(vmin=np.log(min_init), vmax=np.log(max_init))

		x_list = list()
		y_list = list()
		c_list = list()
		for (state, cluster), init in cluster_init.items():
			if state != model['states'][i]:
				continue
			centroid = cluster_to_centroid[cluster]
			if len(centroid) > 2 and np.sum(centroid[2:]) > 0:
				continue
			x_list.append(centroid[0])
			y_list.append(centroid[1])
			c_list.append(cmap(norm(np.log(init))))

		ax = plt.gca()
		ax.scatter(x_list, y_list, c=c_list, alpha=0.9, linewidths=0.0)
		ax.set_xlim([-0.5, model['network']['kmax']+0.5])
		ax.set_ylim([-0.5, model['network']['kmax']+0.5])
		ax.set_aspect(1.0)
		ax.set(xlabel='Neighbors in state {}'.format(model['states'][0]), ylabel='Neighbors in state {}'.format(model['states'][1]))
		plt.title('Initial Distribution: '+model['states'][i])

	plt.tight_layout()
	plt.savefig(model['output_path'].replace('ame_', 'initial_').replace('.py', '.pdf'))	


def plot_cluster_init_full(model, cluster_to_centroid, cluster_to_elems, degree_and_cluster_to_weight):

	plt.clf()
	for i in range(len(model['states'])):
		plt.subplot(1, len(model['states']), i+1)
		import matplotlib.cm as cm
		from matplotlib.colors import Normalize
		import numpy as np

		cluster_init = model['cluster_init'] 
		cmap = cm.plasma
		min_init = np.min(list(cluster_init.values()))
		max_init = np.max(list(cluster_init.values()))
		norm = Normalize(vmin=np.log(min_init), vmax=np.log(max_init))

		x_list = list()
		y_list = list()
		c_list = list()
		for (state, cluster), init in cluster_init.items():
			if state != model['states'][i]:
				continue
			elems = cluster_to_elems[cluster]
			for elem in elems:
				if len(elem) > 2 and np.sum(elem[2:]) > 0:
					continue
				x_list.append(elem[0])
				y_list.append(elem[1])
				w = degree_and_cluster_to_weight[(np.sum(elem), cluster)]
				real_init = init*w
				c_list.append(cmap(norm(np.log(real_init))))

		ax = plt.gca()
		ax.scatter(x_list, y_list, c=c_list, alpha=0.9, linewidths=0.0)
		ax.set_xlim([-0.5, model['network']['kmax']+0.5])
		ax.set_ylim([-0.5, model['network']['kmax']+0.5])
		ax.set_aspect(1.0)
		ax.set(xlabel='Neighbors in state {}'.format(model['states'][0]), ylabel='Neighbors in state {}'.format(model['states'][1]))
		plt.title('Initial Distribution: '+model['states'][i])

	plt.tight_layout()
	plt.savefig(model['output_path'].replace('ame_', 'initial_').replace('.py', 'full.pdf'))	


def plot_scatter2(x, y, outpath, color = 'r', x_label = 'Cluster Number per State', y_label = 'Distance to Original Equation', set_lim = True):
	import numpy as np
	import pandas as pd
	import matplotlib
	matplotlib.use('agg')	#run without an X-server
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	mpl.rc('xtick', labelsize=17)
	mpl.rc('ytick', labelsize=17)
	alpha = 0.8
	area = 300.0
	fig, ax1 = plt.subplots()
	ax1.set_xlabel(x_label,fontsize=17)
	if set_lim: 
		ax1.set_xlim([-0.2, max(x)*1.1])
		ax1.set_ylim([-0.2, max(y)*1.1])
	import seaborn as sns
	s1 = ax1.scatter(x, y, s=area, c=color, alpha=alpha, marker = '*')
	ax1.set_ylabel(y_label,fontsize=17)
	if set_lim: ax1.set_ylim([0, max(y)*1.1])
	#sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 5.5})
	sns.set_style("white")
	sns.despine()
	plt.savefig(outpath, format='pdf', bbox_inches='tight')
	df = pd.DataFrame({x_label: x, y_label : y})
	df.to_csv(outpath[:-4]+'.csv', header='sep=,')
	plt.close()
	#sns.reset_orig()


def plot_scatter(x, y, outpath, color = 2, x_label = 'Cluster Number', y_label = 'Error', set_lim = True):
	import numpy as np
	import pandas as pd
	import seaborn as sns
	sns.reset_orig()

	sns.set(context='paper', style='white', palette='muted', font='sans-serif', font_scale=2.7, color_codes=False, rc={"axes.linewidth": 2.0})

	alpha = 0.8
	area = 500.0
	fig, ax1 = plt.subplots()
	if set_lim: 
		ax1.set_xlim([min(0,-min(x)*0.2), max(x)*1.1])
		ax1.set_ylim([min(0,-min(y)*0.2), max(y)*1.1])
	ax1.set_xlabel(x_label,fontsize=20)
	ax1.set_ylabel(y_label,fontsize=20)
	color = sns.color_palette("muted", max(color,8))[color]
	s1 = ax1.scatter(x, y, s=area, c=color, alpha=alpha, marker = '*')
	sns.despine()
	plt.xticks(np.linspace(0,np.max(x),4))
	plt.yticks([round(v,2) for v in np.linspace(0,np.max(y),4)])
	plt.savefig(outpath, format='pdf', bbox_inches='tight')
	df = pd.DataFrame({x_label: x, y_label : y})
	df.to_csv(outpath[:-4]+'.csv', header='sep=,')
	plt.close()
	sns.reset_orig()
	matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def plot_trajectories_individual_line(line, clustering, outpath, state, states, min_sol, max_sol):
	import seaborn as sns
	sns.reset_orig()
	sns.set(context='paper', style='white', palette='muted', font='sans-serif', font_scale=2.0, color_codes=False, rc={"axes.linewidth": 2.0})

	x_list = list()
	y_list = list()
	value_list = list()
	kmax = 0
	
	for c_line in clustering:
		m = eval(c_line['m'])
		x_list.append(m[0])
		y_list.append(m[1])
		kmax = kmax if m[0]+m[1] < kmax else m[0]+m[1] 
		cluster = c_line['cluster']
		cluster_id = "('{}', '{}')".format(state, cluster)
		value = line[cluster_id]
		value = value if value > 0 else 10**(-30)
		value_list.append(value)
	
	#plot
	import matplotlib.cm as cm
	from matplotlib.colors import Normalize, LogNorm
	import numpy as np
	plt.clf()
	c_list = list()
	cmap = cm.magma
	norm = Normalize(vmin=np.log(min_sol), vmax=np.log(max_sol))
	c_list = [cmap(norm(np.log(value))) for value in value_list]
	value_list_log = [np.log(v) if v >0.0 else 10**(-10) for v in value_list]

	ax = plt.gca()
	sc = ax.scatter(x_list, y_list, c=value_list_log, alpha=0.9, linewidths=0.0, cmap=cm.magma, norm=norm)
	ax.set_xlim([-0.5, kmax+1.0])
	ax.set_ylim([-0.5, kmax+1.0])
	ax.set_aspect(1.0)
	plt.xticks(np.arange(0, kmax+1, int(kmax/5)))
	plt.yticks(np.arange(0, kmax+1, int(kmax/5)))
	sns.despine()
	plt.colorbar(sc)
	
	xlabel='Neighbors in state {}'.format(states[0])
	ylabel='Neighbors in state {}'.format(states[1])
	ax.set_xlabel(xlabel,fontsize=18)
	ax.set_ylabel(ylabel,fontsize=18)
	#ax.set(xlabel='Neighbors in state {}'.format(states[0]),)
	plt.title('Distribution for State '+str(state))
	plt.tight_layout()
	plt.savefig(outpath)	



def plot_trajectories_individual(solution_csv, clustering_csv, outfolder):
	import pandas as pd
	import os

	outfolder = outfolder if outfolder.endswith('/') else outfolder+'/'

	if not os.path.exists(outfolder):
		os.makedirs(outfolder)

	sol = pd.read_csv(solution_csv, header=1, sep=';')
	sol = sol.to_dict(orient='record')

	clustering = pd.read_csv(clustering_csv, header=1, sep=';')
	clustering = clustering.to_dict(orient='record')

	for i, line in enumerate(sol):
		states = sorted(list(set([eval(key)[0] for key in line.keys()])))
		assert(len(states) == 2) # only works for 2 states so far
		for s in states:
			iz = str(i).zfill(10)
			outpath = outfolder+iz+'vis_{}_{}.jpg'.format(s, 'time')
			line_new = {k:v for k,v in line.items() if eval(k)[0] == s}
			plot_trajectories_individual_line(line_new, clustering, outpath, s, states, 10**(-10), 1.0)



if __name__ == '__main__':
	#plot_trajectories_individual('output/SIS/ame_SIS_1326_trajectories_individual.csv', 'output/SIS/clustering_SIS_1326.csv', 'output/SIS/outfull')
	plot_trajectories_individual('output/SIS/ame_SIS_122_trajectories_individual.csv', 'output/SIS/clustering_SIS_122.csv', 'output/SIS/out122x')