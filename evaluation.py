import sys
sys.dont_write_bytecode = True
import glob
import matplotlib
matplotlib.use('agg')	#run without an X-server
import pandas as pd
import matplotlib.pyplot as plt
import time

import ame
from utilities import *
import model_parser
import visualization

def write_2d_heatmap(model, errors, times, degree_cluster_number, proportionality_cluster_number, base_outpath):
	matplotlib.rcParams.update(matplotlib.rcParamsDefault)
	import seaborn as sns
	sns.reset_orig()
	plt.clf()

	pl = np.max(proportionality_cluster_number+degree_cluster_number) +1

	errorframe = np.empty((pl,pl))
	errorframe[:] = np.nan
	timeframe = np.empty((pl,pl))
	timeframe[:] = np.nan

	ax = plt.gca()
	for i, error in enumerate(errors):
		dg = degree_cluster_number[i]
		dp = proportionality_cluster_number[i]
		try:
			errorframe[dg][dp] = error
			timeframe[dg][dp] = times[i]
		except:
			print('error with range')
		
	#?errorframe = np.transpose(errorframe)
	errorframe = pd.DataFrame(data=errorframe, index=range(pl), columns=range(pl))
	timeframe = pd.DataFrame(data=timeframe, index=range(pl), columns=range(pl))
	errorframe.dropna(axis=0, how='all', inplace=True)
	errorframe.dropna(axis=1, how='all', inplace=True)
	timeframe.dropna(axis=0, how='all', inplace=True)
	timeframe.dropna(axis=1, how='all', inplace=True)

	outname_time = base_outpath.replace('ame_', '2D_timeplot_').replace('.py', '.pdf')
	outname_errors = base_outpath.replace('ame_', '2D_errorplot_').replace('.py', '.pdf')

	sns.set(context='paper', style='white', font='sans-serif', font_scale=2.0)

	ax = sns.heatmap(errorframe, vmin=0.0, vmax=np.max(errors), square=True, ax=ax, cmap="RdYlGn_r", xticklabels=4, yticklabels=4)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	ax.xaxis.set_ticks_position('none') 
	ax.yaxis.set_ticks_position('none') 
	ax.set_xlabel('Number Degree Clusters', fontsize=15)
	ax.set_ylabel('Number Proportionality Clusters', fontsize=15)
	errorframe.to_csv(outname_errors.replace('.pdf', '.csv'))
	plt.savefig(outname_errors, format='pdf', bbox_inches='tight')
	plt.clf()
	ax = sns.heatmap(timeframe, vmin=0.0, vmax=np.max(times), cmap="YlGnBu", xticklabels=4, yticklabels=4)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	ax.set_xlabel('Number Degree Clusters', fontsize=20)
	ax.set_ylabel('Number Proportionality Clusters', fontsize=20)
	ax.xaxis.set_ticks_position('none') 
	ax.yaxis.set_ticks_position('none') 
	timeframe.to_csv(outname_time.replace('.pdf', '.csv'))
	plt.savefig(outname_time, format='pdf', bbox_inches='tight')
	plt.clf()

def evaluate_model(modelpath, parameter_change_list, base_clusters=(50,50), break_when_max=True):
	global runtimes
	errors = list()
	cluster_number = list()
	degree_cluster_number = list()
	proportionality_cluster_number = list()
	times = list()

	base = model_parser.read_model(modelpath)
	base['name_extension'] = 'baseline'
	base = model_parser.clean_model(base)

	logger.info('evaluate with baseline')
	ame.generate_and_solve(base, True, True)

	for new_parameters in parameter_change_list:
		model = model_parser.read_model(modelpath)
		for key, value in new_parameters.items():
			model[key] = value
		model = model_parser.clean_model(model)

		ame.generate_and_solve(model, True, False)

		#stop
		if base['actual_cluster_number'] == model['actual_cluster_number']:
			logger.info('maximal cluster number is reached.')
			dcn = model['lumping']['degree_cluster']
			pcn = model['lumping']['proportionality_cluster']
			write_2d_heatmap(model, errors+[0.0], times+[base['time_elapsed']], degree_cluster_number+[dcn], proportionality_cluster_number+[pcn], base['output_path'])
			if break_when_max:
				break

		errors.append(compare_models(model, base))
		cluster_number.append(model['actual_cluster_number'])
		degree_cluster_number.append(model['lumping']['degree_cluster'])
		proportionality_cluster_number.append(model['lumping']['proportionality_cluster'])
		times.append(model['time_elapsed'])

		outname_time = base['output_path'].replace('ame_', 'timeplot_').replace('.py', '.pdf')
		outname_errors = base['output_path'].replace('ame_', 'errorplot_').replace('.py', '.pdf')
		visualization.plot_scatter(cluster_number + [base['actual_cluster_number']], errors + [0.0], outname_errors, color=2)
		visualization.plot_scatter(cluster_number + [base['actual_cluster_number']], times + [base['time_elapsed']], outname_time, color=0, y_label = 'Time (s)')


		write_2d_heatmap(model, errors+[0.0], times+[base['time_elapsed']], degree_cluster_number+[base_clusters[0]], proportionality_cluster_number+[base_clusters[1]], base['output_path'])	
		#write_2d_heatmap(model, errors, times, degree_cluster_number, proportionality_cluster_number, base['output_path'])


def iter_clusternum():
	import random
	change_list = list()
	import itertools
	#bins = itertools.product(range(35),range(35))
	#bins = sorted(bins, key = lambda x: x[0]+x[1])
	bins = range(100)
	#random.shuffle(bins)
	#bins = list(zip(range(55),range(55))) + bins
	for x in bins:
		change_dict = {'lumping': {'degree_cluster': x+5, 'proportionality_cluster': x+5}}
		change_list.append(change_dict)

	evaluate_model('model/SIR3x.yml', change_list, break_when_max=False)



def iter_rates():
	modelpath = 'model/SIS.yml'
	for kmax in range(30):
		network = {'kmax': 20+kmax, 'degree_distribution': 'k**(-2.5) if k > 1 else 0.00001'}
		model = model_parser.read_model(modelpath)
		model['name_extension'] = 'baseline_'+str(kmax)
		model['network'] = network
		model = model_parser.clean_model(model)
		ame.generate_and_solve(model, True, False)


def eval_pa(filepath):
	import os
	os.system('cd LumPy && python3 pa.py "{}" --nolumping'.format(filepath))

def eval_mc(filepath):
	import os
	os.system('cd LumPy && python3 simulation.py "{}" --nodes=3000 --runs=6'.format(filepath))

def eval_ame(filepath):
	try:
		model = model_parser.parse_file(filepath)
		ame.generate_and_solve(model, True, False)
	except  Exception as e:
		raise
		logger.error('Error at file '+filepath+'  '+str(e))


#iter_clusternum()

def evaluate_models():
	import glob, random
	from multiprocessing import Process

	models = glob.glob('model/SIIS.yml')
	random.shuffle(models)
	for m in models:
		print(m)
		#eval_ame(m)
		new_model_path = m.replace('.yml','.model').replace('model/', 'model/')
		#eval_pa(m.replace(new_model_path)
		#eval_pa(m.replace(new_model_path)

		p1 = Process(target=eval_ame, args=(m,))
		p1.start()
		p2 = Process(target=eval_mc, args=(new_model_path,))
		p2.start()
		p3 = Process(target=eval_pa, args=(new_model_path,))
		p3.start()

		p1.join()
		p2.join()
		p3.join()


#iter_clusternum()
evaluate_models()


