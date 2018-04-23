import numpy as np
import sympy
import os
import traceback

#------------------------------------------------------
# Python version check
#------------------------------------------------------

if 1/5 == 0: #there's prbly. a better way to do this
	raise RuntimeError('Do not use Python 2.x!')

#------------------------------------------------------
# Logging
#------------------------------------------------------

import logging
logger = logging.getLogger('LumpingLogger')
logger.setLevel(logging.INFO)
logpath = 'LumpingLog.log'
try:
	# Use different logger output when testing
	import run_all_tests
	logpath = 'tests/LumpingLogTest.log'
except:
	pass
fh = logging.FileHandler(logpath, mode='w') # change to a to overwrite
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(process)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('-------------------------------------------------')
logger.info('                 Start Logging                   ')
logger.info('-------------------------------------------------')


#------------------------------------------------------
# Miscellaneous
#------------------------------------------------------

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
	return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def float2str(f):
	if 'USE_HIGH_PRECISION_STRINGS' in globals() and not globals['USE_HIGH_PRECISION_STRINGS']:
		return str(f)
	from decimal import Decimal
	return str(Decimal(repr(f)))

def to_str(x):
	try:
		s = float2str(x)
		return s
	except:
		# try to get commas in str output
		if 'numpy' in str(type(x)):
			try:
				x = x.tolist()
			except:
				pass
		s = str(x)
		return s

def create_normalized_np(v, return_original_if_zero = False):
	v = np.array(v)
	partition = float(np.sum(v))
	if isclose(partition, 0.0):
		if return_original_if_zero:
			return np.array(v)
		raise ValueError('Cannot normalize vector, sum is zero.')
	if (v < 0.0).any():
		raise ValueError('Cannot normalize vector, negative values.')
	v = v / partition
	return v

# # careful: does not handle sets correctly
# def dict_to_uniquestr(dict_or_value):
# 	if isinstance(dict_or_value,dict):
# 		dict_strkey = {str(k):v for k,v in dict_or_value.items()}
# 		result = [(k,dict_strkey[k]) for k in sorted(dict_strkey.keys())]
# 		return str(result)
# 	else:
# 		return str(dict_or_value)

# def dict_to_uniquekey(dict_or_value):
# 	import hashlib
# 	import random
# 	#s = dict_to_uniquestr(dictor_value) some bug here
# 	s = sorted(str(dict_or_value)) 
# 	s = ''.join(s)
# 	s = s.strip()
# 	random.seed(s)
# 	#hash_object = hashlib.md5(s.encode('utf-8'))
# 	x = str(random.random())[3:13]
# 	#x = hash_object.hexdigest()
# 	return x

#------------------------------------------------------
# File generation
#------------------------------------------------------

def genrate_file_ame(model):
	outfolder = model['output_dir']
	import os
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)

	target = open("templates/ode_python_ame.template", 'r')
	template = target.read()
	target.close()

	# execute python code in curly brackets
	# * is an ugly hack
	template = template.replace(r'}}',r'*{{')
	template = template.split(r'{{')
	for i in range(len(template)):
		if template[i].endswith(r'*'):
			template[i] = eval(template[i][:-1])
			template[i] = to_str(template[i])
	template = ''.join(template)

	target = open(model['output_path'], 'w')
	target.write(template)
	target.close()
	logger.info('Successfully created file.')
	return os.path.abspath(model['output_path'])


#------------------------------------------------------
# Combinatorics
#------------------------------------------------------

def gen_mset(kmax, dim):
	import itertools
	one_dim = list(range(kmax+1))
	mset = itertools.product(one_dim, repeat=dim)
	mset = {m for m in mset if np.sum(m) <= kmax}
	return mset

def gen_mset_for_degree(degree, dim):
	import itertools
	assert(degree > 0 and dim>0)
	mset = gen_mset(degree, dim)
	mset = {m for m in mset if np.sum(m)==degree}
	return set(mset)

def elemcount_mset(kmax, dim):
	from scipy.special import binom
	return binom(kmax+dim, dim-1) * (kmax+1)/dim

def number_of_odes(kmax, dim):
	from scipy.special import binom
	return binom(kmax+dim, dim-1) * (kmax+1)

def elemcount_mset_for_degree(degree, dim):
	from scipy.special import binom
	return binom(degree+dim-1, dim-1)

def multinomial_pmf(choice_vector, probability_vector):
	from scipy.stats import multinomial
	if np.sum(choice_vector) == 0:
		return 1.0
	return multinomial.pmf(choice_vector, n=np.sum(choice_vector), p=probability_vector)


#------------------------------------------------------
# Symbolic symplification
#------------------------------------------------------

#unused
def ode_simplifyformula):
	original = formula
	logger.debug('get formula: {}'.format(formula))
	try:
		prefix = ""
		suffix = ""
		if "=" in formula:
			prefix = formula.split("=")[0]+"="
			formula = formula.split("=")[1]
			assert("=" not in formula)
		if "if" in formula:
			formula = formula.split("if")[0]
			suffix = "if"+formula.split("if")[1]
			assert("=" not in formula)

		formula = formula.replace("[","_B___O_").replace("]","_B___C_")
		formula = formula.replace("{","_CB___O_").replace("}","_CB___C_")
		new_formula = sympy.sympify(formula)
		new_formula = str(new_formula)
		new_formula = prefix + new_formula.replace("_B___O_", "[").replace("_B___C_","]").replace("_CB___O_","{").replace("_CB___C_","}")+suffix
		logger.debug('return converted formula: {}'.format(new_formula))
		return new_formula
	except:
		import sys
		logger.warn('could not convert formula: {} ({})'.format(formula,sys.exc_info()[0]))
		#print(formula, sys.exc_info()[0])
		return original


#------------------------------------------------------
# Compare models
#------------------------------------------------------

def compare_models(model1, model2):
	''' Computes difference of two models as the maximal L2 distance between points of their correspondig trajectories'''
	if 'trajectories' not in model1 or len(model1['trajectories']) == 0:
		raise ValueError('No trajectories in model to compare.')

	states = sorted(list(model1['trajectories'].keys()))
	if states != sorted(list(model2['trajectories'].keys())):
		raise ValueError('Cannot compare models, as they contain different states.')

	error_list = list()
	sample_num = len(model1['trajectories'][states[0]])

	for i in range(sample_num):
		error = 0.0
		for state in states:
			error += (model1['trajectories'][state][i] - model2['trajectories'][state][i])**2
		error_list.append(np.sqrt(error))

	return np.max(error_list)


#------------------------------------------------------
# Write files
#------------------------------------------------------

def write_trajectory_plot(models, filepath, show_plot = False, state_to_color = None):
	''' plots trajectories of one or two models, saves as .png and .svg, do not include filending in filepath arguement'''

	import matplotlib.pyplot as plt

	if not type(models) is list:
		models = [models]

	def state_to_color_default(state):
		state = state.lower().strip()
		color_dict = {'s': 'blue', 'i': plt.get_cmap('gnuplot')(0.45), 'r': 'green', 'ii': plt.get_cmap('gnuplot')(0.575), 'iii': plt.get_cmap('gnuplot')(0.7)}
		return color_dict.get(state, None)

	if state_to_color is None:
		state_to_color = state_to_color_default
	trajectories1 = models[0]['trajectories']
	subtitle = models[0]['name']+'(-)'
	plt.clf()
	for state in trajectories1:
		plt.plot(models[0]['time'], trajectories1[state], label=state, color = state_to_color(state), linewidth = 2)
	try:
		trajectories2 = models[1]['trajectories']
		subtitle += '    '+models[1]['name']+'(--)  '
		for state in trajectories2:
			plt.plot(models[1]['time'], trajectories2[state], label=state, ls='--', color = state_to_color(state), linewidth = 2)
	except IndexError:
		pass
	ncol = 2 #if len(models[0]['states']) > 3 else 1
	plt.legend(loc='best', ncol = ncol)
	plt.xlabel('t')
	plt.suptitle(subtitle)
	plt.grid()
	plt.savefig(filepath+'.png', dpi=300)
	plt.savefig(filepath+'.svg', format='svg', dpi=1200)
	if show_plot:
		plt.show()

def models_to_csv(models, filepath, header='sep=;\n', sep=';'):
	if not type(models) is list:
		models = [models]
	keys = set()
	for model in models:
		for key in model:
			if key not in ['trajectories', 'time']:
				keys.add(key)
				if sep in key:
					raise ValueError('Seperator sign in key.')
	keys = sorted(list(keys))
	with open(filepath,'w') as f:
		f.write(header)
		f.write(sep.join(keys)+'\n')
		line = list()
		for i in range(len(models)):
			line = list()
			for key in keys:
				rep = ',' if sep == ';' else ';'
				line.append(str(models[i].get(key,'')).replace(sep, rep).replace('\n',' --- '))
			f.write(sep.join(line))
			if i != list(range(len(models)))[-1]:
				f.write('\n')

def trajectories_to_csv(model, filepath, header='sep=;\n', sep=';'):
	trajectories = model['trajectories']
	time = model['time']
	# if folder != '' and not os.path.exists(folder):
	# 	os.makedirs(folder)
	# write csv
	with open(filepath, 'w') as f:
		states = sorted(list(trajectories.keys()))
		f.write(header)
		f.write('time'+sep+sep.join(states)+'\n')
		for i in range(len(trajectories[states[0]])):
			f.write(str(time[i])+sep)
			for state in states:
				s = sep if state != states[-1] else ''
				f.write(str(trajectories[state][i])+s)
			if i != list(range(len(trajectories[states[0]])))[-1]:
				f.write('\n')

