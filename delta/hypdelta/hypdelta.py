from 

def hypdelta(D, device=['cpu', 'cuda'], strategy=["naive", "condenced", "heuristic","CCL"], kwargs): # device can be cuda
	if strategy == "naive":
		if device == 'cpu':
			delta = calculate_naive_delta(kwargs)
		elif device == 'gpu':
			delta = calculate_naive_delta_gpu(kwargs)
	# elif strategy == "condenced":
	# 	if device == 'cpu':
	# 		delta = calculate_condenced_delta(kwargs) # heuristic=True
	# 	elif device == 'gpu':
	# 		raise ValueError('AAA')

	# if strategy == "CCL":
	# 	if device == 'cpu':
	# 		delta = calculate_CCL_delta(kwargs)
	# 	elif device == 'gpu':
	# 		delta = calculate_CCL_delta_gpu(kwargs)

	return delta


def CCL_cpu(D, l=0.2)

def CCL_gpu(D, l=0.2)

def condenced(D, heuristic)

def naive_cpu(D)

def naive_gpu(D)



		