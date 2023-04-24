import math
import random
import time
from eva import *
from eva.seal import *
from eva.ckks import *
from eva.metric import valuation_mse

def test_simple_addition(iterations, trials):
	results = []
	simple_add = EvaProgram('simple_add', 1)
	with simple_add:
		result = Input('zero')
		for i in range(trials):
			start = time.perf_counter()
			for j in range(iterations):
				val = random.randint(0, 100)
				result += val
			Output('output', result)
			end = time.perf_counter()
			duration = (end - start) * 10**6
			results.append(duration)

	simple_add.set_input_scales(30)
	simple_add.set_output_ranges(30)

	compiler = CKKSCompiler()
	program, params, signature = compiler.compile(simple_add)

	public_ctx, secret_ctx = generate_keys(params)

	zero = [0]
	inputs = {'zero': zero}
	encInputs = public_ctx.encrypt(inputs, signature)
	encOutputs = public_ctx.execute(program, encInputs)

	return results

def test_vector_addition(iterations, trials):
	results = []
	test_vecs = [[random.randint(2, 102) for _ in range(iterations)] for _ in range(iterations)]

	vec_add = EvaProgram('vec_add', iterations)
	with vec_add:
		rand_vals = Input('rand_vals')
		for i in range(trials):
			start = time.perf_counter()
			for j in range(iterations):
				result = rand_vals + test_vecs[j]
				Output('output', result)
			end = time.perf_counter()
			duration = (end - start) * 10**6
			results.append(duration)

	vec_add.set_input_scales(30)
	vec_add.set_output_ranges(30)

	compiler = CKKSCompiler()
	program, params, signature = compiler.compile(vec_add)

	public_ctx, secret_ctx = generate_keys(params)

	rand_vals = [random.randint(2, 102) for _ in range(iterations)]
	inputs = {'rand_vals': rand_vals}
	encInputs = public_ctx.encrypt(inputs, signature)
	encOutputs = public_ctx.execute(program, encInputs)

	return results

def test_simple_multiplication(iterations, trials):
	results = []
	test_vecs = [[random.randint(2, 102) for _ in range(iterations)] for _ in range(iterations)]

	simple_mult = EvaProgram('simple_mult', iterations)
	with simple_mult:
		rand_vals = Input('rand_vals')
		for i in range(trials):
			start = time.perf_counter()
			for j in range(iterations):
				result = rand_vals * test_vecs[j]
				Output('output', result)
			end = time.perf_counter()
			duration = (end - start) * 10**6
			results.append(duration)

	simple_mult.set_input_scales(30)
	simple_mult.set_output_ranges(30)

	compiler = CKKSCompiler()
	program, params, signature = compiler.compile(simple_mult)

	public_ctx, secret_ctx = generate_keys(params)

	rand_vals = [random.randint(2, 102) for _ in range(iterations)]
	inputs = {'rand_vals': rand_vals}
	encInputs = public_ctx.encrypt(inputs, signature)
	encOutputs = public_ctx.execute(program, encInputs)

	return results

def test_vector_multiplication(iterations, trials):
	results = [] # output list of runtimes
	test_vecs = [[random.randint(2, 102) for _ in range(iterations)] for _ in range(iterations)] # list of lists containing random values

	vec_add = EvaProgram('vec_add', iterations)
	with vec_add:
		# `rand_vals` is a ciphertext 
		rand_vals = Input('rand_vals') 
		for i in range(trials):
			start = time.perf_counter()
			for j in range(iterations):
				result = rand_vals * test_vecs[j]
				Output('output', result)
			end = time.perf_counter()
			duration = (end - start) * 10**6
			results.append(duration)

	vec_add.set_input_scales(30)
	vec_add.set_output_ranges(30)

	compiler = CKKSCompiler()
	program, params, signature = compiler.compile(vec_add)

	public_ctx, secret_ctx = generate_keys(params)

	rand_vals = [random.randint(2, 102) for _ in range(iterations)]
	inputs = {'rand_vals': rand_vals}
	encInputs = public_ctx.encrypt(inputs, signature)
	encOutputs = public_ctx.execute(program, encInputs)

	return results

if __name__ == "__main__":
	iters = 128
	trials = 5

	simple_add_results = test_simple_addition(iters, trials)
	vec_add_results = test_vector_addition(iters, trials)
	simple_mult_results = test_simple_multiplication(iters, trials)
	vec_mult_results = test_vector_multiplication(iters, trials)
	print(simple_add_results)
	print(vec_add_results)
	print(simple_mult_results)
	print(vec_mult_results)