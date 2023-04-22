import math
import random
import time
from eva import *
from eva.seal import *
from eva.ckks import *
from eva.metric import valuation_mse

def test_simple_additions(iterations, trials):
	results = []
	simple_additions = EvaProgram('simple_additions', 1)
	with simple_additions:
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

	simple_additions.set_input_scales(30)
	simple_additions.set_output_ranges(30)

	compiler = CKKSCompiler()
	program, params, signature = compiler.compile(simple_additions)

	public_ctx, secret_ctx = generate_keys(params)

	zero = [0]
	inputs = {'zero': zero}
	encInputs = public_ctx.encrypt(inputs, signature)
	encOutputs = public_ctx.execute(program, encInputs)

	return results


if __name__ == "__main__":
	iters = 100
	trials = 5

	simple_addition_results = test_simple_additions(iters, trials)
	print(simple_addition_results)