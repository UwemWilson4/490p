import math
import random
import numpy as np
from numpy.linalg import pinv, det
from eva import *
from eva.seal import *
from eva.ckks import *
from eva.metric import valuation_mse


def gen_fake_data():
	result = []
	for i in range(10):
		if i < 2:
			result.append([0, 2, 1, 1, 3, 0, 1, 2])
		elif i >= 2 and i < 5:
			result.append([1, 1, 3, 0, 2, 0, 1, 3])
		elif i >= 5 and i < 7:
			result.append([0, 1, 2, 1, 2, 3, 3, 1])
		elif i >= 7 and i < 8:
			result.append([0, 3, 0, 2, 3, 0, 2, 3])

	return np.matrix(result[:])


def vec_from_pred(n, pred):
    # Returns a vector v in {0,1}^n s.t.
    # v[i] = pred(i)
    return [1 if pred(ell) else 0 for ell in range(n)]

# sigmoid function needed for the logistic regression algorithm
def np_sigmoid_func(x):
	return 0.5 - np.dot(1.73496, np.dot(x, (1/8))) + np.dot(4.19407, np.dot(x, (1/8))**3) - np.dot(5.43402, np.dot(x, (1/8))**5) + np.dot(2.50739, np.dot(x, (1/8))**7)

def sigmoid_func(x):
	return 0.5 - 1.73496*(x * (1 / 8)) + 4.19407*((x * (1 / 8))**3) - 5.43402*((x * (1 / 8))**5) + 2.50739*((x * (1 / 8))**7)


"""Performs a rotation of a to the left by l. The EVA documentation does not specify
   this explicitly as the was to perform rotations, but examples in the code support
   this."""
def rotate(a, l):
	return a << l

def replicate(n, vec):
	replicate_res = [0 for i in range(n)]
	one_vector = [0 for i in range(n)]
	for i in range(n):
		one_vector[i] = 1
		# Converts every slot in the ciphertext to 0 except for the ith index
		temp_ciphertext = vec * one_vector
		# Populate each slot of the ciphertext with the given value
		for j in range(int(math.log(n, 2))):
			temp_ciphertext += rotate(temp_ciphertext, 2**j)

		replicate_res[i] = temp_ciphertext
	return replicate_res

def mat_vec_mult(mat, vec):
	replicate_length = len(vec)
	# The matrix-vector multiplication returns a ciphertext
	mat_vec_result = mat[0] * vec[0]
	for i in range(1, replicate_length):
		# Get the ith row (what would normally be the column, according to the algo)
		# and multiply it with the corresponding entry in vec.
		a_i = mat[i]
		#Output('a', a_i)
		mat_vec_result += a_i * vec[i]

	return mat_vec_result

def cp_mat_mult(matA, matB):
	n = len(matB)
	result = [None for _ in range(n)]
	for i in range(n):
		replicated_bi = replicate(n, matB[i])
		result[i] = mat_vec_mult(matA, replicated_bi)

	return result

def he_log_reg(input_matrix, beta_weights, y):
	num_rows = len(input_matrix)
	num_cols = len(input_matrix[0])
	d = num_rows
	n = d**2
	log_reg = EvaProgram('log_reg', vec_size=d)
	with log_reg:
		y_i = Input('y')
		B = Input('beta_weights')
		print(B.term)
		Output('test', B)
		# Compute hessian approximation before entering the loop
		xtxi_cols = [Input(f'xtxi_{i}') for i in range(d)]
		hessian_inverse_approx = [i * 4 for i in xtxi_cols]
		# Algorithm for logistic regression
		for k in range(2):
			# Replicate the beta_weights input
			num_weights = len(beta_weights)
			# For multiplications involving B
			replicate_res = replicate(num_weights, B)

			cols = [Input(f'x{i}') for i in range(d)]

			# Output the result of replication for testing purposes
			#Output('Replicated', replicate_res)

			# CP-MatVecMult
			mat_vec_result = mat_vec_mult(cols, replicate_res)

			# Output the result of matrix-vector multiplication for testing purposes
			#Output('MatVecMultRes', mat_vec_result)

			# Apply the sigmoid function to the result of the MatVec multiplication
			p = sigmoid_func(mat_vec_result)

			Output('SigmoidRes', p)
			

			# RP-MatVecMult
			# The inputs to RP-MatVecMult should be the x inputs (which we treat as the inputs same
			# from encrypting X_transpose) and (y_i - p) as an encrypted vector.
			zeros = [0 for i in range(num_rows)]

			# compute the dot product for the first x input and (y_i - p)
			rp_mat_vec_res = cols[0] * (y_i - p)
			for i in range(int(math.log(num_cols, 2))):
				rp_mat_vec_res += rotate(rp_mat_vec_res, 2**i)
			zeros[0] = 1
			rp_mat_vec_res = rp_mat_vec_res * zeros
			for i in range(1, num_rows):
				a_i = cols[i]
				zeros[i] = 1
				# DotProd between two vectors
				dot_prod = cols[i] * (y_i - p)
				for j in range(int(math.log(num_cols, 2))):
					dot_prod += rotate(dot_prod, 2**j)
				rp_mat_vec_res += dot_prod * zeros

			#Output result of RP matrix vector multiplication for testing purposes
			g = rp_mat_vec_res
			Output('g', g)
			
			length_g = len(y) 
			replicated_g = replicate(length_g, g)
			res = mat_vec_mult(hessian_inverse_approx, replicated_g)
			new_beta_weights = B - res
			Output('new_beta', new_beta_weights)
			B = new_beta_weights
			Output('beta_weights_final', B)
			print(B)
			print(f'Iteration {k}')
		

	# Set parameters
	log_reg.set_input_scales(30)
	log_reg.set_output_ranges(30)

	# Compile the program
	compiler = CKKSCompiler()
	compiled_log_reg, params, signature = compiler.compile(log_reg)

	public_ctx, secret_ctx = generate_keys(params)


	inputs = {}
	# The the transpose to more easily get to the cols
	xtxi_t = pinv(np.matmul(input_matrix.T, input_matrix)).T
	for index, row in enumerate(xtxi_t):
		inputs[f'xtxi_{index}'] = row.tolist()[0]
	a = [elem for row in input_matrix for elem in row]
	inputs['beta_weights'] = beta_weights
	inputs['y'] = y
	for index, column in enumerate(input_matrix):
		inputs[f'x{index}'] = column.tolist()[0]
	print(inputs)
	encInputs = public_ctx.encrypt(inputs, signature)
	encOutputs = public_ctx.execute(compiled_log_reg, encInputs)
	outputs = secret_ctx.decrypt(encOutputs, signature)

	# Run the program on unencrypted inputs to get reference results
	reference = evaluate(compiled_log_reg, inputs)
	print("Evaluated output:")
	print(reference['beta_weights_final'])
	print("\n\n")

	# Print actual outputs
	print("Actual output:")
	print(outputs['beta_weights_final'])
	#print(outputs['test2'])
	print("\n\n")

	return outputs

# Finds the smallest power of 2 greater than or equal to x
def power_bit_length(x):
	return 2**(x-1).bit_length()

# Suppose that vec has k filled slots and n - k unfilled slots. 
# duplicated duplicates the k filled slots throughout the rest
# of vec
def duplicate(n, k, vec):
	result = 0
	for i in range(math.log(n / power_bit_length(k), 2) - 1):
		result += rotate(vec, power_bit_length(k))

def cp_rep_mat_mult(n, matA, matB):
	result = 0
	for i in range(n):
		result += matA[i] * matB[i]

	return result

def inverse_slots(w, n):
	guess = [3 for _ in range(n)]
	for i in range(3):
		w_i = 2 + -(w * guess)
		guess = w_i

	return guess

# Assume that we want X, S, and xt_x_i to be column-packed. This means
# that we have to pass in transposes so that columns are rows. 
def hom_gwas(X, beta, y, p, xt_x_i, xt, S):
	d = len(beta)
	assert(len(beta) == X.shape[0])
	assert(len(beta) == len(y))
	assert(len(beta) == len(p)) 

	# Identity matrix used in gwas
	id_matrix = np.identity(d, dtype=int)
	print(id_matrix)
	# X transpose
	xt = X.T

	gwas = EvaProgram('gwas', vec_size=d)
	with gwas:
		_X = [Input(f'x{i}') for i in range(d)]
		_xt = [Input(f'xt_{i}') for i in range(d)]
		_xtxi = [Input(f'xtxi_{i}') for i in range(d)]
		_id = [Input(f'id_{i}') for i in range(d)]
		_beta = Input('beta')
		_y = Input('y')
		_p = Input('p')

		w = _p * (1 - _p)
		# Calculate inverse slots
		w_i = inverse_slots(w, d)
		replicated_beta = replicate(d, _beta)
		z = mat_vec_mult(_X, replicated_beta) + w_i* (_y - _p)
		#Output('z', w_i)
		Output('z', z)

		res = cp_mat_mult(_xtxi, _X)
		res = cp_rep_mat_mult(d, res, _xt)
		Output('res', res)
		replicated_z = replicate(d, z)
		iz = mat_vec_mult(_id, replicated_z)
		resz = res * z
		z_prime = iz - resz
		Output('z_prime', z_prime)

	gwas.set_input_scales(30)
	gwas.set_output_ranges(30)

	compiler = CKKSCompiler()
	compiled_gwas, params, signature = compiler.compile(gwas)

	public_ctx, secret_ctx = generate_keys(params)

	inputs = {'beta': beta, 'y': y, 'p': p}
	for index, row in enumerate(X):
		inputs[f'x{index}'] = row.tolist()[0]
	for index, row in enumerate(xt):
		inputs[f'xt_{index}'] = row.tolist()[0]
	for index, row in enumerate(id_matrix):
		inputs[f'id_{index}'] = row
	#for index, row in enumerate(S):
	#	inputs[f's{index}'] = row
	for index, row in enumerate(xt_x_i):
		inputs[f'xtxi_{index}'] = row.tolist()[0]
	print(inputs)

	encInputs = public_ctx.encrypt(inputs, signature)
	encOutputs = public_ctx.execute(compiled_gwas, encInputs)
	outputs = secret_ctx.decrypt(encOutputs, signature)

	# Run the program on unencrypted inputs to get reference results
	reference = evaluate(compiled_gwas, inputs)
	print("Evaluated output:")
	print(reference['z_prime'])
	print("\n\n")

	# Print actual outputs
	print("Actual output:")
	print(outputs['z_prime'])
	#print(outputs['test2'])
	print("\n\n")


def test_no_enc():
	# generate fake data and outcome vector
	data = gen_fake_data()
	data = data.T
	#print(data)
	y = [1, 1, 0, 0, 1, 1, 0, 0]
	y_m = np.reshape(np.matrix(y, dtype=int), (len(data), 1))
	num_rows = len(data)
	num_cols = data.shape[1]
	beta_weights = np.zeros((num_cols, 1), dtype=int)
	print(f'num_rows: {num_rows}\nnum_cols: {num_cols}\nbeta_weights: {beta_weights}\nY: {y_m}\n')

	for i in range(num_rows):
		p = np.dot(data, beta_weights)
		g = np.matmul(data.T, (y_m - p))
		xt_x = np.matmul(data.T, data)
		h_approx = 4 * pinv(np.matmul(data.T, data))
		b_new = beta_weights - (np.matmul(h_approx, g))
		beta_weights = b_new

	print(beta_weights)


if __name__ == '__main__':
	d = 8
	matA = [[random.randint(0, 3) for _ in range(d)] for _ in range(d)]
	matA = np.matrix(matA)
	matB = [[random.randint(0, 3) for _ in range(d)] for _ in range(d)]
	print("Newton-Raphson with no encryption:")
	#test_no_enc()
	print("\n\n")

	data = gen_fake_data()
	beta_weights = [0 for i in range(d)]
	y = [random.randint(0, 1) for _ in range(d)]
	print("Newton-Raphson with encryption:")
	# The the transpose to more easily get to the cols
	xtxi_t = pinv(np.matmul(matA.T, matA)).T
	outputs = he_log_reg(matA, beta_weights, y)
	hom_gwas(matA, outputs['beta_weights_final'], y, outputs['SigmoidRes'], xtxi_t, matA.T, matB)