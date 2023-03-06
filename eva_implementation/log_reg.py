import math
import numpy as np
from numpy.linalg import pinv, det
from eva import EvaProgram, Input, Output

y = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0]
def gen_fake_data():
	result = []
	for i in range(10):
		if i < 2:
			result.append([0, 2, 1, 1, 3, 0, 1, 2,])
		elif i >= 2 and i < 5:
			result.append([1, 1, 3, 0, 2, 0, 1, 3,])
		elif i >= 5 and i < 7:
			result.append([0, 1, 2, 1, 2, 3, 3, 1,])
		elif i >= 7 and i < 9:
			result.append([0, 3, 0, 2, 3, 0, 2, 3,])

	return np.matrix(result[:])



# sigmoid function needed for the logistic regression algorithm
def np_sigmoid_func(x):
	return 0.5 - np.dot(1.73496, np.dot(x, (1/8))) + np.dot(4.19407, np.dot(x, (1/8))**3) - np.dot(5.43402, np.dot(x, (1/8))**5) + np.dot(2.50739, np.dot(x, (1/8))**7)

def sigmoid_func(x):
	return 0.5 - 1.73496*(x / 8) + 4.19407*((x / 8)**3) - 5.43402*((x / 8)**5) + 2.50739*((x / 8)**7)

beta_weights = [0 for i in range(8)]
num_rows = 8
num_cols = 8

"""Performs a rotation of a to the left by l. The EVA documentation does not specify
   this explicitly as the was to perform rotations, but examples in the code support
   this."""
def rotate(a, l):
	return a << l

log_reg = EvaProgram('log_reg', 8*8)
with log_reg:
	y = Input('y')
	B = Input('beta_weights')
	# Algorithm for logistic regression
	for k in range(num_rows):
		# Replicate the beta_weights input
		num_weights = len(beta_weights)
		replicate_res = [None for i in range(64)]
		one_vector = [0 for i in range(64)]
		for i in range(num_weights):
			one_vector[i] = 1
			# Converts every slot in the ciphertext to 0 except for the ith index
			temp_ciphertext = B * one_vector
			# Populate each slot of the ciphertext with the given value
			for j in range(int(math.log(num_weights, 2))):
				temp_ciphertext += rotate(temp_ciphertext, 2**j)

			replicate_res[i] = temp_ciphertext
			print(replicate_res)

		# Output the result of replication for testing purposes
		# Output('Replicated', replicate_res)

		# CP-MatVecMult
		replicate_length = len(replicate_res)
		# The matrix-vector multiplication returns a ciphertext
		mat_vec_result = Input('x0') * replicate_res[0]
		for i in range(1, replicate_length):
			# Get the ith row (what would normally be the column, according to the algo)
			# and multiply it with the corresponding entry in replicate_res.
			a_i = Input(f'x{i}')
			mat_vec_result += a_i * replicate_res[i]

		# Output the result of matrix-vector multiplication for testing purposes
		Output('MatVecMultRes', mat_vec_result)

		# Apply the sigmoid function to the result of the MatVec multiplication
		p = sigmoid_func(mat_vec_result)

		Output('SigmoidRes', p)


		# RP-MatVecMult
		# The inputs to RP-MatVecMult should be the x inputs (which we treat as the inputs same
		# from encrypting X_transpose) and (y - p) as an encrypted vector.
		zeros = [0 for i in range(num_rows)]

		# compute the dot product for the first x input and (y - p)
		rp_mat_vec_res = Input('x0') * (y - p)
		for i in range(math.log(num_cols, 2)):
			rp_mat_vec_res += rotate(rp_mat_vec_res, 2**i)
		zeros[0] = 1
		rp_mat_vec_res = rp_mat_vec_res * zeros

		for i in range(1, num_rows):
			a_i = Input(f'x{i}')
			zeros[i] = 1
			# DotProd between two vectors
			dot_prod = Input('x{i}') * (y - p)
			for j in range(math.log(num_cols, 2)):
				dot_prod += rotate(dot_prod, 2**j)
			rp_mat_vec_res += dot_prod * zeros

		#Output result of RP matrix vector multiplication for testing purposes
		Output('RPMatVecRes', rp_mat_vec_res)
		g = rp_mat_vec_res

		x_xt_inverse = Input('X_Xt_inverse')
		hessian_inverse_approx = 4 * x_xt_inverse

		new_beta_weights = B - (hessian_inverse_approx * g)
		B = new_beta_weights

	# Output final beta weights for testing
	Output('beta_weights_final', B)



def test_no_enc():
	# generate fake data and outcome vector
	data = gen_fake_data()
	data = data.T
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
	test_no_enc()

	log_reg.set_input_scales(30)
	log_reg.set_output_ranges(30)

	compiler = CKKSCompiler()
	compiled_log_reg, params, signature = compiler.compile(log_reg)








