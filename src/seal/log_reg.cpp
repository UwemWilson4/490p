#include <iostream>
#include <vector>
#include <armadillo>
#include <stdlib.h>
#include <chrono>

#include "log_reg.h"
#include "helpers.h"

#define ITERS 1


using namespace std;
using namespace arma;
using namespace seal;
using namespace std::chrono;

typedef std::chrono::duration<long, std::ratio<1, 1000000> > ms_duration;

/* Prints information about Ciphertext's parameters */
void print_Ciphertext_Info(string ctx_name, Ciphertext ctx, shared_ptr<SEALContext> context)
{
    cout << "/" << endl;
    cout << "| " << ctx_name << " Info:" << endl;
    cout << "|\tLevel:\t" << context->get_context_data(ctx.parms_id())->chain_index() << endl;
    cout << "|\tScale:\t" << log2(ctx.scale()) << endl;
    ios old_fmt(nullptr);
    old_fmt.copyfmt(cout);
    cout << fixed << setprecision(10);
    cout << "|\tExact Scale:\t" << ctx.scale() << endl;
    cout.copyfmt(old_fmt);
    cout << "|\tSize:\t" << ctx.size() << endl;
    cout << "\\" << endl;
}

/* Ensures levels are equalized between two ciphertexts */
void equalize_levels(Ciphertext &a, Ciphertext &b, Evaluator &evaluator, EncryptionParameters params) {
	SEALContext context(params);
	auto tmp = make_shared<SEALContext>(context);

	auto start = high_resolution_clock::now();
	// cout << "Getting chain index a..." << endl;
	int level_a = tmp->get_context_data(a.parms_id())->chain_index();
	// cout << "Getting chain index b..." << endl;
	int level_b = tmp->get_context_data(b.parms_id())->chain_index();

	if (level_a > level_b) {
		if (level_a == 0) {
			cout << "Can't mod switch any further" << endl;
		}
		evaluator.mod_switch_to_inplace(a, b.parms_id());
	} else if (level_a < level_b) {
		if (level_b == 0) {
			cout << "Can't mod switch any further" << endl;
		}
		evaluator.mod_switch_to_inplace(b, a.parms_id());
	}
	auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << __func__ << ": " << duration.count() << endl;
}

/* Rotates a Ciphertext by k */
Ciphertext rotate_ciphertext(Ciphertext ctx, int k, GaloisKeys galois_keys, Evaluator &evaluator)
{
	auto start = high_resolution_clock::now();
	Ciphertext rotated;
	evaluator.rotate_vector(ctx, k, galois_keys, rotated);
	auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << __func__ << ": " << duration.count() << endl;

	return rotated;
}

/* The replicate function takes a vector encrypted as a Ciphertext and outputs a vector
	of Ciphertexts, where each Ciphertext is a replicate of ctx. This corresponds to a what
	would be a matrix with n columns that are all identical. */
vector<Ciphertext> replicate(Ciphertext ctx, int n, double scale, CKKSEncoder &ckks_encoder, Encryptor &encryptor, GaloisKeys galois_keys, RelinKeys relin_keys, Evaluator &evaluator, EncryptionParameters params)
{
	SEALContext context(params);
	auto tmp = make_shared<SEALContext>(context);
	auto start = high_resolution_clock::now();
	print_Ciphertext_Info("CTX", ctx, tmp);
	vector<Ciphertext> replicate_res(n);

	for (size_t i = 0; i < n; i++) {
		vector<double> one_vector(n, 0.0);
		for (size_t j; j < i; j++) {
			one_vector[j] = 1;
		}

		Plaintext pt_one_vector;
		Ciphertext ct_one_vector;
		// Encode the values in one_vector into a plaintext
		ckks_encoder.encode(one_vector, scale, pt_one_vector);
		encryptor.encrypt(pt_one_vector, ct_one_vector);

		// Converts every slot in the ciphertext to 0 except for the ith index
		Ciphertext temp_ciphertext;
		equalize_levels(ctx, ct_one_vector, evaluator, params);
		evaluator.multiply(ctx, ct_one_vector, temp_ciphertext);
		// print_Ciphertext_Info("temp_ciphertext", temp_ciphertext, tmp);
		// Relinearization
		evaluator.relinearize_inplace(temp_ciphertext, relin_keys);
		// Rescale
		evaluator.rescale_to_next_inplace(temp_ciphertext);
		// print_Ciphertext_Info("temp_ciphertext", temp_ciphertext, tmp);
		// Manual rescale
		temp_ciphertext.scale() = pow(2.0, 40);
		// print_Ciphertext_Info("temp_ciphertext", temp_ciphertext, tmp);

		for (size_t j = 0; j < (int)log2(n); j++) {
			// NOTE: may have to do a modulus switch here
			Ciphertext temp_rotated = rotate_ciphertext(temp_ciphertext, pow(2, j), galois_keys, evaluator);
			evaluator.add_inplace(temp_ciphertext, temp_rotated);
		}

		replicate_res[i] = temp_ciphertext;
		// print_Ciphertext_Info("replicate_res[i]", replicate_res[i], tmp);
	}
	auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << __func__ << ": " << duration.count() << endl;
	return replicate_res;
}

/* Performs matrix-vector multiplication. */
Ciphertext mat_vec_mult(vector<Ciphertext> mat, Ciphertext vec, CKKSEncoder &ckks_encoder, Encryptor &encryptor, GaloisKeys galois_keys, RelinKeys relin_keys, Evaluator &evaluator, EncryptionParameters params) {
	SEALContext context(params);
	auto tmp = make_shared<SEALContext>(context);
	auto start = high_resolution_clock::now();
	cout << "Replicating vec..." << endl;
	vector<Ciphertext> replicated_vec = replicate(vec, mat.size(), pow(2, 40), ckks_encoder, encryptor, galois_keys, relin_keys, evaluator, params);

	Ciphertext result;

    equalize_levels(mat[0], replicated_vec[0], evaluator, params);
	evaluator.multiply(mat[0], replicated_vec[0], result);

	// Relin
	evaluator.relinearize_inplace(result, relin_keys);
	
	// Rescale
	evaluator.rescale_to_next_inplace(result);
	result.scale() = pow(2.0, (int)log2(result.scale()));
	
	for (size_t i = 1; i < replicated_vec.size(); i++) {
		// cout << "mat_vec_mult ITERATION " << i << endl;
		equalize_levels(mat[i], replicated_vec[i], evaluator, params);
		Ciphertext product;
		evaluator.multiply(mat[i], replicated_vec[i], product);

		// Relin
		evaluator.relinearize_inplace(product, relin_keys);

		// Rescale
		evaluator.rescale_to_next_inplace(product);
		product.scale() = pow(2.0, 40);

		evaluator.add_inplace(result, product);
	}
	auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << __func__ << ": " << duration.count() << endl;
	return result;
}

/* Finds the dot product of 2 vectors encoded as ciphertexts. */
Ciphertext dot_prod(Ciphertext a, Ciphertext b, int num_rows, CKKSEncoder &ckks_encoder, Encryptor &encryptor, GaloisKeys galois_keys, RelinKeys relin_keys, Evaluator &evaluator) {
	Ciphertext c;
	evaluator.multiply(a, b, c);
	evaluator.relinearize_inplace(c, relin_keys);
	c.scale() = pow(2, (int)log2(c.scale()));

	for (size_t i = 0; i < (int)log2(num_rows); i++) {
		evaluator.add_inplace(c, rotate_ciphertext(c, pow(2, (int)i), galois_keys, evaluator));
	}

	return c;
}

/* Performs matrix-vector multiplication for a row-packed matrix (not used)*/
Ciphertext rp_mat_vec_mult(vector<Ciphertext> A, Ciphertext b, int num_rows, double scale, CKKSEncoder &ckks_encoder, Encryptor &encryptor, GaloisKeys galois_keys, RelinKeys relin_keys, Evaluator &evaluator)
{
	vector<double> result(num_rows, 0);
	Plaintext pt_result;
	ckks_encoder.encode(result, pow(2.0, 40), pt_result);
	Ciphertext ct_result;
	encryptor.encrypt(pt_result, ct_result);

	for (size_t i = 0; i < num_rows; i++) {
		vector<double> one_vector(num_rows, 0.0);
		for (size_t j; j < i; j++) {
			one_vector[j] = 1;
		}

		Plaintext pt_one_vector;
		Ciphertext ct_one_vector;
		// Encode the values in one_vector into a plaintext
		ckks_encoder.encode(one_vector, scale, pt_one_vector);

		Ciphertext dot_product = dot_prod(A[i], b, num_rows, ckks_encoder, encryptor, galois_keys, relin_keys, evaluator);
		evaluator.multiply_plain_inplace(dot_product, pt_one_vector);
		// Relinearization
		evaluator.relinearize_inplace(dot_product, relin_keys);
		// Rescale
		evaluator.rescale_to_next_inplace(dot_product);
		// Manual rescale
		dot_product.scale() = pow(2.0, 40);

		evaluator.add_inplace(ct_result, dot_product);
	}

	return ct_result;
}

/* Performs matrix multiplication between two column-packed matrices. (not used) */
vector<Ciphertext> cp_mat_mult(vector<Ciphertext> matA, vector<Ciphertext> matB, int n, double scale, CKKSEncoder &ckks_encoder, Encryptor &encryptor, GaloisKeys galois_keys, RelinKeys relin_keys, Evaluator &evaluator, EncryptionParameters params)
{
	vector<Ciphertext> result(n);

	for (size_t i = 0; i < n; i++) {
		result[i] = mat_vec_mult(matA, matB[i], ckks_encoder, encryptor, galois_keys, relin_keys, evaluator, params);
	}

	return result;
}

// Reference: https://github.com/MarwanNour/SEAL-FYP-Logistic-Regression/blob/master/logistic_regression_ckks.cpp
Ciphertext horner_method(Ciphertext ctx, int degree, vector<double> coeffs, CKKSEncoder &ckks_encoder, double scale, Evaluator &evaluator, Encryptor &encryptor, RelinKeys relin_keys, EncryptionParameters params) 
{
	SEALContext context(params);
    auto tmp = make_shared<SEALContext>(context);

    cout << "->" << __func__ << endl;
    cout << "->" << __LINE__ << endl;

    // print_Ciphertext_Info("CTX", ctx, tmp);
    auto start = high_resolution_clock::now();
    int num_coeffs = 0;
    if (degree == 7) {
    	num_coeffs = 5;
    }
    vector<Plaintext> plain_coeffs(num_coeffs);

    // Random Coefficients from 0-1
    cout << "Polynomial = ";
    int counter = 0;
    int index = 0;
    while (index < num_coeffs) {
    	// coeffs[i] = (double)rand() / RAND_MAX;
    	cout << coeffs[index] << endl;
        ckks_encoder.encode(coeffs[index], scale, plain_coeffs[index]);
        cout << "x^" << counter << " * (" << coeffs[index] << ")"
             << ", ";
        counter += counter == 0 ? 1 : 2;
        index++;
    }

    cout << endl;
    // cout << "->" << __LINE__ << endl;

    double one_eighth = 1 / 8;
    Plaintext pt_eighth;
    ckks_encoder.encode(one_eighth, scale, pt_eighth);

    Ciphertext ct_eighth;
    encryptor.encrypt(pt_eighth, ct_eighth);
    // cout << "Equalizing ciphertexts..." << endl;
    equalize_levels(ctx, ct_eighth, evaluator, params);
    // cout << "Multiplying ciphertexts..." << endl;
    evaluator.multiply_inplace(ctx, ct_eighth);
    // cout << "Relinearizing ciphertexts..." << endl;
    evaluator.relinearize_inplace(ctx, relin_keys);
    // cout << "Rescaling ctx..." << endl;
    ctx.scale() = scale;

    // cout << "Encrypting plain_coeffs[num_coeffs]..." << num_coeffs << endl;
    Ciphertext temp;
    encryptor.encrypt(plain_coeffs[num_coeffs - 1], temp);
    // print_Ciphertext_Info("tmep", temp, tmp);

    Plaintext plain_result;
    vector<double> result;
    // cout << "->" << __LINE__ << endl;

    // cout << "Running horner loop..." << endl;
    for (int i = num_coeffs - 1; i >= 0; i--)
    {
    	// cout << "ITERATION " << i << endl;
        int ctx_level = tmp->get_context_data(ctx.parms_id())->chain_index();
        int temp_level = tmp->get_context_data(temp.parms_id())->chain_index();
        if (ctx_level > temp_level)
        {
            evaluator.mod_switch_to_inplace(ctx, temp.parms_id());
        }
        else if (ctx_level < temp_level)
        {
            evaluator.mod_switch_to_inplace(temp, ctx.parms_id());
        }
        Ciphertext res;
        evaluator.multiply(temp, ctx, res);
        // cout << "->" << __LINE__ << endl;

        evaluator.relinearize_inplace(res, relin_keys);

        evaluator.rescale_to_next_inplace(res);
        // cout << "->" << __LINE__ << endl;

        evaluator.mod_switch_to_inplace(plain_coeffs[i], res.parms_id());

        // Manual rescale
        res.scale() = pow(2.0, 40);
        // cout << "->" << __LINE__ << endl;

        evaluator.add_plain_inplace(res, plain_coeffs[i]);
        temp = res;
    }
    // cout << "->" << __LINE__ << endl;

    // print_Ciphertext_Info("temp", temp, tmp);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << __func__ << ": " << duration.count() << endl;
    return temp;
}

/* Finds the hessian approximation for an encrypted matrix */
vector<Ciphertext> hessian_approx(vector<Ciphertext> ctx, double scale, CKKSEncoder &ckks_encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys relin_keys, EncryptionParameters params) 
{
    auto start = high_resolution_clock::now();
    double coeff = 4;
	Plaintext pt_coeff;
	ckks_encoder.encode(coeff, scale, pt_coeff);

	Ciphertext ct_coeff;
	encryptor.encrypt(pt_coeff, ct_coeff);


	int num_rows = ctx.size();
	vector<Ciphertext> result(num_rows);
	for (size_t i = 0; i < num_rows; i++) {
		evaluator.multiply(ctx[i], ct_coeff, result[i]);

		evaluator.relinearize_inplace(result[i], relin_keys);

		result[i].scale() = pow(2.0, (int)log2(result[i].scale()));
		result[i] = ct_coeff;
	}
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	cout << __func__ << ": " << duration.count() << endl;

	return result;
}

/* Trains a logistic regression model */
Ciphertext train_model(vector<Ciphertext> X, Ciphertext weights, Ciphertext y, vector<Ciphertext> xtxi, vector<Ciphertext> xt, CKKSEncoder &ckks_encoder, Encryptor &encryptor, Evaluator &evaluator, GaloisKeys galois_keys, RelinKeys relin_keys, EncryptionParameters params)
{
	SEALContext context(params);
    auto tmp = make_shared<SEALContext>(context);

    cout << "->" << __func__ << endl;
    cout << "->" << __LINE__ << endl;

    double scale = pow(2.0, 40);
	vector<double> coeffs = {0.5, 1.73496, -4.19407, 5.43402, -2.50739};
	Ciphertext curr_weights = weights;
	for (size_t k = 0; k < ITERS; k++) {
		Ciphertext mat_vec_prod = mat_vec_mult(X, curr_weights, ckks_encoder, encryptor, galois_keys, relin_keys, evaluator, params);
		
		Ciphertext p = horner_method(mat_vec_prod, 7, coeffs, ckks_encoder, scale, evaluator, encryptor, relin_keys, params);
		print_Ciphertext_Info("p", p, tmp);
		print_Ciphertext_Info("y", y, tmp);
		
		Ciphertext y_minus_p;
		equalize_levels(y, p, evaluator, params);
		evaluator.sub(y, p, y_minus_p);

		Ciphertext g = mat_vec_mult(xt, y_minus_p, ckks_encoder, encryptor, galois_keys, relin_keys, evaluator, params);
		
		vector<Ciphertext> H = hessian_approx(xtxi, scale, ckks_encoder, encryptor, evaluator, relin_keys, params);
		print_Ciphertext_Info("H", H[0], tmp);
		print_Ciphertext_Info("g", g, tmp);
		print_Ciphertext_Info("curr_weights", curr_weights, tmp);

		Ciphertext hg = mat_vec_mult(H, g, ckks_encoder, encryptor, galois_keys, relin_keys, evaluator, params);

		equalize_levels(curr_weights, hg, evaluator, params);

		Ciphertext new_weights;
		evaluator.sub(curr_weights, hg, new_weights);

		curr_weights = new_weights;

	}

	return curr_weights;
}

int main() {
	/*
	We start by setting up the CKKS scheme.
	*/
	EncryptionParameters parms(scheme_type::ckks);

	// Set coefficient modulus and polynomial modulus degree
	// Bit count for coeff_modulus must be below bound for poly_modulus_degree
	size_t poly_modulus_degree = 32768;
	cout << "Bound for poly_modulus_degree of size " << poly_modulus_degree << ": " << CoeffModulus::MaxBitCount(poly_modulus_degree) << endl;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60}));

	// Set scale
    double scale = pow(2.0, 40);

	// Creat context and generate keys
	SEALContext context(parms);
    auto tmp = make_shared<SEALContext>(context);

    // Generate keys, encryptor, decryptor and evaluator
    KeyGenerator keygen(context);
    SecretKey sk = keygen.secret_key();
    PublicKey pk;
    keygen.create_public_key(pk);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    Encryptor encryptor(context, pk);
    Evaluator evaluator(context);
    Decryptor decryptor(context, sk);

    // Create CKKS encoder
    CKKSEncoder ckks_encoder(context);

    //--------- Prep Data ---------//
    cout << "Prepping data..." << endl;
	// Read in SNP data from a file. Store as 2-dimensional vector of strings. 
	string file_name = "test_data.csv";
	vector<vector<string>> s_data = csv_to_matrix(file_name);

	// Turn into 2-dimensional vector of doubles
	cout << "Converting string matrix to double matrix..." << endl;
	vector<vector<double>> d_data = string_to_double(s_data);
	int rows = d_data.size();
	int cols = d_data[0].size();
	mat A(rows, cols);
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = j; j < cols; j++) {
			A(i, j) = d_data[i][j];
		}
	}
	cout << "Transposing double matrix..." << endl;
	// Transpose the data because it needs to be encrypt as a CP matrix
	vector<vector<double>> d_data_tranpsose = transpose_matrix(d_data);
	cout << "Transposing double matrix using armadillo..." << endl;
	mat A_t = A.t();
	cout << "Finding A_t * A using armadillo..." << endl;

	// vector<vector<double>> d_data_transpose_inv = pinv(d_data_tranpsose, 0.1);

	mat xtx = A_t * A;
	cout << "Finding A_t_A inverse using armadillo..." << endl;
	mat xtxi = pinv(xtx);
	cout << "Finding A_t_A inverse using armadillo..." << endl;

	typedef vector<double> stdvec;
	typedef vector<vector<double>> stdvecvec;

	// stdvecvec xtxi_t = conv_to<stdvecvec>::from(xtxi.t());
	stdvecvec xtxi_(rows);
	stdvecvec xt(rows);
	for (size_t i = 0; i < rows; i++) {
		xtxi_[i] = conv_to<stdvec>::from(xtxi.row(i));  
		xt[i] = conv_to<stdvec>::from(A_t.row(i));
	}
	cout << "Generating weights and outcomes vectors..." << endl;

	// Weights 
	vector<double> weights(cols, 0);

	// Generate outcomes vector
	vector<double> outcomes(rows);
	for (size_t i = 0; i < rows; i++) {
		outcomes[i] = (double)(rand() % 2);
	} 

	//--------- Encode Data ---------//
	cout << "Encoding data..." << endl;
	// Encode the data using scale
	vector<Plaintext> pt_data(rows);
	vector<Plaintext> pt_xtxi(rows);
	vector<Plaintext> pt_xt(rows);
	for (size_t i = 0; i < rows; i++) {
		ckks_encoder.encode(d_data[i], scale, pt_data[i]);
		ckks_encoder.encode(xtxi_[i], scale, pt_xtxi[i]);
		ckks_encoder.encode(xt[i], scale, pt_xt[i]);
	}

	Plaintext pt_weights; 
	ckks_encoder.encode(weights, scale, pt_weights);

	Plaintext pt_outcomes;
	ckks_encoder.encode(outcomes, scale, pt_outcomes);

	//--------- Encrypt Data ---------//
	cout << "Encrypting data..." << endl;
	// Encrypt the data
	vector<Ciphertext> ct_data(rows);
	vector<Ciphertext> ct_xtxi(rows);
	vector<Ciphertext> ct_xt(rows);
	for (size_t i = 0; i < rows; i++) {
		encryptor.encrypt(pt_data[i], ct_data[i]);
		encryptor.encrypt(pt_xtxi[i], ct_xtxi[i]);
		encryptor.encrypt(pt_xt[i], ct_xt[i]);
	}

	Ciphertext ct_weights;
	encryptor.encrypt(pt_weights, ct_weights);

	Ciphertext ct_outcomes;
	encryptor.encrypt(pt_outcomes, ct_outcomes);

	// Check scales
	// cout << "Checking scales..." << endl;
	// cout << "|________ Starting Scales ________|" << endl;
	// cout << "data: " << ct_data[0].scale() << endl;
	// cout << "weights: " << ct_weights.scale() << endl;
	// cout << "outcomes: " << ct_outcomes.scale() << endl;

	cout << "Beginning model training..." << endl;
	auto start = high_resolution_clock::now();
	Ciphertext new_weights = train_model(ct_data, ct_weights, ct_outcomes, ct_xtxi, ct_xt, ckks_encoder, encryptor, evaluator, gal_keys, relin_keys, parms);
	auto end = high_resolution_clock::now();

	ms_duration duration = duration_cast<microseconds>(end - start);
	cout << "Runtime: " << duration.count() << endl;
}