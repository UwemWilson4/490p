#include <iostream>
#include <vector>
#include <armadillo>
#include <stdlib.h>

#include "log_reg.h"
#include "helpers.h"

#define ITERS 10


using namespace std;
using namespace arma;
using namespace seal;

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

/* Rotates a Ciphertext by k */
Ciphertext rotate_ciphertext(Ciphertext ctx, int k, GaloisKeys galois_keys, Evaluator &evaluator)
{
	Ciphertext rotated;
	evaluator.rotate_vector(ctx, k, galois_keys, rotated);

	return rotated;
}

vector<Ciphertext> replicate(Ciphertext ctx, int n, double scale, CKKSEncoder &ckks_encoder, Encryptor &encryptor, GaloisKeys galois_keys, RelinKeys relin_keys, Evaluator &evaluator)
{
	vector<Ciphertext> replicate_res(n);

	for (size_t i = 0; i < n; i++) {
		vector<double> one_vector(n, 0.0);
		for (size_t j; j < i; i++) {
			one_vector[j] = 1;
		}

		Plaintext pt_one_vector;
		Ciphertext ct_one_vector;
		// Encode the values in one_vector into a plaintext
		ckks_encoder.encode(one_vector, scale, pt_one_vector);

		// Converts every slot in the ciphertext to 0 except for the ith index
		Ciphertext temp_ciphertext;
		evaluator.multiply_plain(ctx, pt_one_vector, temp_ciphertext);
		// Relinearization
		evaluator.relinearize_inplace(temp_ciphertext, relin_keys);
		// Rescale
		evaluator.rescale_to_next_inplace(temp_ciphertext);
		// Manual rescale
		temp_ciphertext.scale() = pow(2.0, 40);

		for (size_t j = 0; i < (int)log2(n); i++) {
			// NOTE: may have to do a modulus switch here
			Ciphertext temp_rotated = rotate_ciphertext(temp_ciphertext, pow(2, j), galois_keys, evaluator);
			evaluator.add_inplace(temp_ciphertext, temp_rotated);
		}

		replicate_res[i] = temp_ciphertext;
	}

	return replicate_res;
}

/* Performs matrix-vector multiplication. */
Ciphertext mat_vec_mult(vector<Ciphertext> mat, Ciphertext vec, CKKSEncoder &ckks_encoder, Encryptor &encryptor, GaloisKeys galois_keys, RelinKeys relin_keys, Evaluator &evaluator) {
	vector<Ciphertext> replicated_vec = replicate(vec, mat.size(), pow(2, 40), ckks_encoder, encryptor, galois_keys, relin_keys, evaluator);
	
	Ciphertext result;
	evaluator.multiply(mat[0], replicated_vec[0], result);
	// Relin
	evaluator.relinearize_inplace(result, relin_keys);
	// Rescale
	evaluator.rescale_to_next_inplace(result);
	result.scale() = pow(2.0, 40);
	for (size_t i = 1; i < replicated_vec.size(); i++) {
		Ciphertext product;
		evaluator.multiply(mat[i], replicated_vec[i], product);
		// Relin
		evaluator.relinearize_inplace(product, relin_keys);
		// Rescale
		evaluator.rescale_to_next_inplace(product);
		product.scale() = pow(2.0, 40);

		evaluator.add_inplace(result, product);
	}

	return result;
}

/* Performs matrix multiplication between two column-packed matrices. */
vector<Ciphertext> cp_mat_mult(vector<Ciphertext> matA, vector<Ciphertext> matB, int n, double scale, CKKSEncoder &ckks_encoder, Encryptor &encryptor, GaloisKeys galois_keys, RelinKeys relin_keys, Evaluator &evaluator)
{
	vector<Ciphertext> result(n);

	for (size_t i = 0; i < n; i++) {
		result[i] = mat_vec_mult(matA, matB[i], ckks_encoder, encryptor, galois_keys, relin_keys, evaluator);
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

    print_Ciphertext_Info("CTX", ctx, tmp);

    vector<Plaintext> plain_coeffs(degree + 1);

    // Random Coefficients from 0-1
    cout << "Polynomial = ";
    int counter = 0;
    for (size_t i = 0; i < degree + 1; i++)
    {
        // coeffs[i] = (double)rand() / RAND_MAX;
        ckks_encoder.encode(coeffs[i], scale, plain_coeffs[i]);
        cout << "x^" << counter << " * (" << coeffs[i] << ")"
             << ", ";
        counter += 2;
    }
    cout << endl;
    // cout << "->" << __LINE__ << endl;

    double one_eigth = 1 / 8;
    Plaintext pt_eighth;
    ckks_encoder.encode(one_eighth, scale, pt_eigth);

    evaluator.multiply_plain_inplace(ctx, pt_eighth);
    evaluator.relinearize_inplace(ctx, relin_keys);
    ctx.scale() = scale;

    Ciphertext temp;
    encryptor.encrypt(plain_coeffs[degree], temp);

    Plaintext plain_result;
    vector<double> result;
    // cout << "->" << __LINE__ << endl;

    for (int i = degree - 1; i >= 0; i--)
    {
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
        evaluator.multiply_inplace(temp, ctx);
        // cout << "->" << __LINE__ << endl;

        evaluator.relinearize_inplace(temp, relin_keys);

        evaluator.rescale_to_next_inplace(temp);
        // cout << "->" << __LINE__ << endl;

        evaluator.mod_switch_to_inplace(plain_coeffs[i], temp.parms_id());

        // Manual rescale
        temp.scale() = pow(2.0, 40);
        // cout << "->" << __LINE__ << endl;

        evaluator.add_plain_inplace(temp, plain_coeffs[i]);
    }
    // cout << "->" << __LINE__ << endl;

    print_Ciphertext_Info("temp", temp, tmp);

    return temp;
}

vector<Ciphertext> train_model(vector<Ciphertext> X, Ciphertext weights, vector<Ciphertext> xtxi, CKKSEncoder &ckks_encoder, Encryptor &encryptor, GaloisKeys galois_keys, RelinKeys relin_keys, EncryptionParameters params)
{
	vector<double> coeffs = {0.5, 1.73496, -4.19407, 5.43402, -2.50739};
	for (size_t k = 0; k < ITERS; k++) {
		Ciphertext mat_vec_prod = mat_vec_mult(X, weigths, ckks_encoder, enryptor, galois_keys, relin_keys, evaluator);
	}
}

int main() {
	cout << "Test\n" << endl;
	/*
	We start by setting up the CKKS scheme.
	*/
	EncryptionParameters parms(scheme_type::ckks);

	// Set coefficient modulus and polynomial modulus degree
	// Bit count for coeff_modulus must be below bound for poly_modulus_degree
	size_t poly_modulus_degree = 16384;
	cout << "Bound for poly_modulus_degree of size" << poly_modulus_degree << ": " << CoeffModulus::MaxBitCount(poly_modulus_degree) << endl;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 60}));

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
	cout << "CSV successfully read in" << endl;

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
	cout << "Successfully converted string matrix to double matrix" << endl;
	cout << "Transposing double matrix..." << endl;
	// Transpose the data because it needs to be encrypt as a CP matrix
	vector<vector<double>> d_data_tranpsose = transpose_matrix(d_data);
	cout << "Transposed double matrix with transpose_matrix()" << endl;
	cout << "Transposing double matrix using armadillo..." << endl;
	mat A_t = A.t();
	cout << "Transposed double matrix with armadillo" << endl;
	cout << "Finding A_t * A using armadillo..." << endl;

	// vector<vector<double>> d_data_transpose_inv = pinv(d_data_tranpsose, 0.1);

	mat xtx = A_t * A;
	cout << "Found A_t * A with armadillo" << endl;
	cout << "Finding A_t_A inverse using armadillo..." << endl;
	mat xtxi = pinv(xtx);
	cout << "Found A_t * A with armadillo" << endl;
	cout << "Finding A_t_A inverse using armadillo..." << endl;

	typedef vector<vector<double>> stdvecvec;

	// stdvecvec xtxi_t = conv_to<stdvecvec>::from(xtxi.t()); 
	cout << "Found inverse with armadillo" << endl;
	cout << "Generating weights and outcomes vectors..." << endl;

	// Weights 
	vector<double> weights(cols, 0);

	// Generate outcomes vector
	vector<double> outcomes(rows);
	for (size_t i = 0; i < rows; i++) {
		outcomes[i] = (double)(rand() % 2);
	} 

	//--------- Encode Data ---------//
	cout << "Found weights and outcomes vectors" << endl;
	cout << "Encoding data..." << endl;
	// Encode the data using scale
	vector<Plaintext> pt_data(rows);
	for (size_t i = 0; i < rows; i++) {
		ckks_encoder.encode(d_data[i], scale, pt_data[i]);
	}

	Plaintext pt_weights; 
	ckks_encoder.encode(weights, scale, pt_weights);

	Plaintext pt_outcomes;
	ckks_encoder.encode(outcomes, scale, pt_outcomes);
	cout << "Finished encoding data" << endl;

	//--------- Encrypt Data ---------//
	cout << "Encrypting data..." << endl;
	// Encrypt the data
	vector<Ciphertext> ct_data(rows);
	for (size_t i = 0; i < rows; i++) {
		encryptor.encrypt(pt_data[i], ct_data[i]);
	}

	Ciphertext ct_weights;
	encryptor.encrypt(pt_weights, ct_weights);

	Ciphertext ct_outcomes;
	encryptor.encrypt(pt_outcomes, ct_outcomes);
	cout << "Finished encrypting data" << endl;

	// Check scales
	cout << "Checking scales..." << endl;
	cout << "|________ Starting Scales ________|" << endl;
	cout << "data: " << ct_data[0].scale() << endl;
	cout << "weights: " << ct_weights.scale() << endl;
	cout << "outcomes: " << ct_outcomes.scale() << endl;

}