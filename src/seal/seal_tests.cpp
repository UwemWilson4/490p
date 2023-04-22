#include <chrono>
#include "seal/seal.h"

#define ITERS 128
#define NUM_TRIALS 5

using namespace std;
using namespace std::chrono;
using namespace seal;

typedef std::chrono::duration<long, std::ratio<1, 1000000> > ms_duration;
void test_simple_addition(vector<ms_duration> &results_vec, vector<double> rand_vals, double scale, CKKSEncoder &ckks_encoder, Encryptor &encryptor, Evaluator &evaluator) {
	double a = 0;
    double result = 0;
    Plaintext pt_a;
    Plaintext pt_result;
    ckks_encoder.encode(a, scale, pt_a);
    ckks_encoder.encode(result, scale, pt_result);

    Ciphertext ct_result;
    encryptor.encrypt(pt_result, ct_result);
    for (size_t trial = 0; trial < NUM_TRIALS; trial++) {
    	auto start = high_resolution_clock::now();
    	for (size_t i = 0; i < ITERS; i++) {
	    	Plaintext pt_val;
	    	ckks_encoder.encode(rand_vals[i], scale, pt_val);
	    	evaluator.add_plain_inplace(ct_result, pt_val);
	    }
	    auto end = high_resolution_clock::now();
	    auto duration = duration_cast<microseconds>(end - start);
	    cout << duration.count() << endl;
	    results_vec[trial] = duration_cast<microseconds>(end - start);
    }
}

void test_vector_addition(vector<ms_duration> &results_vec, vector<double> rand_vals, double scale, CKKSEncoder &ckks_encoder, Encryptor &encryptor, Evaluator &evaluator) {
	vector<Plaintext> pt_test_vecs(ITERS);
	vector<Ciphertext> ct_test_vecs(ITERS);
	vector<vector<double>> test_vecs(ITERS, vector<double> (ITERS));
	for (size_t vec = 0; vec < ITERS; vec++) {
		for (size_t val = 0; val < ITERS; val++) {
			test_vecs[vec][val] = (double)(rand() % 101);
		}
		ckks_encoder.encode(test_vecs[vec], scale, pt_test_vecs[vec]);
		encryptor.encrypt(pt_test_vecs[vec], ct_test_vecs[vec]);
	}

    vector<double> result(ITERS, 0.0);

    Plaintext pt_result;
    Plaintext pt_rand_vals;
    ckks_encoder.encode(rand_vals, scale, pt_rand_vals);
    ckks_encoder.encode(result, scale, pt_result);

    Ciphertext ct_result;
    Ciphertext ct_rand_vals;
    encryptor.encrypt(pt_rand_vals, ct_rand_vals);
    encryptor.encrypt(pt_result, ct_result);
    for (size_t trial = 0; trial < NUM_TRIALS; trial++) {
    	auto start = high_resolution_clock::now();
    	for (size_t i = 0; i < ITERS; i++) {
	    	evaluator.add(ct_rand_vals, ct_test_vecs[i], ct_result);
	    }
	    auto end = high_resolution_clock::now();
	    auto duration = duration_cast<microseconds>(end - start);
	    cout << duration.count() << endl;
	    results_vec[trial] = duration_cast<microseconds>(end - start);
    }
}

void test_simple_multiplication(vector<ms_duration> &results_vec, vector<double> rand_vals, double scale, CKKSEncoder &ckks_encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys relin_keys) {
	double a = (double)((rand() % 101) + 1);
    double result = 0;
    Plaintext pt_a;
    Plaintext pt_result;
    ckks_encoder.encode(a, scale, pt_a);
    ckks_encoder.encode(result, scale, pt_result);

    Ciphertext ct_a;
    encryptor.encrypt(pt_a, ct_a);
    for (size_t trial = 0; trial < NUM_TRIALS; trial++) {
    	auto start = high_resolution_clock::now();
    	for (size_t i = 0; i < ITERS; i++) {
	    	Plaintext pt_val;
	    	ckks_encoder.encode(rand_vals[i], scale, pt_val);

    		Ciphertext ct_result;
    		encryptor.encrypt(pt_result, ct_result);

	    	evaluator.multiply_plain(ct_a, pt_val, ct_result);

	    	evaluator.relinearize_inplace(ct_result, relin_keys);

	    	evaluator.rescale_to_next_inplace(ct_result);

	    	ct_result.scale() = pow(2.0, (int)log2(ct_result.scale()));
	    }
	    auto end = high_resolution_clock::now();
	    auto duration = duration_cast<microseconds>(end - start);
	    cout << duration.count() << endl;
	    results_vec[trial] = duration_cast<microseconds>(end - start);
    }
}

void test_vector_multiplication(vector<ms_duration> &results_vec, vector<double> rand_vals, double scale, CKKSEncoder &ckks_encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys relin_keys) {
	vector<Plaintext> pt_test_vecs(ITERS);
	vector<Ciphertext> ct_test_vecs(ITERS);
	vector<vector<double>> test_vecs(ITERS, vector<double> (ITERS));
	for (size_t vec = 0; vec < ITERS; vec++) {
		for (size_t val = 0; val < ITERS; val++) {
			test_vecs[vec][val] = (double)(rand() % 101);
		}
		ckks_encoder.encode(test_vecs[vec], scale, pt_test_vecs[vec]);
		encryptor.encrypt(pt_test_vecs[vec], ct_test_vecs[vec]);
	}

    vector<double> result(ITERS, 0.0);

    Plaintext pt_result;
    Plaintext pt_rand_vals;
    ckks_encoder.encode(rand_vals, scale, pt_rand_vals);
    ckks_encoder.encode(result, scale, pt_result);

    Ciphertext ct_result;
    Ciphertext ct_rand_vals;
    encryptor.encrypt(pt_rand_vals, ct_rand_vals);
    encryptor.encrypt(pt_result, ct_result);
    for (size_t trial = 0; trial < NUM_TRIALS; trial++) {
    	auto start = high_resolution_clock::now();
    	for (size_t i = 0; i < ITERS; i++) {
	    	evaluator.multiply(ct_rand_vals, ct_test_vecs[i], ct_result);

	    	evaluator.relinearize_inplace(ct_result, relin_keys);

	    	evaluator.rescale_to_next_inplace(ct_result);
	    }
	    auto end = high_resolution_clock::now();
	    auto duration = duration_cast<microseconds>(end - start);
	    cout << duration.count() << endl;
	    results_vec[trial] = duration_cast<microseconds>(end - start);
    }
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
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 60}));

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

    vector<double> rand_vals(ITERS);
    for (size_t i = 0; i < ITERS; i++) {
    	double val = (double)(rand() % 101);
    	rand_vals[i] = val;
    }
    vector<ms_duration> simple_additions(NUM_TRIALS);
    vector<ms_duration> vector_additions(NUM_TRIALS);
    vector<ms_duration> simple_multiplications(NUM_TRIALS);
    vector<ms_duration> vector_multiplications(NUM_TRIALS);

    //--------- Simple Additions ---------//
    cout << "|________ Simple Addition ________|" << endl;
    test_simple_addition(simple_additions, rand_vals, scale, ckks_encoder, encryptor, evaluator);
    cout << "{ ";
    for (size_t i = 0; i < simple_additions.size(); i++) {
    	string line_end = (i < simple_additions.size() - 1 ) ? ", " : " ";
    	cout << simple_additions[i].count() << line_end;
    }
    cout << "}" << endl;
    cout << endl;

    //--------- Vector Additions ---------//
    cout << "|________ Vector Addition ________|" << endl;
    test_vector_addition(vector_additions, rand_vals, scale, ckks_encoder, encryptor, evaluator);
    cout << "{ ";
    for (size_t i = 0; i < vector_additions.size(); i++) {
    	cout << vector_additions[i].count();
    	if (i < vector_additions.size() - 1) {
    		cout << ", ";
    	} else {
    		cout << " ";
    	}
    }
    cout << "}" << endl;
    cout << endl;

    //--------- Simple Multiplications ---------//
    cout << "|________ Simple Multiplication ________|" << endl;
    test_simple_multiplication(simple_multiplications, rand_vals, scale, ckks_encoder, encryptor, evaluator, relin_keys);
    cout << "{ ";
    for (size_t i = 0; i < simple_multiplications.size(); i++) {
    	string line_end = (i < simple_multiplications.size() - 1 ) ? ", " : " ";
    	cout << simple_multiplications[i].count() << line_end;
    }
    cout << "}" << endl;
    cout << endl;

    //--------- Vector Multiplications ---------//
    cout << "|________ Vector Multiplication ________|" << endl;
    test_vector_multiplication(vector_multiplications, rand_vals, scale, ckks_encoder, encryptor, evaluator, relin_keys);
    cout << "{ ";
    for (size_t i = 0; i < vector_multiplications.size(); i++) {
    	string line_end = (i < vector_multiplications.size() - 1 ) ? ", " : " ";
    	cout << vector_multiplications[i].count() << line_end;
    }
    cout << "}" << endl;
    cout << endl;
}