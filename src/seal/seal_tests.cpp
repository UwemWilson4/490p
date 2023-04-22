#include <chrono>
#include "seal/seal.h"

#define ITERS 100
#define NUM_TRIALS 5

using namespace std;
using namespace std::chrono;
using namespace seal;

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

    vector<std::chrono::duration<long, std::ratio<1, 1000000> >> additions(NUM_TRIALS);

    //--------- Additions ---------//
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
	    	double val = (double)(rand() % 101);
	    	Plaintext pt_val;
	    	ckks_encoder.encode(val, scale, pt_val);
	    	evaluator.add_plain_inplace(ct_result, pt_val);
	    }
	    auto end = high_resolution_clock::now();
	    auto duration = duration_cast<microseconds>(end - start);
	    cout << duration.count() << endl;
	    additions[trial] = duration_cast<microseconds>(end - start);
    }

    cout << "{ ";
    for (size_t i = 0; i < additions.size(); i++) {
    	cout << additions[i].count() << ", ";
    }
    cout << "}" << endl;
}