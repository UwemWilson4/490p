#include <iostream>
#include "log_reg.h"
#include "helper.h"


using namespace std;
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
	evaluator.rotate(ctx, k, galois_keys, rotated);
	
	return rotated;
}

Ciphertext horner_method(Ciphertext ctx, int degree, vector<double> coeffs, CKKSEncoder &ckks_encoder, double scale, Evaluator &evaluator, Encryptor &encryptor, RelinKeys relin_keys, EncryptionParameters params) 
{
	SEALContext context(params);
    auto tmp = make_shared<SEALContext>(context);

    cout << "->" << __func__ << endl;
    cout << "->" << __LINE__ << endl;

    print_Ciphertext_Info("CTX", ctx, tmp);

    vector<Plaintext> plain_coeffs(degree + 1);

    // Encode coefficients into plaintexts
    /**
     * NOTE: This code may assume that the polynomial goes smallest degree
     * to largest to degree from left to right. 
     * */
    cout << "Polynomial = ";
    int counter = 0;
    for (size_t i = 0; i < degree + 1; i++)
    {
        // coeffs[i] = (double)rand() / RAND_MAX;
        ckks_encoder.encode(coeffs[i], scale, plain_coeffs[i]);
        cout << "x^" << counter << " * (" << coeffs[i] << ")"
             << ", ";
        counter++;
    }
    cout << endl;
    // cout << "->" << __LINE__ << endl;

    Ciphertext temp;
    encryptor.encrypt(plain_coeffs[0], temp);

    Plaintext plain_result;
    vector<double> result;
    // cout << "->" << __LINE__ << endl;

    for (int i = 1; i < degree + 1; i++)
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

        // Manual rescale to match plain_coeffs[i] scale
        temp.scale() = pow(2.0, 40);
        // cout << "->" << __LINE__ << endl;

        evaluator.add_plain_inplace(temp, plain_coeffs[i]);
    }
    // cout << "->" << __LINE__ << endl;

    print_Ciphertext_Info("temp", temp, tmp);

    return temp;
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
	SEALContext context(params);
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

    // TODO: Create function to print parameters

	// Read in SNP data from a file. Store as 2-dimensional vector of strings. 
	// Turn into 2-dimensional vector of doubles
	

	// 
	// Create 
	// Encode the data using scale

}