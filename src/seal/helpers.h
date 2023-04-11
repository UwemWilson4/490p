#include "seal/seal.h"

#include <iostream>
#include <iomanip>
#include <fstream>

/* Helper functions taken from the examples.h file of native/examples in SEAL */

using namespace std;
using namespace seal;

/*
Helper function: Prints the name of the example in a fancy banner.
*/
inline void print_example_banner(std::string title)
{
    if (!title.empty())
    {
        std::size_t title_length = title.length();
        std::size_t banner_length = title_length + 2 * 10;
        std::string banner_top = "+" + std::string(banner_length - 2, '-') + "+";
        std::string banner_middle = "|" + std::string(9, ' ') + title + std::string(9, ' ') + "|";

        std::cout << std::endl << banner_top << std::endl << banner_middle << std::endl << banner_top << std::endl;
    }
}

inline void print_parameters(const seal::SEALContext &context)
{
    auto &context_data = *context.key_context_data();

    /*
    Which scheme are we using?
    */
    std::string scheme_name;
    switch (context_data.parms().scheme())
    {
    case seal::scheme_type::bfv:
        scheme_name = "BFV";
        break;
    case seal::scheme_type::ckks:
        scheme_name = "CKKS";
        break;
    default:
        throw std::invalid_argument("unsupported scheme");
    }
    std::cout << "/" << std::endl;
    std::cout << "| Encryption parameters :" << std::endl;
    std::cout << "|   scheme: " << scheme_name << std::endl;
    std::cout << "|   poly_modulus_degree: " << context_data.parms().poly_modulus_degree() << std::endl;

    /*
    Print the size of the true (product) coefficient modulus.
    */
    std::cout << "|   coeff_modulus size: ";
    std::cout << context_data.total_coeff_modulus_bit_count() << " (";
    auto coeff_modulus = context_data.parms().coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++)
    {
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }
    std::cout << coeff_modulus.back().bit_count();
    std::cout << ") bits" << std::endl;

    /*
    For the BFV scheme print the plain_modulus parameter.
    */
    if (context_data.parms().scheme() == seal::scheme_type::bfv)
    {
        std::cout << "|   plain_modulus: " << context_data.parms().plain_modulus().value() << std::endl;
    }

    std::cout << "\\" << std::endl;
}

/* Reads a CSV file into a matrix line-by-line */
vector<vector<string>> csv_to_matrix(string file_name)
{
	vector<vector<string>> result;
	string line, value;

	ifstream csv_file(file_name);
	if (csv_file.is_open())
	{
		while (getline(csv_file, line))
		{
			stringstream cur_line(line);
			vector<string> parsed_row;
			while (getline(cur_line, value, ","))
			{
				parsed_row.push_back(value);
			}
			result.push_back(parsed_row);
		}
	}

	return result;
}

/* Turn a matrix of string doubles into a matrix of doubles */
vector<vector<double>> string_to_double(vector<vector<string>> s_matrix)
{
	int n = s_matrix.size();
	int m = s_matrix[0].size();

	vector<vector<double>> result(n, vector<double> (m));

	for (size_t i = 0; i < n; i++) {
		for (size_t j = 0; j < m; j++) {
			result[i][j] = stod(s_matrix[i][j]);
		}
	}

	return result;
}