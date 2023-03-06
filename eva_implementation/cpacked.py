from eva import EvaProgram, Input, Output

# Create a class for a column-packed matrix encoder. 
class CPEncoder():
	"""Class for encoding matrices in a column-packed manner"""
	def __init__(self, original_matrix):
		self.original_matrix = original_matrix

	
	"""Creates a column-packed matrix by encrypting every column of the matrix.
	   This assumes the rows represent the SNPs and the columns represent the
	   samples."""
	def encodeMatrix(self):
		# Specify a program for creating a column-packed matrix


		# Set fixed-point scale and maximum range of coefficients. Currently not sure
		# what these should ideally be set to, but refer to the GitHub for example values.

		# Compile the program

		# Optionally print a visualization

		# Generate encryption keys and encrypt inputs using specified program


	"""Encrypt a column of a matrix, turning it into a ciphertext."""
	def encryptCol(self, col):

	
	"""This function is under construction. It will be converted to take a ciphertext
	   as input and perform the replacte operation on it."""
	def replicate(self):	
		# Specify a program for creating a column-packed matrix
		enc = EvaProgram('enc', len(col))
		with enc:
			samples = Input('samples')
			
			Output('samples', samples*2)

		# Set fixed-point scale and maximum range of coefficients. Currently not sure
		# what these should ideally be set to, but refer to the GitHub for example values.

		# Compile the program

		# Optionally print a visualization

		# Generate encryption keys and encrypt inputs using specified program