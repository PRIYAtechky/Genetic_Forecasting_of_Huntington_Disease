import numpy as np

def dna_sequence_to_matrix(sequence, matrix_size=32):
    mapping = {'A': 0.25, 'C': 0.5, 'G': 0.75, 'T': 1.0}
    encoded = [mapping.get(base, 0) for base in sequence.upper()]
    
    padded = encoded + [0] * (matrix_size * matrix_size - len(encoded))
    matrix = np.array(padded[:matrix_size * matrix_size]).reshape(matrix_size, matrix_size)
    return matrix
