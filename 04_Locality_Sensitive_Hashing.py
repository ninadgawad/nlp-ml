# Locality Sensitive Hashing
# This methid help reduce the computational cost of finding the k-nearset neighbours
import numpy as np

def vector_side_plane(P,v):
    dotProduct = np.dot(P,v.T)
    signOfDotProduct = np.sign(dotProduct)
    return np.asscalar(signOfDotProduct)

def hash_plane(P_1,v):
    hash_value = 0 
    for i, P in enumerate(P_1):
        sign = vector_side_plane(P_1)
        hash_i = 1 if sign >= 0 else 0
        hash_value += 2**i * hash_i
    
    return hash_value
