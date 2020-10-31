# Locality Sensitive Hashing
# This methid help reduce the computational cost of finding the k-nearset neighbours
import numpy as np

def vector_side_plane(P,v):
    dotProduct = np.dot(P,v.T)
    signOfDotProduct = np.sign(dotProduct)
    return np.asscalar(signOfDotProduct)
