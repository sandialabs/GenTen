## Interface for higher order joint moment tensors

This interface specializes a use case for tensor decompositions. While the main repo andthe interface operates on the raw input data itself as the tensor, this interface does a transformation on the raw data, computing higher-order joint moments/cumulants, and generating a tensor as a result. The resulting (symmetric) tensor can then be factorized using various options, for various purposes. 
