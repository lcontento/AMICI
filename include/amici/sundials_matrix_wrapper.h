#ifndef AMICI_SUNDIALS_MATRIX_WRAPPER_H
#define AMICI_SUNDIALS_MATRIX_WRAPPER_H

#include <sunmatrix/sunmatrix_band.h>   // SUNMatrix_Band
#include <sunmatrix/sunmatrix_dense.h>  // SUNMatrix_Dense
#include <sunmatrix/sunmatrix_sparse.h> // SUNMatrix_Sparse

#include <gsl/gsl-lite.hpp>

#include <vector>

#include "amici/vector.h"

namespace amici {

/**
 * @brief A RAII wrapper for SUNMatrix structs.
 *
 * This can create dense, sparse, or banded matrices using the respective
 * constructor.
 */
class SUNMatrixWrapper {
  public:
    SUNMatrixWrapper() = default;

    /**
     * @brief Create sparse matrix. See SUNSparseMatrix in sunmatrix_sparse.h
     * @param M Number of rows
     * @param N Number of columns
     * @param NNZ Number of nonzeros
     * @param sparsetype Sparse type
     */
    SUNMatrixWrapper(int M, int N, int NNZ, int sparsetype);

    /**
     * @brief Create dense matrix. See SUNDenseMatrix in sunmatrix_dense.h
     * @param M Number of rows
     * @param N Number of columns
     */
    SUNMatrixWrapper(int M, int N);

    /**
     * @brief Create banded matrix. See SUNBandMatrix in sunmatrix_band.h
     * @param M Number of rows and columns
     * @param ubw Upper bandwidth
     * @param lbw Lower bandwidth
     */
    SUNMatrixWrapper(int M, int ubw, int lbw);

    /**
     * @brief Create sparse matrix from dense or banded matrix. See
     * SUNSparseFromDenseMatrix and SUNSparseFromBandMatrix in
     * sunmatrix_sparse.h
     * @param A Wrapper for dense matrix
     * @param droptol tolerance for dropping entries
     * @param sparsetype Sparse type
     */
    SUNMatrixWrapper(const SUNMatrixWrapper &A, realtype droptol,
                     int sparsetype);

    /**
     * @brief Wrap existing SUNMatrix
     * @param mat
     */
    explicit SUNMatrixWrapper(SUNMatrix mat);

    ~SUNMatrixWrapper();

    /**
     * @brief Copy constructor
     * @param other
     */
    SUNMatrixWrapper(const SUNMatrixWrapper &other);

    /**
     * @brief Move constructor
     * @param other
     */
    SUNMatrixWrapper(SUNMatrixWrapper &&other);

    /**
     * @brief Copy assignment
     * @param other
     * @return
     */
    SUNMatrixWrapper &operator=(const SUNMatrixWrapper &other);

    /**
     * @brief Move assignment
     * @param other
     * @return
     */
    SUNMatrixWrapper &operator=(SUNMatrixWrapper &&other);
    
    /**
     * @brief Reallocate space for sparse matrix according to specified nnz
     * @param nnz new number of nonzero entries
     */
    void reallocate(int nnz);
    
    /**
     * @brief Reallocate space for sparse matrix to used space according to last entry in indexptrs
     */
    void realloc();
    

    /**
     * @brief Access raw data
     * @return raw data pointer
     */
    realtype *data() const;

    /**
     * @brief Get the wrapped SUNMatrix
     * @return SlsMat
     */
    SUNMatrix get() const;

    /**
     * @brief Get the number of rows
     * @return number
     */
    sunindextype rows() const;

    /**
     * @brief Get the number of columns
     * @return number
     */
    sunindextype columns() const;

    /**
     * @brief Get the number of specified non-zero elements (sparse matrices only)
     * @note values will be unininitialized before indexptrs are set.
     * @return number
     */
    sunindextype nonzeros() const;
    
    /**
     * @brief Get the number of allocated non-zero elements (sparse matrices only)
     * @return number
     */
    sunindextype nonzero_space() const;

    /**
     * @brief Get the index values of a sparse matrix
     * @return index array
     */
    sunindextype *indexvals() const;

    /**
     * @brief Get the index pointers of a sparse matrix
     * @return index array
     */
    sunindextype *indexptrs() const;

    /**
     * @brief Get the type of sparse matrix
     * @return index array
     */
    int sparsetype() const;

    /**
     * @brief reset data to zeroes
     */
    void reset();

    /**
     * @brief multiply with a scalar (in-place)
     * @param a scalar value to multiply matrix
     */
    void scale(realtype a);

    /**
     * @brief N_Vector interface for multiply
     * @param c output vector, may already contain values
     * @param b multiplication vector
     */
    void multiply(N_Vector c, const_N_Vector b) const;

    /**
     * @brief Perform matrix vector multiplication c += A*b
     * @param c output vector, may already contain values
     * @param b multiplication vector
     */
    void multiply(gsl::span<realtype> c, gsl::span<const realtype> b) const;

    /**
     * @brief Perform reordered matrix vector multiplication c += A[:,cols]*b
     * @param c output vector, may already contain values
     * @param b multiplication vector
     * @param cols int vector for column reordering
     * @param transpose bool transpose A before multiplication
     */
    void multiply(N_Vector c,
                  const N_Vector b,
                  gsl::span <const int> cols,
                  bool transpose) const;

    /**
     * @brief Perform reordered matrix vector multiplication c += A[:,cols]*b
     * @param c output vector, may already contain values
     * @param b multiplication vector
     * @param cols int vector for column reordering
     * @param transpose bool transpose A before multiplication
     */
    void multiply(gsl::span<realtype> c,
                  gsl::span<const realtype> b,
                  gsl::span <const int> cols,
                  bool transpose) const;

    /**
     * @brief Perform matrix matrix multiplication A * B
              for sparse A, B, C
     * @param C output matrix, may not contain values but may be preallocated
     * @param B multiplication matrix
     */
    void sparse_multiply(SUNMatrixWrapper *C,
                         SUNMatrixWrapper *B) const;
    
    /**
     * @brief x = x + beta * A(:,j), where x is a dense vector and A(:,j) is sparse, and construct the pattern
     * for C(:,j)
     * @param j column index
     * @param beta scaling factor
     * @param w temporary index workspace, this keeps track of the sparsity pattern in C
     * @param x temporary data workspace, this keeps track of the data in C
     * @param mark marker for w to indicate nonzero pattern
     * @param C  output matrix
     * @param nz number of nonzeros that were already written to C
     * @return updated number of nonzeros in C
     */
    sunindextype scatter(const sunindextype j, const realtype beta,
                         sunindextype *w, realtype *x, const sunindextype mark,
                         SUNMatrixWrapper *C, sunindextype nz) const;

    /**
     * @brief Set to 0.0
     */
    void zero();

  private:
    void update_ptrs();

    /**
     * @brief CSC matrix to which all methods are applied
     */
    SUNMatrix matrix_ {nullptr};
    realtype *data_ptr_ {nullptr};
    sunindextype *indexptrs_ptr_ {nullptr};
    sunindextype *indexvals_ptr_ {nullptr};
};

} // namespace amici

namespace gsl {
/**
 * @brief Create span from SUNMatrix
 * @param m SUNMatrix
 * @return Created span
 */
inline span<realtype> make_span(SUNMatrix m)
{
    switch (SUNMatGetID(m)) {
    case SUNMATRIX_DENSE:
        return span<realtype>(SM_DATA_D(m), SM_LDATA_D(m));
    case SUNMATRIX_SPARSE:
        return span<realtype>(SM_DATA_S(m), SM_NNZ_S(m));
    default:
        throw amici::AmiException("Unimplemented SUNMatrix type for make_span");
    }
}
} // namespace gsl

#endif // AMICI_SUNDIALS_MATRIX_WRAPPER_H
