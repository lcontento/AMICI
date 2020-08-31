#include <amici/sundials_matrix_wrapper.h>
#include <sundials/sundials_matrix.h> // return codes

#include <amici/cblas.h>

#include <new> // bad_alloc
#include <utility>
#include <stdexcept> // invalid_argument and domain_error
#include <assert.h>

namespace amici {

SUNMatrixWrapper::SUNMatrixWrapper(int M, int N, int NNZ, int sparsetype)
    : matrix_(SUNSparseMatrix(M, N, NNZ, sparsetype)) {

    if (sparsetype != CSC_MAT && sparsetype != CSR_MAT)
        throw std::invalid_argument("Invalid sparsetype. Must be CSC_MAT or "
                                    "CSR_MAT");

    if (NNZ && M && N && !matrix_)
        throw std::bad_alloc();

    update_ptrs();
}

SUNMatrixWrapper::SUNMatrixWrapper(int M, int N)
    : matrix_(SUNDenseMatrix(M, N)) {
    if (M && N && !matrix_)
        throw std::bad_alloc();

    update_ptrs();
}

SUNMatrixWrapper::SUNMatrixWrapper(int M, int ubw, int lbw)
    : matrix_(SUNBandMatrix(M, ubw, lbw)) {
    if (M && !matrix_)
        throw std::bad_alloc();

    update_ptrs();
}

SUNMatrixWrapper::SUNMatrixWrapper(const SUNMatrixWrapper &A, realtype droptol,
                                   int sparsetype) {
    if (sparsetype != CSC_MAT && sparsetype != CSR_MAT)
        throw std::invalid_argument("Invalid sparsetype. Must be CSC_MAT or "
                                    "CSR_MAT");

    switch (SUNMatGetID(A.get())) {
    case SUNMATRIX_DENSE:
        matrix_ = SUNSparseFromDenseMatrix(A.get(), droptol, sparsetype);
        break;
    case SUNMATRIX_BAND:
        matrix_ = SUNSparseFromBandMatrix(A.get(), droptol, sparsetype);
        break;
    default:
        throw std::invalid_argument("Invalid Matrix. Must be SUNMATRIX_DENSE or"
                                    " SUNMATRIX_BAND");
    }

    if (!matrix_)
        throw std::bad_alloc();

    update_ptrs();
}

SUNMatrixWrapper::SUNMatrixWrapper(SUNMatrix mat) : matrix_(mat) {
    update_ptrs();
}

SUNMatrixWrapper::~SUNMatrixWrapper() {
    if (matrix_)
        SUNMatDestroy(matrix_);
}

SUNMatrixWrapper::SUNMatrixWrapper(const SUNMatrixWrapper &other) {
    if (!other.matrix_)
        return;

    matrix_ = SUNMatClone(other.matrix_);
    if (!matrix_)
        throw std::bad_alloc();

    SUNMatCopy(other.matrix_, matrix_);
    update_ptrs();
}

SUNMatrixWrapper::SUNMatrixWrapper(SUNMatrixWrapper &&other) {
    std::swap(matrix_, other.matrix_);
    update_ptrs();
}

SUNMatrixWrapper &SUNMatrixWrapper::operator=(const SUNMatrixWrapper &other) {
    if(&other == this)
        return *this;
    return *this = SUNMatrixWrapper(other);
}

SUNMatrixWrapper &SUNMatrixWrapper::
operator=(SUNMatrixWrapper &&other) {
    std::swap(matrix_, other.matrix_);
    update_ptrs();
    return *this;
}

void SUNMatrixWrapper::reallocate(int NNZ) {
    if (sparsetype() != CSC_MAT && sparsetype() != CSR_MAT)
        throw std::invalid_argument("Invalid sparsetype. Must be CSC_MAT or "
                                    "CSR_MAT.");
    
    if (int ret = SUNSparseMatrix_Reallocate(matrix_, NNZ) != SUNMAT_SUCCESS)
        throw std::runtime_error("SUNSparseMatrix_Reallocate failed with "
                                 "error code " + std::to_string(ret) + ".");

    update_ptrs();
    assert((NNZ && columns()*rows()) ^ !matrix_);
    assert(NNZ == capacity());
}

void SUNMatrixWrapper::realloc() {
    if (sparsetype() != CSC_MAT && sparsetype() != CSR_MAT)
        throw std::invalid_argument("Invalid sparsetype. Must be CSC_MAT or "
                                    "CSR_MAT.");
    if (int ret = SUNSparseMatrix_Realloc(matrix_) != SUNMAT_SUCCESS)
        throw std::runtime_error("SUNSparseMatrix_Realloc failed with "
                                 "error code " + std::to_string(ret) + ".");
    
    update_ptrs();
    assert(capacity() ^ !matrix_);
    assert(capacity() == num_nonzeros());
}

realtype *SUNMatrixWrapper::data() const {
    return data_ptr_;
}

sunindextype SUNMatrixWrapper::rows() const {
    if (!matrix_)
        return 0;

    switch (SUNMatGetID(matrix_)) {
    case SUNMATRIX_DENSE:
        return SM_ROWS_D(matrix_);
    case SUNMATRIX_SPARSE:
        return SM_ROWS_S(matrix_);
    case SUNMATRIX_BAND:
        return SM_ROWS_B(matrix_);
    case SUNMATRIX_CUSTOM:
        throw std::domain_error("Amici currently does not support custom matrix"
                                " types.");
    default:
        throw std::domain_error("Invalid SUNMatrix type.");
    }
}

sunindextype SUNMatrixWrapper::columns() const {
    if (!matrix_)
        return 0;

    switch (SUNMatGetID(matrix_)) {
    case SUNMATRIX_DENSE:
        return SM_COLUMNS_D(matrix_);
    case SUNMATRIX_SPARSE:
        return SM_COLUMNS_S(matrix_);
    case SUNMATRIX_BAND:
        return SM_COLUMNS_B(matrix_);
    case SUNMATRIX_CUSTOM:
        throw std::domain_error("Amici currently does not support custom matrix"
                                " types.");
    default:
        throw std::domain_error("Invalid SUNMatrix type.");
    }
}

sunindextype SUNMatrixWrapper::capacity() const {
    if (!matrix_)
        return 0;

    switch (SUNMatGetID(matrix_)) {
    case SUNMATRIX_SPARSE:
        return SM_NNZ_S(matrix_);
    default:
        throw std::domain_error("Non-zeros property only available for "
                                "sparse matrices");
    }
}

sunindextype SUNMatrixWrapper::num_nonzeros() const {
    if (!matrix_)
        return 0;
    
    if (SUNMatGetID(matrix_) == SUNMATRIX_SPARSE)
        return SM_INDEXPTRS_S(matrix_)[SM_NP_S(matrix_)];
    else
        throw std::domain_error("Non-zeros property only available for "
                                "sparse matrices");
}

sunindextype *SUNMatrixWrapper::indexvals() const {
    return indexvals_ptr_;
}

sunindextype *SUNMatrixWrapper::indexptrs() const {
    return indexptrs_ptr_;
}

int SUNMatrixWrapper::sparsetype() const {
    if (SUNMatGetID(matrix_) == SUNMATRIX_SPARSE)
        return SM_SPARSETYPE_S(matrix_);
    throw std::domain_error("Function only available for sparse matrices");
}

void SUNMatrixWrapper::reset() {
    if (matrix_)
        SUNMatZero(matrix_);
}

void SUNMatrixWrapper::scale(realtype a) {
    if (matrix_) {
        int nonzeros_ = static_cast<int>(num_nonzeros());
        for (int i = 0; i < nonzeros_; ++i)
            data_ptr_[i] *= a;
    }
}

void SUNMatrixWrapper::multiply(N_Vector c, const_N_Vector b) const {
    multiply(gsl::make_span<realtype>(NV_DATA_S(c), NV_LENGTH_S(c)),
             gsl::make_span<const realtype>(NV_DATA_S(b), NV_LENGTH_S(b)));
}

static void check_dim(int n, int m, std::string name_n, std::string name_m,
                      std::string name_mat_n, std::string name_mat_m) {
    if (n != m)
        throw std::invalid_argument("Dimension mismatch between number of "
                                    + name_n + " in " + name_mat_n + " ("
                                    + std::to_string(n)
                                    + ") and number of "
                                    + name_m + " in " + name_mat_m + " ("
                                    + std::to_string(m) + ")");
}

static void check_csc(const SUNMatrixWrapper *mat, std::string fun,
                      std::string name_mat) {
    if (mat->matrix_id() != SUNMATRIX_SPARSE)
        throw std::invalid_argument(fun + " only implemented for "
                                    "sparse matrices, but "
                                    + name_mat + " is not sparse.");

    if (mat->sparsetype() != CSC_MAT)
        throw std::invalid_argument(fun + " only implemented for "
                                    "matrix type CSC, but "
                                    + name_mat + "is not of type CSC.");
}

void SUNMatrixWrapper::multiply(gsl::span<realtype> c,
                                gsl::span<const realtype> b) const {
    if (!matrix_)
        return;

    sunindextype nrows = rows();
    sunindextype ncols = columns();

    check_dim(nrows, c.size(), "rows", "elements", "A", "c");
    check_dim(ncols, b.size(), "cols", "elements", "A", "b");

    switch (SUNMatGetID(matrix_)) {
    case SUNMATRIX_DENSE:
        amici_dgemv(BLASLayout::colMajor, BLASTranspose::noTrans,
                    static_cast<int>(nrows), static_cast<int>(ncols),
                    1.0, data(), static_cast<int>(nrows),
                    b.data(), 1, 1.0, c.data(), 1);
        break;
    case SUNMATRIX_SPARSE:
        if(!SM_NNZ_S(matrix_)) {
            /* empty matrix, nothing to multiply, return to avoid out-of-bounds
             * access of pointer access below
             */
            return;
        }
        switch (sparsetype()) {
        case CSC_MAT:
            for (sunindextype i = 0; i < ncols; ++i) {
                for (sunindextype k = indexptrs_ptr_[i]; k < indexptrs_ptr_[i + 1];
                     ++k) {
                    c[indexvals_ptr_[k]] += data_ptr_[k] * b[i];
                }
            }
            break;
        case CSR_MAT:
            for (sunindextype i = 0; i < nrows; ++i) {
                for (sunindextype k = indexptrs_ptr_[i]; k < indexptrs_ptr_[i + 1];
                     ++k) {
                    c[i] += data_ptr_[k] * b[indexvals_ptr_[k]];
                }
            }
            break;
        }
        break;
    case SUNMATRIX_BAND:
        throw std::domain_error("Not Implemented.");
    case SUNMATRIX_CUSTOM:
        throw std::domain_error("Amici currently does not support custom"
                                " matrix types.");
    case SUNMATRIX_SLUNRLOC:
        throw std::domain_error("Not Implemented.");
    case SUNMATRIX_CUSPARSE:
        throw std::domain_error("Not Implemented.");
    }

}

void SUNMatrixWrapper::multiply(N_Vector c,
                                const N_Vector b,
                                gsl::span <const int> cols,
                                bool transpose) const {
    multiply(gsl::make_span<realtype>(NV_DATA_S(c), NV_LENGTH_S(c)),
             gsl::make_span<const realtype>(NV_DATA_S(b), NV_LENGTH_S(b)),
             cols, transpose);
}

void SUNMatrixWrapper::multiply(gsl::span<realtype> c,
                                gsl::span<const realtype> b,
                                gsl::span<const int> cols,
                                bool transpose) const {
    if (!matrix_)
        return;

    sunindextype nrows = rows();
    sunindextype ncols = columns();

    if (transpose) {
        check_dim(ncols, c.size(), "columns", "elements", "A", "b");
        check_dim(nrows, b.size(), "rows", "elements", "A", "b");
    } else {
        check_dim(nrows, c.size(), "rows", "elements", "A", "c");
        check_dim(ncols, b.size(), "columns", "elements", "A", "b");
    }

    check_csc(this, "Reordered multiply", "A");
    
    if (!num_nonzeros())
        return;

    /* Carry out actual multiplication */
    if (transpose) {
        for (int i = 0; i < (int)cols.size(); ++i)
            for (sunindextype k = indexptrs_ptr_[cols[i]];
                 k < indexptrs_ptr_[cols[i] + 1]; ++k)
                c[i] += data_ptr_[k] * b[indexvals_ptr_[k]];
    } else {
        for (sunindextype i = 0; i < ncols; ++i)
            for (sunindextype k = indexptrs_ptr_[cols[i]];
                 k < indexptrs_ptr_[cols[i] + 1]; ++k)
                c[indexvals_ptr_[k]] += data_ptr_[k] * b[i];
    }
}


void SUNMatrixWrapper::sparse_multiply(SUNMatrixWrapper *C,
                                       SUNMatrixWrapper *B) const {
    if (!matrix_)
        return;

    sunindextype nrows = rows();
    sunindextype ncols = columns();

    check_csc(this, "sparse_multiply", "A");
    check_csc(B, "sparse_multiply", "B");
    check_csc(C, "sparse_multiply", "C");

    check_dim(nrows, C->rows(), "rows", "rows", "A", "C");
    check_dim(C->columns(), B->columns(), "columns", "columns", "C", "B");
    check_dim(B->rows(), ncols, "rows", "columns", "B", "A");
    
    if (ncols == 0 || nrows == 0 || B->columns() == 0)
        return; // matrix will also have zero size
    
    if (num_nonzeros() == 0 || B->num_nonzeros() == 0)
        return; // nothing to multiply
    

    /* see https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/CSparse/Source/cs_multiply.c
     * modified such that we don't need to use CSparse memory structure and can
     * work with preallocated C. This should minimize number of necessary
     * reallocations as we can assume that C doesn't change size.
     */
    
    sunindextype nnz = 0; // this keeps track of the nonzero index in C
    auto Bx = B->data();
    auto Bi = B->indexvals();
    auto Bp = B->indexptrs();
    
    sunindextype j;
    sunindextype p;
    auto Cx = C->data();
    auto Ci = C->indexvals();
    auto Cp = C->indexptrs();
    
    sunindextype m = nrows;
    sunindextype n = B->columns();
    
    auto w = std::vector<sunindextype>(m);
    auto x = std::vector<realtype>(m);
    
    for (j = 0; j < n; j++)
    {
        Cp[j] = nnz;                          /* column j of C starts here */
        if ((Bp[j+1] > Bp[j]) && (nnz + m > C->capacity()))
        {
            /*
             * if memory usage becomes a concern, remove the factor two here,
             * as it effectively trades memory efficiency against less
             * reallocations
             */
            C->reallocate(2*C->capacity() + m);
            // all pointers will change after reallocation
            Cx = C->data();
            Ci = C->indexvals();
            Cp = C->indexptrs();
        }
        for (p = Bp[j]; p < Bp[j+1]; p++)
        {
            nnz = scatter(Bi[p], Bx[p], w.data(), x.data(), j+1, C, nnz);
            assert(nnz - Cp[j] <= m);
        }
        for (p = Cp[j]; p < nnz; p++)
            Cx[p] = x[Ci[p]]; // copy data to C
    }
    Cp[n] = nnz;
    /*
     * do not reallocate here since we rather keep a matrix that is a bit
     * bigger than repeatedly resizing this matrix.
     */
}

void SUNMatrixWrapper::sparse_add(SUNMatrixWrapper *A, realtype alpha,
                                  SUNMatrixWrapper *B, realtype beta) {
    if (!matrix_)
        return;

    sunindextype nrows = rows();
    sunindextype ncols = columns();

    check_csc(this, "sparse_multiply", "C");
    check_csc(A, "sparse_multiply", "A");
    check_csc(B, "sparse_multiply", "B");

    check_dim(nrows, A->rows(), "rows", "rows", "C", "A");
    check_dim(nrows, B->rows(), "rows", "rows", "C", "B");
    check_dim(ncols, A->columns(), "columns", "columns", "C", "A");
    check_dim(ncols, B->columns(), "columns", "columns", "C", "B");
    
    if (ncols == 0 || nrows == 0 ||
        (A->num_nonzeros() == 0 && B->num_nonzeros() == 0))
        return; // nothing to do
    

    /* see https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/CSparse/Source/cs_add.c
     * modified such that we don't need to use CSparse memory structure and can
     * work with preallocated C. This should minimize number of necessary
     * reallocations as we can assume that C doesn't change size.
     */
    
    sunindextype nz = 0; // this keeps track of the nonzero index in C
    
    sunindextype j, p;
    auto Cx = data();
    auto Ci = indexvals();
    auto Cp = indexptrs();
    
    sunindextype m = nrows;
    sunindextype n = B->columns();
    
    auto w = std::vector<sunindextype>(m);
    auto x = std::vector<realtype>(m);
    
    for (j = 0; j < n; j++)
    {
        Cp[j] = nz;                          /* column j of C starts here */
        nz = A->scatter(j, alpha, w.data(), x.data(), j+1, this, nz);
        nz = B->scatter(j, beta, w.data(), x.data(), j+1, this, nz);
        // no reallocation should happen here
        for (p = Cp[j]; p < nz; p++)
            Cx[p] = x.at(Ci[p]); // copy data to C
    }
    Cp[n] = nz;
    realloc();
}

sunindextype SUNMatrixWrapper::scatter(const sunindextype j,
                                       const realtype beta,
                                       sunindextype *w, realtype *x,
                                       const sunindextype mark,
                                       SUNMatrixWrapper *C,
                                       sunindextype nnz) const {
    if (sparsetype() != CSC_MAT)
        throw std::invalid_argument("Matrix A not of type CSC_MAT");
    
    if (C->sparsetype() != CSC_MAT)
        throw std::invalid_argument("Matrix C not of type CSC_MAT");
    
    if (!num_nonzeros())
        return nnz;
    
    /* see https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/master/CSparse/Source/cs_scatter.c */
    
    sunindextype *Ci;
    if (C)
        Ci = C->indexvals();
    auto Ap = indexptrs();
    auto Ai = indexvals();
    auto Ax = data();
    for (sunindextype p = Ap[j]; p < Ap[j+1]; p++)
    {
        auto i = Ai[p];                   /* A(i,j) is nonzero */
        assert((C && w) ^ (!C && !w));
        if (C && w && w[i] < mark) {
            w[i] = mark;                  /* i is new entry in column j */
            Ci[nnz++] = i;                 /* add i to pattern of C(:,j) */
            x[i] = beta * Ax[p];          /* x(i) = beta*A(i,j) */
        }
        else x[i] += beta * Ax[p];        /* i exists in C(:,j) already */
    }
    return nnz;
}

void SUNMatrixWrapper::zero()
{
    if(int res = SUNMatZero(matrix_))
        throw std::runtime_error("SUNMatrixWrapper::zero() failed with "
                                 + std::to_string(res));
}

void SUNMatrixWrapper::update_ptrs() {
    if(!matrix_)
        return;

    switch (SUNMatGetID(matrix_)) {
    case SUNMATRIX_DENSE:
        if (columns() > 0 && rows() > 0)
            data_ptr_ = SM_DATA_D(matrix_);
        break;
    case SUNMATRIX_SPARSE:
        if (SM_NNZ_S(matrix_) > 0) {
            data_ptr_ = SM_DATA_S(matrix_);
            indexptrs_ptr_ = SM_INDEXPTRS_S(matrix_);
            indexvals_ptr_ = SM_INDEXVALS_S(matrix_);
        }
        break;
    case SUNMATRIX_BAND:
        if (columns() > 0 && rows() > 0)
            data_ptr_ = SM_DATA_B(matrix_);
        break;
    case SUNMATRIX_CUSTOM:
        throw std::domain_error("Amici currently does not support "
                                "custom matrix types.");
    case SUNMATRIX_SLUNRLOC:
        throw std::domain_error("Not Implemented.");
    case SUNMATRIX_CUSPARSE:
        throw std::domain_error("Not Implemented.");
    }
}

SUNMatrix SUNMatrixWrapper::get() const { return matrix_; }

} // namespace amici

