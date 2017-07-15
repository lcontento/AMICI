#ifndef amici_h
#define amici_h
#include <include/symbolic_functions.h>
#include <include/udata.h>
#include <include/rdata.h>
#include <include/edata.h>
#include <include/tdata.h>
#include <cstdbool>
#include <cvodes/cvodes.h>

#include <include/amici_defines.h>

// ensure definitions are in sync
static_assert(AMICI_SUCCESS == CV_SUCCESS, "AMICI_SUCCESS != CV_SUCCESS");
static_assert(AMICI_DATA_RETURN == CV_TSTOP_RETURN, "AMICI_DATA_RETURN != CV_TSTOP_RETURN");
static_assert(AMICI_ROOT_RETURN == CV_ROOT_RETURN, "AMICI_ROOT_RETURN != CV_ROOT_RETURN");
static_assert(AMICI_NORMAL == CV_NORMAL, "AMICI_NORMAL != CV_NORMAL");
static_assert(AMICI_ONE_STEP == CV_ONE_STEP, "AMICI_ONE_STEP != CV_ONE_STEP");

void errMsgIdAndTxt(
                            const char * identifier, /* string with error message identifier */
                            const char * err_msg,    /* string with error message printf-style format */
                            ...                      /* any additional arguments */
);

void warnMsgIdAndTxt(
                             const char * identifier, /* string with error message identifier */
                             const char * err_msg,    /* string with error message printf-style format */
                             ...                      /* any additional arguments */
);

int runAmiciSimulation(UserData *udata, const ExpData *edata, ReturnData *rdata);

void *setupAMI(UserData *udata, TempData *tdata);
int setupAMIB(void *ami_mem, UserData *udata, TempData *tdata);

int prepDataSensis(int it, void *ami_mem, UserData *udata, ReturnData *rdata, const ExpData *edata, TempData *tdata);
int prepEventSensis(int ie, UserData *udata, ReturnData *rdata, const ExpData *edata, TempData *tdata);

int getDataSensisFSA(int it, void *ami_mem, UserData *udata, ReturnData *rdata, const ExpData *edata, TempData *tdata);
int getEventSensisFSA(int ie, UserData *udata, ReturnData *rdata, const ExpData *edata, TempData *tdata);

int getDataOutput(int it, void *ami_mem, UserData *udata, ReturnData *rdata, const ExpData *edata, TempData *tdata);
int getEventOutput(realtype *tlastroot, UserData *udata, ReturnData *rdata, const ExpData *edata, TempData *tdata);

int handleEvent(int *iroot, realtype *tlastroot, void *ami_mem, UserData *udata, ReturnData *rdata, const ExpData *edata, TempData *tdata, int seflag);
int handleDataPoint(int it, void *ami_mem, UserData *udata, ReturnData *rdata, const ExpData *edata, TempData *tdata);
int handleEventB(int iroot, UserData *udata, TempData *tdata);
int handleDataPointB(int it, void *ami_mem, UserData *udata, ReturnData *rdata, TempData *tdata);

int applyEventBolus(UserData *udata, TempData *tdata);
int applyEventSensiBolusFSA( UserData *udata, TempData *tdata);

realtype getTnext(realtype *troot, int iroot, realtype *tdata, int it, UserData *udata);

int initHeaviside(UserData *udata, TempData *tdata);
int updateHeaviside(UserData *udata, TempData *tdata);
int updateHeavisideB(int iroot, UserData *udata, TempData *tdata);

int getDiagnosis(int it, void *ami_mem, UserData *udata, ReturnData *rdata);
int getDiagnosisB(int it, void *ami_mem, UserData *udata, ReturnData *rdata, TempData *tdata);

int workForwardProblem(UserData *udata, TempData *tdata, ReturnData *rdata, const ExpData *edata, void *ami_mem, int* iroot);
int workBackwardProblem(UserData *udata, TempData *tdata, ReturnData *rdata, const ExpData *edata, void *ami_mem, int *iroot);
int storeJacobianAndDerivativeInReturnData(UserData *udata, TempData *tdata, ReturnData *rdata);

void amici_dgemv(AMICI_BLAS_LAYOUT layout,
                 AMICI_BLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY);

void amici_dgemm(AMICI_BLAS_LAYOUT layout, AMICI_BLAS_TRANSPOSE TransA,
                 AMICI_BLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);

#endif /* amici_h */
