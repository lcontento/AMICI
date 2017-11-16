#include <cvodes/cvodes.h>
/*#include <cvodes/cvodes_lapack.h>*/
#include <cvodes/cvodes_band.h>
#include <cvodes/cvodes_bbdpre.h>
#include <cvodes/cvodes_dense.h>
#include <cvodes/cvodes_diag.h>
#include <cvodes/cvodes_klu.h>
#include <cvodes/cvodes_spbcgs.h>
#include <cvodes/cvodes_spgmr.h>
#include <cvodes/cvodes_sptfqmr.h>

#include <amd.h>
#include <btf.h>
#include <colamd.h>
#include <klu.h>

#include <include/amici_model_ode.h>
#include <include/amici.h>
#include <include/amici_solver_cvodes.h>
#include <include/amici_exception.h>
#include <include/udata.h>

/**
 * @ brief extract information from a property of a matlab class (matrix)
 * @ param OPTION name of the property
 */

namespace amici {

void CVodeSolver::init(AmiVector *x, AmiVector *dx, realtype t) {
    int status = CVodeInit(ami_mem, fxdot, RCONST(t), x->getNVector());
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeInit");
}

void CVodeSolver::binit(int which, AmiVector *xB, AmiVector *dxB, realtype t) {
    int status = CVodeInitB(ami_mem, which, fxBdot, RCONST(t), xB->getNVector());
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeInitB");
}

void CVodeSolver::qbinit(int which, AmiVector *qBdot) {
    int status = CVodeQuadInitB(ami_mem, which, fqBdot, qBdot->getNVector());
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeQuadInitB");
}

void CVodeSolver::rootInit(int ne) {
    int status = CVodeRootInit(ami_mem, ne, froot);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeRootInit");
}

void CVodeSolver::sensInit1(AmiVectorArray *sx, AmiVectorArray *sdx, const UserData *udata) {
    int status = CVodeSensInit1(ami_mem, udata->nplist(), udata->sensmeth(), fsxdot,
                          sx->getNVectorArray());
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSensInit1");
}

void CVodeSolver::setDenseJacFn() {
    int status = CVDlsSetDenseJacFn(ami_mem, fJ);
    if(status != CV_SUCCESS)
        throw CvodeException(status,"CVDlsSetDenseJacFn");
}

void CVodeSolver::setSparseJacFn() {
    int status = CVSlsSetSparseJacFn(ami_mem, fJSparse);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVSlsSetSparseJacFn");
}

void CVodeSolver::setBandJacFn() {
    int status = CVDlsSetBandJacFn(ami_mem, fJBand);
    if(status != CV_SUCCESS)
        throw CvodeException(status,"CVDlsSetBandJacFn");
}

void CVodeSolver::setJacTimesVecFn() {
    int status = CVSpilsSetJacTimesVecFn(ami_mem, fJv);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVSpilsSetJacTimesVecFn");
}

void CVodeSolver::setDenseJacFnB(int which) {
    int status = CVDlsSetDenseJacFnB(ami_mem, which, fJB);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVDlsSetDenseJacFnB");
}

void CVodeSolver::setSparseJacFnB(int which) {
    int status = CVSlsSetSparseJacFnB(ami_mem, which, fJSparseB);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVSlsSetSparseJacFnB");
}

void CVodeSolver::setBandJacFnB(int which) {
    int status = CVDlsSetBandJacFnB(ami_mem, which, fJBandB);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVDlsSetBandJacFnB");
}

void CVodeSolver::setJacTimesVecFnB(int which) {
    int status = CVSpilsSetJacTimesVecFnB(ami_mem, which, fJvB);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVSpilsSetJacTimesVecFnB");
}

void *CVodeSolver::AMICreate(int lmm, int iter) {
    return CVodeCreate(lmm, iter);
}

void CVodeSolver::AMISStolerances(double rtol, double atol) {
    int status = CVodeSStolerances(ami_mem, rtol, atol);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSStolerances");
}

void CVodeSolver::AMISensEEtolerances() {
    int status = CVodeSensEEtolerances(ami_mem);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSensEEtolerances");
}

void CVodeSolver::AMISetSensErrCon(bool error_corr) {
    int status = CVodeSetSensErrCon(ami_mem, error_corr);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSetSensErrCon");
}

void CVodeSolver::AMISetQuadErrConB(int which, bool flag) {
    int status = CVodeSetQuadErrConB(ami_mem, which, flag);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSetQuadErrConB");
}

void CVodeSolver::AMIGetRootInfo(int *rootsfound) {
    int status = CVodeGetRootInfo(ami_mem, rootsfound);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeGetRootInfo");
}

void CVodeSolver::AMISetErrHandlerFn() {
    int status = CVodeSetErrHandlerFn(ami_mem, wrapErrHandlerFn, NULL);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSetErrHandlerFn");
}

void CVodeSolver::AMISetUserData(void *user_data) {
    int status = CVodeSetUserData(ami_mem, user_data);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSetUserData");
}

void CVodeSolver::AMISetUserDataB(int which, void *user_data) {
    int status = CVodeSetUserDataB(ami_mem, which, user_data);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSetUserDataB");
}

void CVodeSolver::AMISetMaxNumSteps(long mxsteps) {
    int status = CVodeSetMaxNumSteps(ami_mem, mxsteps);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSetMaxNumSteps");
}

void CVodeSolver::AMISetStabLimDet(int stldet) {
    int status = CVodeSetStabLimDet(ami_mem, stldet);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSetStabLimDet");
}

void CVodeSolver::AMISetStabLimDetB(int which, int stldet) {
    int status = CVodeSetStabLimDetB(ami_mem, which, stldet);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSetStabLimDetB");
}

void CVodeSolver::AMISetId(Model *model) { return; }

void CVodeSolver::AMISetSuppressAlg(bool flag) { return; }

void CVodeSolver::AMIReInit(realtype t0, AmiVector *yy0, AmiVector *yp0) {
    int status = CVodeReInit(ami_mem, t0, yy0->getNVector());
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeReInit");
}

void CVodeSolver::AMISensReInit(int ism, AmiVectorArray *yS0, AmiVectorArray *ypS0) {
    int status = CVodeSensReInit(ami_mem, ism, yS0->getNVectorArray());
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSensReInit");
}

void CVodeSolver::AMISetSensParams(realtype *p, realtype *pbar, int *plist) {
    int status = CVodeSetSensParams(ami_mem, p, pbar, plist);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSetSensParams");
}

void CVodeSolver::AMIGetDky(realtype t, int k, AmiVector *dky) {
    int status = CVodeGetDky(ami_mem, t, k, dky->getNVector());
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeGetDky");
}

void CVodeSolver::AMIGetSens(realtype *tret, AmiVectorArray *yySout) {
    int status = CVodeGetSens(ami_mem, tret, yySout->getNVectorArray());
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeGetSens");
}
    
void CVodeSolver::AMIFree() {
    CVodeFree(&ami_mem);
    ami_mem = NULL;
}

void CVodeSolver::AMIAdjInit(long steps, int interp) {
    int status = CVodeAdjInit(ami_mem, steps, interp);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeAdjInit");
}

void CVodeSolver::AMICreateB(int lmm, int iter, int *which) {
    int status = CVodeCreateB(ami_mem, lmm, iter, which);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeCreateB");
}

void CVodeSolver::AMIReInitB(int which, realtype tB0, AmiVector *yyB0,
                            AmiVector *ypB0) {
    int status = CVodeReInitB(ami_mem, which, tB0, yyB0->getNVector());
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeReInitB");
}

void CVodeSolver::AMISStolerancesB(int which, realtype relTolB,
                                  realtype absTolB) {
    int status = CVodeSStolerancesB(ami_mem, which, relTolB, absTolB);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSStolerancesB");
}

void CVodeSolver::AMIQuadReInitB(int which, AmiVector *yQB0) {
    int status = CVodeQuadReInitB(ami_mem, which, yQB0->getNVector());
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeQuadReInitB");
}

void CVodeSolver::AMIQuadSStolerancesB(int which, realtype reltolQB,
                                      realtype abstolQB) {
    int status = CVodeQuadSStolerancesB(ami_mem, which, reltolQB, abstolQB);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeQuadSStolerancesB");
}

int CVodeSolver::AMISolve(realtype tout, AmiVector *yret, AmiVector *ypret,
                          realtype *tret, int itask) {
    int status = CVode(ami_mem, tout, yret->getNVector(), tret, itask);
    if(status<0) {
        throw IntegrationFailure(status,*tret);
    } else{
        return status;
    }
}

int CVodeSolver::AMISolveF(realtype tout, AmiVector *yret, AmiVector *ypret,
                           realtype *tret, int itask, int *ncheckPtr) {
    int status = CVodeF(ami_mem, tout, yret->getNVector(), tret, itask, ncheckPtr);
    if(status<0) {
        throw IntegrationFailure(status,*tret);
    } else{
        return status;
    }
}

void CVodeSolver::AMISolveB(realtype tBout, int itaskB) {
    int status = CVodeB(ami_mem, tBout, itaskB);
    if(status != CV_SUCCESS)
         throw IntegrationFailure(status,tBout);
}

void CVodeSolver::AMISetMaxNumStepsB(int which, long mxstepsB) {
    int status = CVodeSetMaxNumStepsB(ami_mem, which, mxstepsB);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSetMaxNumStepsB");
}

void CVodeSolver::AMIGetB(int which, realtype *tret, AmiVector *yy, AmiVector *yp) {
    int status = CVodeGetB(ami_mem, which, tret, yy->getNVector());
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeGetB");
}

void CVodeSolver::AMIGetQuadB(int which, realtype *tret, AmiVector *qB) {
    int status = CVodeGetQuadB(ami_mem, which, tret, qB->getNVector());
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeGetQuadB");
}

void CVodeSolver::AMIDense(int nx) {
    int status = CVDense(ami_mem, nx);
    if(status != CV_SUCCESS)
        throw CvodeException(status,"CVDense");
}

void CVodeSolver::AMIDenseB(int which, int nx) {
    int status = CVDenseB(ami_mem, which, nx);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVDenseB");
}

void CVodeSolver::AMIBand(int nx, int ubw, int lbw) {
    int status = CVBand(ami_mem, nx, ubw, lbw);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVBand");
}

void CVodeSolver::AMIBandB(int which, int nx, int ubw, int lbw) {
    int status = CVBandB(ami_mem, which, nx, ubw, lbw);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVBandB");
}

void CVodeSolver::AMIDiag() {
    int status = CVDiag(ami_mem);
    if(status != CV_SUCCESS)
        throw CvodeException(status,"CVDiag");
}

void CVodeSolver::AMIDiagB(int which) {
    int status = CVDiagB(ami_mem, which);
        if(status != CV_SUCCESS)
            throw CvodeException(status,"CVDiagB");
}

void CVodeSolver::AMISpgmr(int prectype, int maxl) {
    int status = CVSpgmr(ami_mem, prectype, maxl);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVSpgmr");
}

void CVodeSolver::AMISpgmrB(int which, int prectype, int maxl) {
    int status = CVSpgmrB(ami_mem, which, prectype, maxl);
    if(status != CV_SUCCESS)
        throw CvodeException(status,"CVSpgmrB");
}

void CVodeSolver::AMISpbcg(int prectype, int maxl) {
    int status = CVSpbcg(ami_mem, prectype, maxl);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVSpbcg");
}

void CVodeSolver::AMISpbcgB(int which, int prectype, int maxl) {
    int status = CVSpbcgB(ami_mem, which, prectype, maxl);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVSpbcgB");
}

void CVodeSolver::AMISptfqmr(int prectype, int maxl) {
    int status = CVSptfqmr(ami_mem, prectype, maxl);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"AMISptfqmr");
}

void CVodeSolver::AMISptfqmrB(int which, int prectype, int maxl) {
    int status = CVSptfqmrB(ami_mem, which, prectype, maxl);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVSptfqmrB");
}

void CVodeSolver::AMIKLU(int nx, int nnz, int sparsetype) {
    int status = CVKLU(ami_mem, nx, nnz, sparsetype);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVKLU");
}

void CVodeSolver::AMIKLUSetOrdering(int ordering) {
    int status = CVKLUSetOrdering(ami_mem, ordering);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVKLUSetOrdering");
}

void CVodeSolver::AMIKLUSetOrderingB(int which, int ordering) {
    int status = CVKLUSetOrderingB(ami_mem, which, ordering);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVKLUSetOrderingB");
}

void CVodeSolver::AMIKLUB(int which, int nx, int nnz, int sparsetype) {
    int status = CVKLUB(ami_mem, which, nx, nnz, sparsetype);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVKLUB");
}

void CVodeSolver::AMIGetNumSteps(void *ami_mem, long *numsteps) {
    int status = CVodeGetNumSteps(ami_mem, numsteps);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeGetNumSteps");
}

void CVodeSolver::AMIGetNumRhsEvals(void *ami_mem, long *numrhsevals) {
    int status = CVodeGetNumRhsEvals(ami_mem, numrhsevals);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeGetNumRhsEvals");
}

void CVodeSolver::AMIGetNumErrTestFails(void *ami_mem, long *numerrtestfails) {
    int status = CVodeGetNumErrTestFails(ami_mem, numerrtestfails);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeGetNumErrTestFails");
}

void CVodeSolver::AMIGetNumNonlinSolvConvFails(void *ami_mem,
                                              long *numnonlinsolvconvfails) {
    int status = CVodeGetNumNonlinSolvConvFails(ami_mem, numnonlinsolvconvfails);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeGetNumNonlinSolvConvFails");
}

void CVodeSolver::AMIGetLastOrder(void *ami_ami_mem, int *order) {
    int status = CVodeGetLastOrder(ami_mem, order);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeGetLastOrder");
}

void *CVodeSolver::AMIGetAdjBmem(void *ami_mem, int which) {
    return CVodeGetAdjCVodeBmem(ami_mem, which);
}

void CVodeSolver::AMICalcIC(realtype tout1, AmiVector *x, AmiVector *dx) { };

void CVodeSolver::AMICalcICB(int which, realtype tout1, AmiVector *xB,
                             AmiVector *dxB) {};

void CVodeSolver::AMISetStopTime(realtype tstop) {
    int status = CVodeSetStopTime(ami_mem, tstop);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeSetStopTime");
}

void CVodeSolver::turnOffRootFinding() {
    int status = CVodeRootInit(ami_mem, 0, NULL);
    if(status != CV_SUCCESS)
         throw CvodeException(status,"CVodeRootInit");
}
    
    /** Jacobian of xdot with respect to states x
     * @param N number of state variables
     * @param t timepoint
     * @param x Vector with the states
     * @param xdot Vector with the right hand side
     * @param J Matrix to which the Jacobian will be written
     * @param user_data object with user input @type UserData
     * @param tmp1 temporary storage vector
     * @param tmp2 temporary storage vector
     * @param tmp3 temporary storage vector
     * @return status flag indicating successful execution
     **/
    int CVodeSolver::fJ(long int N, realtype t, N_Vector x, N_Vector xdot,
           DlsMat J, void *user_data, N_Vector tmp1,
           N_Vector tmp2, N_Vector tmp3) {
        Model_ODE *model = (Model_ODE*) user_data;
        model->fdwdx(t,x);
        memset(J->data,0.0,sizeof(realtype)*N);
        model->model_J(J->data,t,N_VGetArrayPointer(x),model->p.data(),model->k.data(),model->h.data(),
                       model->w.data(),model->dwdx.data());
        return checkVals(N,J->data,"Jacobian");
    }
    
    /** Jacobian of xBdot with respect to adjoint state xB
     * @param NeqBdot number of adjoint state variables
     * @param t timepoint
     * @param x Vector with the states
     * @param xB Vector with the adjoint states
     * @param xBdot Vector with the adjoint right hand side
     * @param JB Matrix to which the Jacobian will be written
     * @param user_data object with user input @type UserData
     * @param tmp1B temporary storage vector
     * @param tmp2B temporary storage vector
     * @param tmp3B temporary storage vector
     * @return status flag indicating successful execution
     **/
    int CVodeSolver::fJB(long int NeqBdot, realtype t, N_Vector x, N_Vector xB,
                   N_Vector xBdot, DlsMat JB, void *user_data, N_Vector tmp1B,
                   N_Vector tmp2B, N_Vector tmp3B) {
        Model_ODE *model = (Model_ODE*) user_data;
        model->fdwdx(t,x);
        memset(JB->data,0.0,sizeof(realtype)*NeqBdot);
        if(model->model_JB(JB->data,t,N_VGetArrayPointer(x),model->p.data(),model->k.data(),model->h.data(),
                     N_VGetArrayPointer(xB),model->w.data(),model->dwdx.data()) != AMICI_SUCCESS)
            return AMICI_ERROR;
        return checkVals(NeqBdot,JB->data,"Jacobian");
    }
    
    /** J in sparse form (for sparse solvers from the SuiteSparse Package)
     * @param t timepoint
     * @param cj scalar in Jacobian (inverse stepsize)
     * @param x Vector with the states
     * @param dx Vector with the derivative states
     *N_Vector
     * @param xdot Vector with the right hand side
     * @param J Matrix to which the Jacobian will be written
     * @param user_data object with user input @type UserData
     * @param tmp1 temporary storage vector
     * @param tmp2 temporary storage vector
     * @param tmp3 temporary storage vector
     * @return status flag indicating successful execution
     */
    int CVodeSolver::fJSparse(realtype t, N_Vector x, N_Vector xdot, SlsMat J,
                        void *user_data, N_Vector tmp1, N_Vector tmp2,
                        N_Vector tmp3) {
        Model_ODE *model = (Model_ODE*) user_data;
        model->fdwdx(t,x);
        memset(J->data,0.0,sizeof(realtype)*J->NNZ);
        model->model_JSparse(J->data,t,N_VGetArrayPointer(x),model->p.data(),model->k.data(),model->h.data(),
                      model->w.data(),model->dwdx.data());
        return checkVals(J->NNZ,J->data,"Jacobian");
    }
    
    /** JB in sparse form (for sparse solvers from the SuiteSparse Package)
     * @param t timepoint
     * @param x Vector with the states
     * @param xB Vector with the adjoint states
     * @param xBdot Vector with the adjoint right hand side
     * @param JB Matrix to which the Jacobian will be written
     * @param user_data object with user input @type UserData
     * @param tmp1B temporary storage vector
     * @param tmp2B temporary storage vector
     * @param tmp3B temporary storage vector
     * @return status flag indicating successful execution
     */
    int CVodeSolver::fJSparseB(realtype t, N_Vector x, N_Vector xB, N_Vector xBdot,
                         SlsMat JB, void *user_data, N_Vector tmp1B,
                         N_Vector tmp2B, N_Vector tmp3B) {
        Model_ODE *model = (Model_ODE*) user_data;
        model->fdwdx(t,x);
        memset(JB->data,0.0,sizeof(realtype)*JB->NNZ);
        if(model->model_JSparseB(JB->data,t,N_VGetArrayPointer(x),model->p.data(),model->k.data(),model->h.data(),
                           N_VGetArrayPointer(xB),model->w.data(),model->dwdx.data()) != AMICI_SUCCESS)
            return AMICI_ERROR;
        return checkVals(JB->NNZ,JB->data,"Jacobian");
    }
    
    /** J in banded form (for banded solvers)
     * @param N number of states
     * @param mupper upper matrix bandwidth
     * @param mlower lower matrix bandwidth
     * @param t timepoint
     * @param x Vector with the states
     * @param xdot Vector with the right hand side
     * @param J Matrix to which the Jacobian will be written
     * @param user_data object with user input @type UserData
     * @param tmp1 temporary storage vector
     * @param tmp2 temporary storage vector
     * @param tmp3 temporary storage vector
     * @return status flag indicating successful execution
     */
    int CVodeSolver::fJBand(long int N, long int mupper, long int mlower, realtype t,
                      N_Vector x, N_Vector xdot, DlsMat J, void *user_data,
                      N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
        return fJ(N,t,x,xdot,J,user_data,tmp1,tmp2,tmp3);
    }
    
    /** JB in banded form (for banded solvers)
     * @param NeqBdot number of states
     * @param mupper upper matrix bandwidth
     * @param mlower lower matrix bandwidth
     * @param t timepoint
     * @param x Vector with the states
     * @param xB Vector with the adjoint states
     * @param xBdot Vector with the adjoint right hand side
     * @param JB Matrix to which the Jacobian will be written
     * @param user_data object with user input @type UserData
     * @param tmp1B temporary storage vector
     * @param tmp2B temporary storage vector
     * @param tmp3B temporary storage vector
     * @return status flag indicating successful execution
     */
    int CVodeSolver::fJBandB(long int NeqBdot, long int mupper, long int mlower,
                       realtype t, N_Vector x, N_Vector xB, N_Vector xBdot,
                       DlsMat JB, void *user_data, N_Vector tmp1B,
                       N_Vector tmp2B, N_Vector tmp3B) {
        return fJB(NeqBdot,t,x,xB,xBdot,JB,user_data,tmp1B,tmp2B,tmp3B);
    }
    
    /** diagonalized Jacobian (for preconditioning)
     * @param t timepoint
     * @param JDiag Vector to which the Jacobian diagonal will be written
     * @param cj scaling factor, inverse of the step size
     * @param x Vector with the states
     * @param dx Vector with the derivative states
     * @param user_data object with user input @type UserData
     **/
    int CVodeSolver::fJDiag(realtype t, N_Vector JDiag, N_Vector x,
                      void *user_data) {
        Model_ODE *model = (Model_ODE*) user_data;
        model->fdwdx(t,x);
        memset(N_VGetArrayPointer(JDiag),0.0,sizeof(realtype)*model->nx);
        if(model->model_JDiag(N_VGetArrayPointer(JDiag),t,N_VGetArrayPointer(x),model->p.data(),model->k.data(),model->h.data(),
                        model->w.data(),model->dwdx.data()) != AMICI_SUCCESS)
            return AMICI_ERROR;
        return checkVals(model->nx,N_VGetArrayPointer(JDiag),"Jacobian");
    }
    
    /** Matrix vector product of J with a vector v (for iterative solvers)
     * @param t timepoint @type realtype
     * @param x Vector with the states
     * @param xdot Vector with the right hand side
     * @param v Vector with which the Jacobian is multiplied
     * @param Jv Vector to which the Jacobian vector product will be
     *written
     * @param user_data object with user input @type UserData
     * @param tmp temporary storage vector
     * @return status flag indicating successful execution
     **/
    int CVodeSolver::fJv(N_Vector v, N_Vector Jv, realtype t, N_Vector x, N_Vector xdot,
                   void *user_data, N_Vector tmp) {
        Model_ODE *model = (Model_ODE*) user_data;
        model->fdwdx(t,x);
        memset(N_VGetArrayPointer(Jv),0.0,sizeof(realtype)*model->nx);
        if(model->model_Jv(N_VGetArrayPointer(Jv),t,N_VGetArrayPointer(x),model->p.data(),model->k.data(),model->h.data(),
                     N_VGetArrayPointer(v),model->w.data(),model->dwdx.data()) != AMICI_SUCCESS)
            return AMICI_ERROR;
        return checkVals(model->nx,N_VGetArrayPointer(Jv),"Jacobian");
    }
    
    /** Matrix vector product of JB with a vector v (for iterative solvers)
     * @param t timepoint
     * @param x Vector with the states
     * @param xB Vector with the adjoint states
     * @param xBdot Vector with the adjoint right hand side
     * @param vB Vector with which the Jacobian is multiplied
     * @param JvB Vector to which the Jacobian vector product will be
     *written
     * @param user_data object with user input @type UserData
     * @param tmpB temporary storage vector
     * @return status flag indicating successful execution
     **/
    int CVodeSolver::fJvB(N_Vector vB, N_Vector JvB, realtype t, N_Vector x, N_Vector xB, N_Vector xBdot,
                    void *user_data, N_Vector tmpB) {
        Model_ODE *model = (Model_ODE*) user_data;
        model->fdwdx(t,x);
        memset(N_VGetArrayPointer(JvB),0.0,sizeof(realtype)*model->nx);
        if(model->model_JvB(N_VGetArrayPointer(JvB),t,N_VGetArrayPointer(x),model->p.data(),model->k.data(),model->h.data(),
                      N_VGetArrayPointer(xB),N_VGetArrayPointer(vB),model->w.data(),model->dwdx.data()) != AMICI_SUCCESS)
        return AMICI_ERROR;
        return checkVals(model->nx,N_VGetArrayPointer(JvB),"Jacobian");
    }
    
    /** Event trigger function for events
     * @param t timepoint
     * @param x Vector with the states
     * @param root array with root function values
     * @param user_data object with user input @type UserData
     * @return status flag indicating successful execution
     */
    int CVodeSolver::froot(realtype t, N_Vector x, realtype *root,
                     void *user_data) {
        Model_ODE *model = (Model_ODE*) user_data;
        memset(root,0.0,sizeof(realtype)*model->ne);
        if(model->model_root(root,t,N_VGetArrayPointer(x),model->p.data(),model->k.data(),model->h.data()) != AMICI_SUCCESS)
            return AMICI_ERROR;
        return checkVals(model->ne,root,"root function");
    }
    
    /** residual function of the DAE
     * @param t timepoint
     * @param x Vector with the states
     * @param xdot Vector with the right hand side
     * @param user_data object with user input @type UserData
     * @return status flag indicating successful execution
     */
    int CVodeSolver::fxdot(realtype t, N_Vector x, N_Vector xdot, void *user_data) {
        Model_ODE *model = (Model_ODE*) user_data;
        model->fw(t,x);
        memset(N_VGetArrayPointer(xdot),0.0,sizeof(realtype)*model->nx);
        model->model_xdot(N_VGetArrayPointer(xdot),t,N_VGetArrayPointer(x),model->p.data(),model->k.data(),model->h.data(),
                   model->w.data());
        return checkVals(model->nx,N_VGetArrayPointer(xdot),"residual function");
    }
    
    /** Right hand side of differential equation for adjoint state xB
     * @param t timepoint
     * @param x Vector with the states
     * @param xB Vector with the adjoint states
     * @param xBdot Vector with the adjoint right hand side
     * @param user_data object with user input @type UserData
     * @return status flag indicating successful execution
     */
    int CVodeSolver::fxBdot(realtype t, N_Vector x, N_Vector xB,
                      N_Vector xBdot, void *user_data) {
        Model_ODE *model = (Model_ODE*) user_data;
        model->fdwdx(t,x);
        memset(N_VGetArrayPointer(xBdot),0.0,sizeof(realtype)*model->nx);
        model->model_xBdot(N_VGetArrayPointer(xBdot),t,N_VGetArrayPointer(x),model->p.data(),model->k.data(),model->h.data(),
                    N_VGetArrayPointer(xB),model->w.data(),model->dwdx.data());
        return checkVals(model->nx,N_VGetArrayPointer(xBdot),"adjoint residual function");
    }
    
    /** Right hand side of integral equation for quadrature states qB
     * @param t timepoint @type realtype
     * @param x Vector with the states @type N_Vector
     * @param dx Vector with the derivative states (only DAE) @type
     *N_Vector
     * @param xB Vector with the adjoint states @type N_Vector
     * @param dxB Vector with the adjoint derivative states (only DAE)
     * @type N_Vector
     * @param qBdot Vector with the adjoint quadrature right hand side
     * @type N_Vector
     * @param user_data pointer to temp data object @type TempDat
     * @return status flag indicating successful execution @type int
     */
    int CVodeSolver::fqBdot(realtype t, N_Vector x, N_Vector xB, N_Vector qBdot,
                      void *user_data) {
        Model_ODE *model = (Model_ODE*) user_data;
        model->fdwdp(t,x);
        memset(N_VGetArrayPointer(qBdot),0.0,sizeof(realtype)*model->nJ*model->plist.size());
        realtype *qBdot_tmp = N_VGetArrayPointer(qBdot);
        for(int ip = 1; ip < model->plist.size(); ip++)
            if(model->model_qBdot(&qBdot_tmp[ip*model->nJ],model->plist[ip],t,N_VGetArrayPointer(x),model->p.data(),model->k.data(),model->h.data(),
                               N_VGetArrayPointer(xB),model->w.data(),model->dwdp.data()) != AMICI_SUCCESS)
                return AMICI_ERROR;
        return checkVals(model->nx,N_VGetArrayPointer(qBdot),"adjoint quadrature function");
    }
    
    /** Right hand side of differential equation for state sensitivities sx
     * @param Ns number of parameters
     * @param t timepoint
     * @param x Vector with the states
     * @param dx Vector with the derivative states
     * @param xdot Vector with the right hand side
     * @param ip parameter index
     * @param sx Vector with the state sensitivities
     * @param sdx Vector with the derivative state sensitivities
     * @param sxdot Vector with the sensitivity right hand side
     * @param user_data object with user input @type UserData
     * @param tmp1 temporary storage vector
     * @param tmp2 temporary storage vector
     * @param tmp3 temporary storage vector
     * @return status flag indicating successful execution
     */
    int CVodeSolver::fsxdot(int Ns, realtype t, N_Vector x, N_Vector xdot, int ip,
                      N_Vector sx, N_Vector sxdot, void *user_data,
                      N_Vector tmp1, N_Vector tmp2) {
        Model_ODE *model = (Model_ODE*) user_data;
        if(ip == 0) { // we only need to call this for the first parameter index will be the same for all remaining
            if(model->fdxdotdp(t,x) != AMICI_SUCCESS)
                return AMICI_ERROR;
            if(fJSparse(t,x,nullptr,model->J,model,tmp1,tmp2,nullptr) != AMICI_SUCCESS)
                return AMICI_ERROR;
        }
        memset(N_VGetArrayPointer(sxdot),0.0,sizeof(realtype)*model->nx);
        if(model->model_sxdot(N_VGetArrayPointer(sxdot),t,N_VGetArrayPointer(x),model->p.data(),model->k.data(),model->h.data(),
                        model->plist[ip],N_VGetArrayPointer(sx),
                        model->w.data(),model->dwdx.data(),model->J->data,&model->dxdotdp.at(ip*model->nx)) != AMICI_SUCCESS)
            return AMICI_ERROR;
        return checkVals(model->nx,N_VGetArrayPointer(sxdot),"sensitivity rhs");
    }

CVodeSolver::~CVodeSolver() { AMIFree(); }

} // namespace amici
