
#include <include/symbolic_functions.h>
#include <string.h>
#include <include/udata.h>
#include <include/tdata.h>
#include <include/rdata.h>
#include "model_neuron_o2_w.h"

int z_model_neuron_o2(realtype t, int ie, N_Vector x, void *user_data, TempData *tdata, ReturnData *rdata) {
int status = 0;
UserData *udata = (UserData*) user_data;
realtype *x_tmp = N_VGetArrayPointer(x);
status = w_model_neuron_o2(t,x,NULL,user_data);
    switch(ie) { 
        case 0: {
  rdata->z[tdata->nroots[ie]+udata->nmaxevent*0] = t;
  rdata->z[tdata->nroots[ie]+udata->nmaxevent*1] = -x_tmp[2]/(udata->k[1]+x_tmp[0]*5.0-x_tmp[1]+(x_tmp[0]*x_tmp[0])*(1.0/2.5E1)+1.4E2);
  rdata->z[tdata->nroots[ie]+udata->nmaxevent*2] = -x_tmp[4]/(udata->k[1]+x_tmp[0]*5.0-x_tmp[1]+(x_tmp[0]*x_tmp[0])*(1.0/2.5E1)+1.4E2);
  rdata->z[tdata->nroots[ie]+udata->nmaxevent*3] = -x_tmp[6]/(udata->k[1]+x_tmp[0]*5.0-x_tmp[1]+(x_tmp[0]*x_tmp[0])*(1.0/2.5E1)+1.4E2);
  rdata->z[tdata->nroots[ie]+udata->nmaxevent*4] = -x_tmp[8]/(udata->k[1]+x_tmp[0]*5.0-x_tmp[1]+(x_tmp[0]*x_tmp[0])*(1.0/2.5E1)+1.4E2);

        } break;

    } 
return(status);

}

