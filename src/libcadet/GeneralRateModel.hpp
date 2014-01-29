// =============================================================================
//  CADET - The Chromatography Analysis and Design Toolkit
//  
//  Copyright © 2008-2014: Eric von Lieres¹, Joel Andersson,
//                         Andreas Puettmann¹, Sebastian Schnittert¹,
//                         Samuel Leweke¹
//                                      
//    ¹ Forschungszentrum Juelich GmbH, IBG-1, Juelich, Germany.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#ifndef GENERALRATEMODEL_HPP_
#define GENERALRATEMODEL_HPP_

#include "ChromatographyModel.hpp"
#include "JacobianData.hpp"

namespace cadet {

#define CADET_STRICT true
#define CADET_LOOSE false

class GeneralRateModel : public ChromatographyModel
{
public:

    GeneralRateModel(SimulatorPImpl& sim);
    virtual ~GeneralRateModel();

    ///todo Check destructors and other stuff... rule of three, for all classes!

    int residualDae(double t, N_Vector NV_y, N_Vector NV_yDot, N_Vector NV_res, void* userData);
    int residualSens(int ns, double t, N_Vector NV_y, N_Vector NV_yDot, N_Vector NV_res,
            N_Vector* NV_yS, N_Vector* NV_ySDot, N_Vector* NV_resS,
            void* userData, N_Vector NV_tmp1, N_Vector NV_tmp2, N_Vector NV_tmp3);

    void calcIC(const double t);
    void calcICSens(const double t);

    void specialSetup();

private:

    // this residual function now only handles column and particles
    // boundaries are treated differently elsewhere
    template <typename StateType, typename ResidType, typename ParamType, bool wantJac>
    int residualColumnParticle(const double t, const StateType* y, const double* ydot, ResidType* res) throw (CadetException);


    //       t     input       current time
    //       y     input       [array] pointer to first column element of state vector
    //      yp     input       [array] pointer to first column element of derivative state vector
    //       p     input       [scalar] pointer to parameter data structure containing all model parameters (sensisitve as well as non-sensitive)
    //     res    output       [array] calculated residual values
    //  csdata                 ChromsimData structure
    //
    template <typename StateType, typename ResidType, typename ParamType, bool wantJac>
    int residualColumn(const double t, const StateType* y, const double* ydot, ResidType* res) throw (CadetException);

    template <typename StateType, typename ResidType, typename ParamType, bool wantJac>
    int residualParticle(const double t, const int pblk, const StateType* y, const double* ydot, ResidType* res) throw (CadetException);

    template <typename ResidType, typename ParamType>
    int residualBoundaries(const double* y, const double* ydot, ResidType* res) throw (CadetException);

    void assembleOffdiagJac() throw (CadetException);

    template <typename ParamType>
    void setInletParamDerivatives(std::vector<ParamType>& concInlet);


    SimulatorPImpl&         _psim;
    const JacobianData&     _jac;
    const WenoScheme&       _ws;
    std::vector<double>     _c_in;
    std::vector<std::vector<double> > _dc_indp;

    void dFdy_times_s(N_Vector NV_s, N_Vector NV_ret);
    void dFdyDot_times_sDot(N_Vector NV_sDot, N_Vector NV_ret);

};

} // namespace cadet


#endif /* GENERALRATEMODEL_HPP_ */
