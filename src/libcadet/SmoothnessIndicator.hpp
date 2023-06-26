// =============================================================================
//  CADET
//  
//  Copyright © 2008-2022: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

/**
 * @file
 * Implements smooothness indicators used in oscillation suppression for DG
 */


#ifndef LIBCADET_SMOOTHNESSINDICATOR_HPP_
#define LIBCADET_SMOOTHNESSINDICATOR_HPP_

#include "AutoDiff.hpp"
#include <Eigen/Dense>

namespace cadet
{

	class SmoothnessIndicator {
	public:
		virtual ~SmoothnessIndicator() {}
		virtual double calcSmoothness(const double* const localC, const int strideNode, const int strideLiquid) = 0;
		virtual double calcSmoothness(const active* const localC, const int strideNode, const int strideLiquid) = 0;
	};

	class PolynomialEnergyIndicator : public SmoothnessIndicator {

	public:

		PolynomialEnergyIndicator(int polyDeg, const Eigen::MatrixXd& modalVanInv)
			: _polyDeg(polyDeg), _modalVanInv(modalVanInv)
		{
			_nNodes = _polyDeg + 1;
			_modalCoeff.resize(_nNodes);
			_modalCoeff.setZero();
		}

		double calcSmoothness(const double* const localC, const int strideNode, const int strideLiquid) override
		{
			// todo? limit maximum of blending coefficient?
			// todo minimal threshold below which no FV subcell is added

			// TODO: how to choose these constants, store them.
			double energyThreshold = 1e-5;
			const double _s = 9.2102;

			Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<Eigen::Dynamic>> _C(localC, _nNodes, Eigen::InnerStride<Eigen::Dynamic>(strideNode));
			_modalCoeff = _modalVanInv * _C;

			double modalEnergySquareSum = _modalCoeff.cwiseAbs2().sum();
			double hm = (modalEnergySquareSum < energyThreshold) ? 0.0 : std::pow(_modalCoeff[_polyDeg], 2.0) / modalEnergySquareSum;

			modalEnergySquareSum = _modalCoeff.segment(0, _polyDeg).cwiseAbs2().sum();
			hm = std::max(hm, (modalEnergySquareSum < 1e-10) ? 0.0 : std::pow(_modalCoeff[_polyDeg - 1], 2.0) / modalEnergySquareSum);


			const double _T = 0.5 * std::pow(10.0, -1.8 * std::pow(static_cast<double>(_nNodes), 0.25)); // 10-1.8(N + 1)0.2; // todo store this constant
			return 1.0 / (1.0 + exp(-_s / _T * (hm - _T)));
		}
		double calcSmoothness(const active* const localC, const int strideNode, const int strideLiquid) override
		{
			// todo? limit maximum of blending coefficient?
			// todo minimal threshold below which no FV subcell is added

			// TODO: how to choose these constants, store them.
			double energyThreshold = 1e-5;
			const double _s = 9.2102;

			Eigen::Map<const Eigen::Vector<active, Eigen::Dynamic>, 0, Eigen::InnerStride<Eigen::Dynamic>> _C(localC, _nNodes, Eigen::InnerStride<Eigen::Dynamic>(strideNode));
			_modalCoeff = _modalVanInv * _C.template cast<double>();

			double modalEnergySquareSum = _modalCoeff.cwiseAbs2().sum();
			double hm = (modalEnergySquareSum < energyThreshold) ? 0.0 : std::pow(_modalCoeff[_polyDeg], 2.0) / modalEnergySquareSum;

			modalEnergySquareSum = _modalCoeff.segment(0, _polyDeg).cwiseAbs2().sum();
			hm = std::max(hm, (modalEnergySquareSum < 1e-10) ? 0.0 : std::pow(_modalCoeff[_polyDeg - 1], 2.0) / modalEnergySquareSum);


			const double _T = 0.5 * std::pow(10.0, -1.8 * std::pow(static_cast<double>(_nNodes), 0.25)); // 10-1.8(N + 1)0.2; // todo store this constant
			return 1.0 / (1.0 + exp(-_s / _T * (hm - _T)));
		}

	private: 

		int _nNodes;
		int _polyDeg;

		Eigen::VectorXd _modalCoeff;
		Eigen::MatrixXd _modalVanInv;

	};

}

#endif // LIBCADET_SMOOTHNESSINDICATOR_HPP_
