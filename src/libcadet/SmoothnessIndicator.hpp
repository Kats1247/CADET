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
		// todo: no templates for virtual functions. Is there another way to overload?
		virtual double calcSmoothness(const double* const localC, const int strideNode, const int strideLiquid, const int cellIdx) = 0;
		virtual double calcSmoothness(const active* const localC, const int strideNode, const int strideLiquid, const int cellIdx) = 0;
	};

	class PolynomialEnergyIndicator : public SmoothnessIndicator {

	public:

		PolynomialEnergyIndicator(int polyDeg, const Eigen::MatrixXd& modalVanInv, double modalCoefThreshold, double nodalCoefThreshold, double nodalCoefShift)
			: _polyDeg(polyDeg), _modalVanInv(modalVanInv), _modalCoefThreshold(modalCoefThreshold), _nodalCoefThreshold(nodalCoefThreshold), _nodalCoefShift(nodalCoefShift)
		{
			_nNodes = _polyDeg + 1;
			_modalCoeff.resize(_nNodes);
			_modalCoeff.setZero();
		}

		double calcSmoothness(const double* const localC, const int strideNode, const int strideLiquid, const int cellIdx) override
		{

			// TODO: how to choose these constants, store them.
			const double _s = 9.2102;

			Eigen::Map<const Eigen::VectorXd, 0, Eigen::InnerStride<Eigen::Dynamic>> _C(localC, _nNodes, Eigen::InnerStride<Eigen::Dynamic>(strideNode));

			if (_C.cwiseAbs().maxCoeff() < _nodalCoefThreshold)
				return 0.0;
			else
				_modalCoeff = _modalVanInv * (_C + _nodalCoefShift * Eigen::VectorXd::Ones(_nNodes));


			double modalEnergySquareSum = _modalCoeff.cwiseAbs2().sum();
			double hm = (modalEnergySquareSum <= _modalCoefThreshold) ? 0.0 : std::pow(_modalCoeff[_polyDeg], 2.0) / modalEnergySquareSum;

			modalEnergySquareSum = _modalCoeff.segment(0, _polyDeg).cwiseAbs2().sum();
			hm = std::max(hm, (modalEnergySquareSum <= _modalCoefThreshold) ? 0.0 : std::pow(_modalCoeff[_polyDeg - 1], 2.0) / modalEnergySquareSum);


			const double _T = 0.5 * std::pow(10.0, -1.8 * std::pow(static_cast<double>(_nNodes), 0.25)); // todo store this constant
			return 1.0 / (1.0 + exp(-_s / _T * (hm - _T)));
		}
		double calcSmoothness(const active* const localC, const int strideNode, const int strideLiquid, const int cellIdx) override
		{

			// TODO: how to choose these constants, store them.
			const double _s = 9.2102;

			Eigen::Map<const Eigen::Vector<active, Eigen::Dynamic>, 0, Eigen::InnerStride<Eigen::Dynamic>> _C(localC, _nNodes, Eigen::InnerStride<Eigen::Dynamic>(strideNode));
			
			if (_C.template cast<double>().cwiseAbs().maxCoeff() < _nodalCoefThreshold)
				return 0.0;
			else
				_modalCoeff = _modalVanInv * (_C.template cast<double>() + _nodalCoefShift * Eigen::VectorXd::Ones(_nNodes));

			double modalEnergySquareSum = _modalCoeff.cwiseAbs2().sum();
			double hm = (modalEnergySquareSum < _modalCoefThreshold) ? 0.0 : std::pow(_modalCoeff[_polyDeg], 2.0) / modalEnergySquareSum;

			modalEnergySquareSum = _modalCoeff.segment(0, _polyDeg).cwiseAbs2().sum();
			hm = std::max(hm, (modalEnergySquareSum < _modalCoefThreshold) ? 0.0 : std::pow(_modalCoeff[_polyDeg - 1], 2.0) / modalEnergySquareSum);


			const double _T = 0.5 * std::pow(10.0, -1.8 * std::pow(static_cast<double>(_nNodes), 0.25)); // 10-1.8(N + 1)0.2; // todo store this constant
			return 1.0 / (1.0 + exp(-_s / _T * (hm - _T)));
		}

	private: 

		int _nNodes; //!< Number of polynomial interpolation nodes of DG ansatz
		int _polyDeg; //!< Polynomial degree of DG ansatz

		double _modalCoefThreshold; //!< Threshold for modal polynomial coefficients / modal energy
		double _nodalCoefThreshold; //!< Threshold for nodal polynomial coefficients
		double _nodalCoefShift; //!< Shift for nodal polynomial coefficients (To get c(z) >= eps > 0 at constant 0 concentration values)
		Eigen::VectorXd _modalCoeff; //!< Modal polynomial coefficient memory buffer
		Eigen::MatrixXd _modalVanInv; //!< Inverse Vandermonde matrix of modal polynomial basis

	};

}

#endif // LIBCADET_SMOOTHNESSINDICATOR_HPP_
