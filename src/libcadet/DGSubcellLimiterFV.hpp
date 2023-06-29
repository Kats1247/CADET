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
 * Implements the subcell FV limiting method for collocation DG
 */

#ifndef LIBCADET_DGSubcellLimiterFV_HPP_
#define LIBCADET_DGSubcellLimiterFV_HPP_

#include "AutoDiff.hpp"
#include "MathUtil.hpp"
#include "Memory.hpp"
#include "common/CompilerSpecific.hpp"
#include "cadet/Exceptions.hpp"

#include <algorithm>

namespace cadet
{
	// todo active types required in reconstruction?
	class SlopeLimiterFV {
	public:
		virtual ~SlopeLimiterFV() {}
		virtual double call(double slope0, double slope1) = 0;
		virtual active call(active slope0, active slope1) = 0;
	};

	class NoLimiter : public SlopeLimiterFV {

	public:
		NoLimiter() {}
		double call(double slope0, double slope1) override {
			return slope0;
		}
		active call(active slope0, active slope1) override {
			return slope0;
		}
	};

	class Minmod : public SlopeLimiterFV {

	public:
		Minmod() {}
		double call(double slope0, double slope1) override {
			if (std::signbit(slope0) == std::signbit(slope1))
				return std::copysign(std::min(std::abs(slope0), std::abs(slope1)), slope0);
			else
				return 0.0;
		}
		active call(active slope0, active slope1) override {

			if (std::signbit(static_cast<double>(slope0)) == std::signbit(static_cast<double>(slope1)))
				return abs(static_cast<double>(slope0)) < abs(static_cast<double>(slope1)) ? slope0 : slope1;
			else
				return active(0.0);
		}
	};

	// todo, note: if more slope limiters are added, note that they have to be symmetrical in the current implementation (i.e. in reconstruction function reconstructedUpwindValue)

	/**
	 * @brief Implements the subcell FV limiting scheme for convection
	 * @details //@TODO.
	 */
	class DGSubcellLimiterFV
	{
	public:

		/**
		 * @brief Boundary treatment method determines how the reconstruction handles DG element boundaries.
		 */
		enum class BoundaryTreatment : int
		{
			LimiterSlope = 0, //!< Slope limiter reconstruction using neighbour interface value.
			CentralSlope = 1, //!< Central slope reconstruction with interior information.
			Constant = 2 //!< Constant reconstruction of boundary subcells.
		};

		/**
		 * @brief Creates the subcell Finite Volume scheme
		 */
		DGSubcellLimiterFV() : _LGLweights(nullptr), _LGLnodes(nullptr), _subcellGrid(nullptr) { }

		~DGSubcellLimiterFV() CADET_NOEXCEPT
		{
			delete[] _LGLweights;
			delete[] _LGLnodes;
			delete[] _subcellGrid;
		}

		void init(std::string limiter, const int FVorder, const int  boundaryTreatment, const unsigned int nNodes, double* LGLnodes, double* invWeights, Eigen::MatrixXd modalVanInv, const unsigned int nCells, const unsigned int nComp) {

			_nNodes = nNodes;
			_polyDeg = nNodes - 1;
			_nComp = nComp;

			_LGLweights = new double[nNodes];
			for (int node = 0; node < nNodes; node++)
				_LGLweights[node] = 1.0 / invWeights[node];

			_LGLnodes = new double[nNodes];
			std::copy(LGLnodes, LGLnodes + nNodes, _LGLnodes);

			_subcellGrid = new double[nNodes + 1];
			_subcellGrid[0] = -1.0;
			for (int subcell = 1; subcell < nNodes + 1; subcell++)
				_subcellGrid[subcell] = _subcellGrid[subcell - 1] + _LGLweights[subcell - 1];

			_FVorder = FVorder;
			if (_FVorder == 2) {
				if (limiter == "MINMOD")
					_slope_limiter = std::make_unique<Minmod>();
				else if (limiter == "NONE")
					_slope_limiter = std::make_unique<NoLimiter>();
				else
					throw InvalidParameterException("Subcell FV slope limiter " + limiter + " unknown.");

				switch (boundaryTreatment)
				{
				case static_cast<typename std::underlying_type<BoundaryTreatment>::type>(BoundaryTreatment::LimiterSlope):
					_FVboundaryTreatment = BoundaryTreatment::LimiterSlope;
					break;
				case static_cast<typename std::underlying_type<BoundaryTreatment>::type>(BoundaryTreatment::CentralSlope):
					_FVboundaryTreatment = BoundaryTreatment::CentralSlope;
					break;
				case static_cast<typename std::underlying_type<BoundaryTreatment>::type>(BoundaryTreatment::Constant):
					_FVboundaryTreatment = BoundaryTreatment::Constant;
					break;
				default:
					throw InvalidParameterException("Unknown subcell FV boundary treatment.");
				}
			}
			else if (_FVorder != 1)
				throw InvalidParameterException("Subcell FV order must be 1 or 2, but was specified as " + std::to_string(_FVorder));
		}

		/**
		 * @brief Implements FV (limited) reconstruction
		 * @details TODO
		 * @param [in] leftState left neighbour state
		 * @param [in] centerState current state
		 * @param [in] rightState right neighbour state
		 * @param [in] subcellIdx current (center state) subcell index
		 * @param [in] rightInterface specifies whther right or left interface reconstruction value is returned
		 * @return @c reconstructed FV subcell value at left or right interface
		 */
		// forward backward flow unterscheidung. Check slope computation.
		template<typename StateType>
		StateType reconstructedInterfaceValue(const StateType leftState, const StateType centerState, const StateType rightState, const int subcellIdx, bool rightInterface) {

			if (_FVorder == 1) // No reconstruction
				return centerState;
			else // Order = 2 -> reconstruction
			{
				// TODO what happens here with non-symmetrical slope limiters?
				StateType slope;

				// Boundary cell reconstruction
				if (subcellIdx == 0)
				{
					switch (_FVboundaryTreatment)
					{
					default:
					case BoundaryTreatment::LimiterSlope:
						slope = _slope_limiter->call((rightState - centerState) / (_LGLnodes[1] - _LGLnodes[0]), (rightState - leftState) / (_LGLnodes[1] - _LGLnodes[0]));
						break;

					case BoundaryTreatment::CentralSlope:
						slope = (rightState - centerState) / (_LGLnodes[1] - _LGLnodes[0]);
						break;

					case BoundaryTreatment::Constant:
						return centerState;
					}
				}
				else if (subcellIdx == _nNodes - 1)
				{
					switch (_FVboundaryTreatment)
					{
					default:
					case BoundaryTreatment::LimiterSlope:
						slope = _slope_limiter->call((centerState - leftState) / (_LGLnodes[_nNodes - 1] - _LGLnodes[_nNodes - 2]), (rightState - leftState) / (_LGLnodes[_nNodes - 1] - _LGLnodes[_nNodes - 2]));
						break;

					case BoundaryTreatment::CentralSlope:
						slope = (centerState - leftState) / (_LGLnodes[_nNodes - 1] - _LGLnodes[_nNodes - 2]);
						break;

					case BoundaryTreatment::Constant:
						return centerState;
					}
				}
				else // Inner subcells
					slope = _slope_limiter->call((rightState - centerState) / (_LGLnodes[subcellIdx + 1] - _LGLnodes[subcellIdx]), (centerState - leftState) / (_LGLnodes[subcellIdx] - _LGLnodes[subcellIdx - 1]));

				if (rightInterface == 1)
					return centerState + slope * (_subcellGrid[subcellIdx + 1] - _LGLnodes[subcellIdx]); // Return state at right interface, i.e. forward flow upwind state
				else
					return centerState + slope * (_subcellGrid[subcellIdx] - _LGLnodes[subcellIdx]); // Return state at left interface, i.e. backward flow upwind state
			}
		}

		private:

			std::unique_ptr<SlopeLimiterFV> _slope_limiter; //!< Slope limiter for second order FV reconstruction
			unsigned int _FVorder; //!< FV subcell method order
			BoundaryTreatment _FVboundaryTreatment; //!< boundary treatment of FV subcell method
			double* _LGLweights; //!< DG method LGL weights, i.e. subcell sizes
			double* _LGLnodes; //!< DG method LGL weights, i.e. subcell sizes
			int _nComp; //!< Number of liquid phase components
			int _nNodes; //!< Number of DG nodes per element, i.e. FV subcells
			int _polyDeg; //!< Polynomial degree of DG Ansatz
			double* _subcellGrid; //!< FV subcell grid, i.e. subcell borders
	};

} // namespace cadet

#endif  // LIBCADET_DGSubcellLimiterFV_HPP_
