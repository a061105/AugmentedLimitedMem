#pragma once

#include <deque>
#include <sstream>
#include <string>
#include "util.h"

class OptimizerState;

struct TerminationCriterion {
	virtual double GetValue(const OptimizerState& state, std::ostream& message) = 0;
	virtual ~TerminationCriterion() { }
};

class RelativeMeanImprovementCriterion : public TerminationCriterion {
	const Int numItersToAvg;
	std::deque<double> prevVals;

public:
	RelativeMeanImprovementCriterion(Int numItersToAvg = 5) : numItersToAvg(numItersToAvg) {}

	double GetValue(const OptimizerState& state, std::ostream& message);
};
