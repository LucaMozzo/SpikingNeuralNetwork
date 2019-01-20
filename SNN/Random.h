#pragma once
#include "stdafx.h"

class Random
{
protected:
	static int index;
	static std::vector<std::array<bool, LFSR_SEQ_LENGTH>> sequence;
public:
	/**
	Initialise the LFSR
	*/
	static void InitLFSR();
	/**
	Generate a random number using the previously generated LFSR
	@returns The random number that has been generated
	*/
	static float Generate();
};
