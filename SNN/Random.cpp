#include "stdafx.h"
#include "Utils.h"
#include "Random.h"

int Random::index;
std::vector<array<bool, LFSR_SEQ_LENGTH>> Random::sequence;

void Random::InitLFSR()
{
	index = 0;

	const array<bool, LFSR_SEQ_LENGTH> seed = { /*1, 0, 1, 1, 0, 1, 0, 1, 1, 0*/ };
	const array<int, TAP_LENGTH> tap = { 9, 2 };

	sequence = Utils::LFSR(seed, tap);
}

float Random::Generate()
{
	if (index == sequence.size())
		index = 0;
	//LFSR_SEQ_LENGTH - 8 because it must be 8 bit, everything else ignored
	return Utils::BinaryToDec(sequence[index++], LFSR_SEQ_LENGTH - 8);
}
