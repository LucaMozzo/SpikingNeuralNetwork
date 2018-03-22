// TestPerf.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <array>
#include <Windows.h>
#include<time.h>

using namespace std;

array<double, 8-1> Conv(array<bool, 8> const & f, array<double, 3> const & g)
{
	int const out_size = 8 - 1;
	array<double, 8 - 1> out{};
	for (auto i(0); i < out_size; ++i) {
		int const jmn = (i >= 3 - 1) ? i - (3 - 1) : 0;
		int const jmx = (i <  out_size) ? i : out_size;
		for (auto j(jmn); j <= jmx; ++j) {
			out[i] += (f[j] * g[i - j]);
		}
	}
	return out;
}

int main()
{
	LARGE_INTEGER beginTime;
	QueryPerformanceCounter(&beginTime);
	

	LARGE_INTEGER endTime;
	QueryPerformanceCounter(&endTime);
	LARGE_INTEGER timerFreq;
	QueryPerformanceFrequency(&timerFreq);
	double freq = 1.0f / timerFreq.QuadPart;

	double timeSeconds = (endTime.QuadPart - beginTime.QuadPart) * freq;
	cout << "\n\nVal: " << timeSeconds << endl;


	getchar();
	return 0;
}

