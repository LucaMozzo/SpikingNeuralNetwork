#include "stdafx.h"
#include "CppUnitTest.h"
#include "MatrixOps.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;

namespace SNNTest
{		
	TEST_CLASS(MatrixTest)
	{
	public:
		
		TEST_METHOD(TestConvolve)
		{
			vector<vector<double>> arr = vector<vector<double>>(10);


			for (int i = 0; i < 20; ++i) {
				arr[i] = { 1, 2, 3, 4, 5, 6 };
			}

			auto res = MatrixOps::SumColumnsMod(arr, 3);

			if(MatrixOps::Sum(res) != 42)
				Assert::Fail();
		}

	};
}