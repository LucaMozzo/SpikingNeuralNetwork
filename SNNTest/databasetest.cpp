#include "stdafx.h"
#include "CppUnitTest.h"
#include "../SNN/DatabaseOps.h"
#include "../SNN/Network.h"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace std;

namespace SNNTest
{
	TEST_CLASS(DatabaseTest)
	{
	public:

		TEST_METHOD(ImportExport)
		{
			Network n = Network();
			n.ExportData();
		}

	};
}