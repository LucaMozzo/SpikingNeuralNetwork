#include "Network.h"
#include "Utils.h"
#include <time.h>	

using namespace std;
/*! \file */

/**
This is where your program should be written
*/
int main(int argc, char** argv)
{
	srand(time(NULL));

	//create a network
	auto n = Network();
	//only train on the first 5 digits
	//array<unsigned char, 5> filter = { 0, 1, 2, 3, 4};
	//n.ExportData("data.db");
	//n.Train<0>(200, 60000, nullptr, nullptr, 1000);
	//export the training data
	n.TrainVal(200, 1000, 1000, true);
	//n.ExportData("dataT4.db");
	//validate against test dataset
	//n.Validate<0>(1000, false, nullptr);
	//n.CrossValidate();

	Utils::PrintLine("Done");
	return 0;
}
