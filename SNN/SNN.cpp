#include "stdafx.h"
#include "Utils.h"
#include "MatrixOps.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	bool train[] = { 0,1,1,0,1,1 };
	double alpha[] = { 0.1, 0.3, 0.15, 0.17 };
	auto res = MatrixOps::Conv(train, alpha);

	for (int i = 0; i < 5; ++i)
		cout << res[i] << endl;

	return 0;
}