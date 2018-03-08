#pragma once
#include "stdafx.h"
#include "sqlite3.h"
#include "Layer.h"

class DatabaseOps
{
public:

	static void ExportData(InputLayer* inputLayer, OutputLayer* outputLayer);
	static void ImportData(InputLayer* inputLayer, OutputLayer* outputLayer);
};