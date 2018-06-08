#pragma once
#include "sqlite3.h"
#include "Layer.h"
/**
This class contains methods to perform I/O operations on a SQLite 3 database
*/
class DatabaseOps
{
public:
	/**
	Generates a SQLite database at the specified path and exports all the weights and hyperparameters to it
	@param inputLayer The input layer
	@param outputLayer The output layer
	@param fileName The output file name
	*/
	static void ExportData(InputLayer* inputLayer, OutputLayer* outputLayer, string fileName);
	/**
	Reads and imports the weights of the network from the specified SQLite database
	@param inputLayer The input layer
	@param outputLayer The output layer
	@param fileName The input file name
	*/
	static void ImportData(InputLayer* inputLayer, OutputLayer* outputLayer, string fileName);
};