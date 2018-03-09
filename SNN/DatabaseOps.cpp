#include "stdafx.h"
#include "DatabaseOps.h"

void DatabaseOps::ImportData(InputLayer* inputLayer, OutputLayer* outputLayer)
{
	sqlite3* db;
	auto rc = sqlite3_open("D:\\data.db", &db);
	if (rc)
		return; //failed to connect

	const char *sqlSelect = "SELECT * FROM Layer;";
	char **results = NULL;
	char *error;
	int rows, columns;
	sqlite3_get_table(db, sqlSelect, &results, &rows, &columns, &error);
	if (rc)
	{
		sqlite3_free(error);
	}
	else
	{
		/*TYI = (short)results[5];
		TYO = (short)results[8];*/
	}

	// read alpha weights
	sqlSelect = "SELECT * FROM Weight WHERE LAYERID IN (SELECT ID FROM LAYER WHERE `TYPE`=0);";
	results = NULL;
	sqlite3_get_table(db, sqlSelect, &results, &rows, &columns, &error);
	if (rc)
	{
		sqlite3_free(error);
	}
	else
	{
		short c = 0; //class
		for (int i = 4; i < (rows + 1)*columns; i += columns)
		{
			auto alpha = vector<double>(TYI);
			for (int j = 0; j < TYI; ++j, i += columns)
			{
				alpha[j] = atof(results[i]);
			}
			i -= columns;
			inputLayer->alphas[c++] = alpha;
		}
	}

	// read beta and gamma weights
	sqlSelect = "SELECT * FROM Weight WHERE LAYERID IN (SELECT ID FROM LAYER WHERE `TYPE`=2);";
	results = NULL;
	sqlite3_get_table(db, sqlSelect, &results, &rows, &columns, &error);
	if (rc)
	{
		sqlite3_free(error);
	}
	else
	{
		short c = 0; //class
		for (int i = 4; i < (rows+1)*columns; i+=columns)
		{
			auto beta = vector<double>(TYO);
			for (int j = 0; j < TYO; ++j, i += columns)
			{
				beta[j] = atof(results[i]);
			}
			outputLayer->betas[c] = beta;
			outputLayer->gammas[c++] = atof(results[i]);
		}
	}
	sqlite3_close(db);
}
