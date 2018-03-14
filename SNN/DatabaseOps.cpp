#include "stdafx.h"
#include "DatabaseOps.h"

[[deprecated]]
bool ExecuteNonQuery(sqlite3* db, const char* query)
{
	sqlite3_stmt* stmt = nullptr;
	auto res = sqlite3_prepare_v2(db, query, -1, &stmt, nullptr);

	if (res != SQLITE_OK)
		//error
		return false;

	res = sqlite3_step(stmt);
	while (res != SQLITE_DONE)
		res = sqlite3_step(stmt);

	sqlite3_finalize(stmt);
	return true;
}

void DatabaseOps::ExportData(InputLayer * inputLayer, OutputLayer * outputLayer, string fileName)
{
	/*
	Generates a sqlite database at the specified path.

	Tha database contains different tables:

	|-------------------------------------------------------------------------------------------------------------|
	| LAYER                                                                                                       |
	|-------------------------------------------------------------------------------------------------------------|
	| ID         | TYPE                                    | TY                                                   |
	| unique id  | type of layer 0=input 1=hidden 2=output | size of alpha or beta (depends on the type of layer) |
	|-------------------------------------------------------------------------------------------------------------|

	|-----------------------------------------------------------------------------|
	| WEIGHT                                                                      |
	|-----------------------------------------------------------------------------|
	| ID         | VALUE               | LAYERID                                  |
	| unique id  | value of the weight | id of the layer where the weight belongs |
	|-----------------------------------------------------------------------------|

	*/

	//delete the file regardless if it exists or not
	remove(fileName.c_str());

	sqlite3* db;
	auto res = sqlite3_open(fileName.c_str(), &db);
	if (res)
		return; //failed to connect
	
	sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, NULL);

	//create tables
	const char* query = "CREATE TABLE Layer (ID INTEGER PRIMARY KEY AUTOINCREMENT, `TYPE` INTEGER DEFAULT 0 NOT NULL," \
		 "TY TINYINT NOT NULL, CHECK (TYPE >= 0 AND TYPE <= 2));";

	sqlite3_exec(db, query, NULL, NULL, NULL);

	query = "CREATE TABLE Weight (ID INTEGER PRIMARY KEY AUTOINCREMENT, `VALUE` DOUBLE NULL DEFAULT NULL, LAYERID INTEGER," \
		" FOREIGN KEY (LAYERID) REFERENCES Layer(ID));";

	sqlite3_exec(db, query, NULL, NULL, NULL);
	sqlite3_exec(db, "END TRANSACTION;", NULL, NULL, NULL);

	//insert the data
	query = ("INSERT INTO Layer (`TYPE`, TY) VALUES(0, " + std::to_string(TYI) + ")").c_str();
	sqlite3_exec(db, query, NULL, NULL, NULL);
	query = ("INSERT INTO Layer (`TYPE`, TY) VALUES(2, " + std::to_string(TYO) + ")").c_str();
	sqlite3_exec(db, query, NULL, NULL, NULL);

	sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, NULL);
	for(auto& alpha : inputLayer->alphas)
		for(char i = 0; i < TYI; ++i)
		{
			query = ("INSERT INTO Weight (`VALUE`, LAYERID) VALUES (" + std::to_string(alpha[i]) + ", 1);").c_str();
			sqlite3_exec(db, query, NULL, NULL, NULL);
		}
	sqlite3_exec(db, "END TRANSACTION;", NULL, NULL, NULL);

	// beta and gamma
	sqlite3_exec(db, "BEGIN TRANSACTION;", NULL, NULL, NULL);
	for(char i = 0; i < CLASSES; ++i)
	{
		for(char j = 0; j < TYO; ++j)
		{
			query = ("INSERT INTO Weight (`VALUE`, LAYERID) VALUES (" + std::to_string(outputLayer->betas[i][j]) + ", 2);").c_str();
			sqlite3_exec(db, query, NULL, NULL, NULL);
		}
		query = ("INSERT INTO Weight (`VALUE`, LAYERID) VALUES (" + std::to_string(outputLayer->gammas[i]) + ", 2);").c_str();
		sqlite3_exec(db, query, NULL, NULL, NULL);
	}
	sqlite3_exec(db, "END TRANSACTION;", NULL, NULL, NULL);

	sqlite3_close(db);
}

void DatabaseOps::ImportData(InputLayer* inputLayer, OutputLayer* outputLayer, string fileName)
{
	sqlite3* db;
	const auto res = sqlite3_open(fileName.c_str(), &db);
	if (res)
		return; //failed to connect

	char **results = nullptr;
	char *error;
	int rows, columns;
	sqlite3_get_table(db, "SELECT * FROM Layer;", &results, &rows, &columns, &error);
	if (res)
	{
		sqlite3_free(error);
	}
	else
	{
		/*TYI = (short)results[5];
		TYO = (short)results[8];*/
	}

	// read alpha weights
	results = nullptr;
	sqlite3_get_table(db, "SELECT * FROM Weight WHERE LAYERID IN (SELECT ID FROM LAYER WHERE `TYPE`=0);", &results, &rows, &columns, &error);
	if (res)
	{
		sqlite3_free(error);
	}
	else
	{
		short c = 0; //class
		for (int i = 4; i < (rows + 1)*columns; i += columns)
		{
			auto alpha = array<double, TYI>();
			for (int j = 0; j < TYI; ++j, i += columns)
			{
				alpha[j] = atof(results[i]);
			}
			i -= columns;
			inputLayer->alphas[c++] = alpha;
		}
	}

	// read beta and gamma weights
	results = nullptr;
	sqlite3_get_table(db, "SELECT * FROM Weight WHERE LAYERID IN (SELECT ID FROM LAYER WHERE `TYPE`=2);", &results, &rows, &columns, &error);
	if (res)
	{
		sqlite3_free(error);
	}
	else
	{
		short c = 0; //class
		for (int i = 4; i < (rows+1)*columns; i+=columns)
		{
			auto beta = array<double,TYO>();
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
