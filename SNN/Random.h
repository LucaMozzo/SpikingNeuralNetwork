#pragma once
#include "stdafx.h"
#include <time.h>

static class Random
{

public:
	static int random()
	{
		//std::ifstream file("D:\\random.txt");
		//std::string str;
		//std::getline(file, str);
		//int index = atoi(str.c_str());
		//for (int i = 0; i <= index; ++i)
		//	std::getline(file, str); //skip already used

		//std::fstream ffile("D:\\random.txt", std::ios::in | std::ios::out);
		//ffile.seekp(0);
		//ffile << std::to_string(++index);
		//ffile.flush();

		//	
		//return atoi(str.c_str());
		int a = rand() % 10 + 1;
		//std::cout << a << std::endl;
		return a;
	}
};