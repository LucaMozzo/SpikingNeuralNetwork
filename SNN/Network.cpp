#include "stdafx.h"
#include "Network.h"

Network::Network()
{
	inputLayer = new InputLayer();
	outputLayer = new OutputLayer();
}

Network::~Network()
{
	delete inputLayer;
	delete outputLayer;
}
