g++ -O3 SNN.cpp Layer.cpp DatabaseOps.cpp Network.cpp Utils.cpp MatrixOps.cpp -l sqlite3 -o output `pkg-config --cflags --libs opencv`
