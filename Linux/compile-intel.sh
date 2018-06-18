source /opt/intel/bin/compilervars.sh intel64
icc Layer.cpp MatrixOps.cpp SNN.cpp DatabaseOps.cpp Utils.cpp Network.cpp -O2 -lsqlite3
