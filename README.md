For gtx 1650
nvcc -O3 -arch=sm_75 -std=c++17 -Xcompiler -Wall -o lbm gpumain.cu -lsfml-graphics -lsfml-window -lsfml-system
