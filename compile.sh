g++ -fPIC -O3 -c -g QLearn.c
g++ -fPIC -O3 -g *.o -lm -lglut -lGL -lGLU -lX11 -o QLearn

