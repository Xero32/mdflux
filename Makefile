
CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -funroll-loops -fopenmp -g

LD = gcc
LDFLAGS = -lm -fopenmp

all: ptensor

ptensor: pressure.o p_struct.o
	$(CC) $(CFLAGS) pressure.o p_struct.o -o ptensor $(LDFLAGS)

pressure.o: pressure.c p_struct.h

p_struct.o: pstruct.c
