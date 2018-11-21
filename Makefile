
RM = rm -f
CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -funroll-loops -fopenmp
LDFLAGS = -lm -fopenmp

all: gofr ptensor

gofr: gofr.o md_struct.o
	$(CC) $(CFLAGS) gofr.o md_struct.o -o gofr $(LDFLAGS)

gofr.o: gofr.c md_struct.h

ptensor: ptensor.o pressure.o md_struct.o
	$(CC) $(CFLAGS) ptensor.o pressure.o md_struct.o -o ptensor $(LDFLAGS)
	
ptensor.o: ptensor.c pressure.h md_struct.h

pressure.o: pressure.c md_struct.h

clean:
	$(RM) *.o gofr ptensor
