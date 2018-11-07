gcc -c md_struct.c
gcc -c -Wall -fopenmp gofr.c
gcc -g -Wall -fopenmp gofr.c md_struct.c -o gofr -lm -fopenmp
