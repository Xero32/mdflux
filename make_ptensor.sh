
gcc -c md_struct.c
gcc -c pressure.c
gcc -g ptensor.c pressure.c md_struct.c -o ptensor -lm
