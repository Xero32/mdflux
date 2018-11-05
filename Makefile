

	gcc -c p_struct.c

	gcc -c pressure.c

	gcc -g ptensor.c pressure.c p_struct.c -o ptensor -lm
