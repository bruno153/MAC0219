import numpy as np
import sys

if len(sys.argv) != 4:
	print ("Usage: sequential.py <N> <k> <M>")
	exit()


def f (x, k, M):
	PI = 3.1415926535897932384626433
	return (np.sin((2*M + 1)*PI*x)*np.cos(2*PI*k*x))/np.sin(PI*x);
}

N = int(sys.argv[1])
k = float(sys.argv[2])
M = float(sys.argv[3])
i = 0;
f = 0;
f2 = 0;

while i < N:
	x = np.random.uniform(0, 0.5)
	x = f(x, k, M)
	f += x
	f2 += x*x

error = np.sqrt((f2 - f*f)/N)
print("Resultado para python: " + str(f))
print("Erro para python: " + str(error))


