import numpy as np
import sys

if len(sys.argv) != 3:
	print ("Usage: inputgen.py <out_filename> <number of matrices>")
	exit()

name = sys.argv[1]
amount = int(sys.argv[2])

out = open(name, "w")

out.write(str(amount) + "\n")

for i in range(amount):
	out.write("***\n")
	m = np.random.randint(255, size=[3,3])
	for line in m:
		for ele in line:
			out.write(str(ele)+" ")
		out.write("\n")

out.write("***\n")

