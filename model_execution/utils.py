#*   666    54      integral     0     8342.0000    10043.0000    74521   20.39%
#Found incumbent of value 8342.000000 after 3.45 sec. (2557.83 ticks)

def parse_cplex_log(logstring):
	incumbent_list = []
	lines = logstring.splitlines()
	for line in lines:
		if len(line) > 0 and line[0] == '*':
			line_vals = line.split()
			incumbent_list += [[int(line_vals[1]), -1, -1]]

		elif "Found" in line:
			line_vals = line.split()
			incumbent_list[-1][1] = float(line_vals[6])
			incumbent_list[-1][2] = float(line_vals[4])

	return incumbent_list
