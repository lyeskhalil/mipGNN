#*   666    54      integral     0     8342.0000    10043.0000    74521   20.39%
#Found incumbent of value 8342.000000 after 3.45 sec. (2557.83 ticks)

def parse_cplex_log(logstring):
	incumbent_list = []
	incumbent_str = ""
	lines = logstring.splitlines()

	num_nodes = 0
	is_heuristic_solution = 1
	
	for line in lines:
		if len(line) > 0 and line[0] == '*':
			line_vals = line.split()
			num_nodes = line_vals[1]
			is_heuristic_solution = 0
			if line_vals[1][-1] == '+':
				num_nodes = num_nodes[:-1]
				is_heuristic_solution = 1
			num_nodes = int(num_nodes)

		elif "Found" in line:
			line_vals = line.split()

			incumbent_list += [[num_nodes, -1, -1]]
			incumbent_list[-1][1] = float(line_vals[6])
			incumbent_list[-1][2] = float(line_vals[4])

			incumbent_str += str(num_nodes) + ','
			incumbent_str += line_vals[6] + ','
			incumbent_str += line_vals[4] + ','
			incumbent_str += str(is_heuristic_solution) + '\n'

	return incumbent_list, incumbent_str
