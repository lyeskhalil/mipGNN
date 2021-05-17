#*   666    54      integral     0     8342.0000    10043.0000    74521   20.39%
#Found incumbent of value 8342.000000 after 3.45 sec. (2557.83 ticks)

def parse_cplex_log(logstring, time_offset=0.0):
    incumbent_str = ""
    lines = logstring.splitlines()

    num_nodes = 0
    is_heuristic_solution = 1
    
    for line in lines:
        if len(line) > 0 and line[0] == '*':
            line_vals = line.split()
            num_nodes = line_vals[1]
            is_heuristic_solution = 0
            plus_idx = num_nodes.find('+')
            if plus_idx != -1:
                num_nodes = num_nodes[:plus_idx]
                is_heuristic_solution = 1
            try:
                num_nodes = int(num_nodes)
            except ValueError:
                num_nodes = -1

        elif "Found" in line:
            line_vals = line.split()
            is_mip_start = (line_vals[8] == 'mipstart')
            is_heuristic_solution = 2 if is_mip_start else is_heuristic_solution

            objval, timing = float(line_vals[4]), float(line_vals[6])

            #len(indices_integer), threshold, frac_variables[idx]
            mipstart_fixedvars, mipstart_threshold, mipstart_fracvars = -1, -1, -1
            if is_mip_start:
                mipstart_fixedvars, mipstart_threshold, mipstart_fracvars = int(line_vals[9]), float(line_vals[10]), float(line_vals[11])
            else:
                timing += time_offset
            
            incumbent_str_cur = "%d,%s,%s,%d,%d,%g,%g\n" % (num_nodes, timing, objval, is_heuristic_solution, mipstart_fixedvars, mipstart_threshold, mipstart_fracvars)
            incumbent_str += incumbent_str_cur
    return incumbent_str

