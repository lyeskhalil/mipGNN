METHOD=$1
INSTANCE_DIR=$2
TIMELIMIT=$3
BAREBONES=$4

sbatch graham.sh default_emptycb er_200_SET2_1k 600 1
sbatch graham.sh node_selection er_200_SET2_1k 600 1

sbatch graham.sh node_selection er_200_SET2_1k 600 0
sbatch graham.sh default_emptycb er_200_SET2_1k 600 0

sbatch graham.sh local_branching_exact er_200_SET2_1k 600 0
sbatch graham.sh local_branching_exact er_200_SET2_1k 600 1

sbatch graham.sh default er_200_SET2_1k 600 0
sbatch graham.sh default er_200_SET2_1k 600 1

for FILENAME in OUTPUT/er_200_SET2_1k/default_emptycb/barebones_1/*.out; do tail -1 $FILENAME; done > agg_default_emptycb_barebones.out