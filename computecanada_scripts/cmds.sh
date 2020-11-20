METHOD=$1
INSTANCE_DIR=$2
TIMELIMIT=$3
BAREBONES=$4
MODEL_PATH=$5

# ../gnn_models/er_SET1_400_400_alpha_0_25_setParam_100_train  
# ../gnn_models/er_SET1_400_400_alpha_0_5_setParam_100_train   
# ../gnn_models/er_SET2_300_300_alpha_0_25_setParam_100_train
# ../gnn_models/er_SET1_400_400_alpha_0_75_setParam_100_train  
# ../gnn_models/er_SET2_300_300_alpha_0_5_setParam_100_train
# ../gnn_models/er_SET2_200_200_alpha_0_25_setParam_100_train  
# ../gnn_models/er_SET2_300_300_alpha_0_75_setParam_100_train
# ../gnn_models/er_SET2_200_200_alpha_0_5_setParam_100_train
../gnn_models/er_SET2_200_200_alpha_0_75_setParam_100_train
../gnn_models/er_SET2_200_200_alpha_0_75_setParam_100_test

sbatch graham.sh default_emptycb er_200_SET2_1k 600 1
sbatch graham.sh node_selection er_200_SET2_1k 600 1

sbatch graham.sh node_selection er_200_SET2_1k 600 0
sbatch graham.sh default_emptycb er_200_SET2_1k 600 0

sbatch graham.sh local_branching_exact er_200_SET2_1k 600 0
sbatch graham.sh local_branching_exact er_200_SET2_1k 600 1

sbatch graham.sh default er_200_SET2_1k 600 0
sbatch graham.sh default er_200_SET2_1k 600 1

#for FILENAME in OUTPUT/er_200_SET2_1k/default_emptycb/barebones_1/*.out; do tail -1 $FILENAME; done > agg_default_emptycb_barebones.out