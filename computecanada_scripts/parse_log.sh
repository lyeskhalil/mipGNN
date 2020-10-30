find CLEAN_OUTPUT/ -name '*.out' | xargs sed -ni '/solving stats/,$p'

sed -n '/solving stats/,$p' OUTPUT/er_200_SET2_1k/node_selection/barebones_1/er_n=200_m=1889_p=0.10_SET2_setparam=100.00_alpha=0.75_97.out