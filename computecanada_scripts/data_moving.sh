# step 1: log in to graham.computecanada.ca
...
# step 2: go to data sharing directory
cd ~/projects/def-khalile2/data_sharing/mipGNN/
# step 3: copy new data using rsync to another server
rsync -a --files-from=:./data_graphsonly_paths.txt . /path/to/cerc/server/data/dir/
# step 4: log in to cerc server
...
cd /path/to/cerc/server/data/dir/
# step 4: uncompress all tar.gz files inside directory data_graphsonly/
find data_graphsonly/ -type f -print -exec tar -xzf {} \;
