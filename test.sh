source env/bin/activate

. $1

python src/test.py $dataset $data_path $bz $out_size $nlayers $hidden $dropout $loss $swap $margin $epochs $lr $momentum $decay $schedule $gamma $seed $save $load $test $early_stop $prefetch $ngpu $log $log_interval

deactivate
