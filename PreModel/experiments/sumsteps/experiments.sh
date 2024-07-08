# 10 min

for train in 0
do
for data in sumsteps
do
for model in GRU Transformer DLinear
do


python  run_longExp.py \
        --model $model \
        --data $data \
        --data_path $data'.csv' \
        --target steps \
        --is_training $train \
        --model_id model_$model'_data_'$data


done
done
done

