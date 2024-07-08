for train in 1
do
for data in traffic
do
# for model in GRU Transformer DLinear
for model in Transformer
do


python  run_longExp.py \
        --model $model \
        --data $data \
        --data_path $data'.csv' \
        --target 0 \
        --is_training $train \
        --model_id model_$model'_data_'$data


done
done
done

