
# # household_power_consumption
# https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set

# Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years. Different electrical quantities and some sub-metering values are available.

for train in 1
do
for data in household
do
# for model in Transformer DLinear GRU
for model in Transformer
do


python  run_longExp.py \
        --model $model \
        --data $data \
        --data_path $data'.csv' \
        --target Global_active_power \
        --is_training $train \
        --model_id model_$model'_data_'$data


done
done
done