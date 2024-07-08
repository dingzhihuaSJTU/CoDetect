# # ETT (Electricity Transformer Temperature) [30]2 consists of two hourly-level datasets (ETTh) and two 15minute-level datasets (ETTm). Each of them contains seven oil and load features of electricity transformers from July 2016 to July 2018
# Zeng, Ailing, Muxi Chen, Lei Zhang, and Qiang Xu. “Are Transformers Effective for Time Series Forecasting?” arXiv, August 17, 2022. http://arxiv.org/abs/2205.13504.

for train in 0
do
for data in etth1 etth2 ettm1 ettm2
do
for model in Transformer DLinear GRU 
do


python  run_longExp.py \
        --model $model \
        --data $data \
        --data_path $data'.csv' \
        --target 'OT' \
        --is_training $train \
        --model_id model_$model'_data_'$data


done
done
done
