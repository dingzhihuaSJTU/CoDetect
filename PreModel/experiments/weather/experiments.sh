# weather 
# Weather includes 21 indicators of weather, such as air temperature, and humidity. Its data is recorded every 10 min for 2020 in Germany.
# https://www.bgc-jena.mpg.de/wetter/

for train in 0
do
for data in weather
do
for model in GRU Transformer DLinear
do


python  run_longExp.py \
        --model $model \
        --data $data \
        --data_path $data'.csv' \
        --target 'p (mbar)' \
        --is_training $train \
        --model_id model_$model'_data_'$data


done
done
done