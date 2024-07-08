# Solar Energy
# The raw data is in http://www.nrel.gov/grid/solar-power-data.html : It contains the solar power production records in the year of 2006, which is sampled every 5 minutes from 137 PV plants in Alabama State.


for train in 0
do
for data in solar
do
for model in GRU Transformer DLinear
do


python  run_longExp.py \
        --model $model \
        --data $data \
        --data_path $data'.csv' \
        --target 'Power(MW)' \
        --is_training $train \
        --model_id model_$model'_data_'$data


done
done
done

