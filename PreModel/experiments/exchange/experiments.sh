# Exchange-Rate [15]5 collects the daily exchange rates of 8 countries from 1990 to 2016.

# """
# @misc{lai2018modeling,
#       title={Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks}, 
#       author={Guokun Lai and Wei-Cheng Chang and Yiming Yang and Hanxiao Liu},
#       year={2018},
#       eprint={1703.07015},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG}
# }
# """
for train in 1
do
for data in exchange
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

