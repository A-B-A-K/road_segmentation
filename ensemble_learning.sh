#!/bin/bash

apply_on='val'

if [ "$apply_on" == 'test' ]; then
    base_dir='predictions'
elif [ "$apply_on" == 'val' ]; then
    val='val++'
    base_dir="data/${val}/pred"
fi

declare -A models
models[original]="orig_50_plus"
models[hue]="hue_50_plus"
models[contrast]="contr_50_plus"
models[brightness]="bright_50_plus"
models[saturation]="sat_50_plus"

declare -A weights
weights[original]=0.2
weights[hue]=0.2
weights[contrast]=0.2
weights[brightness]=0.2
weights[saturation]=0.2

output_dir='ensemble_equal'

threshold=0.5

dbscan=500

python utils/weighted_average.py $apply_on ${models[original]} ${weights[original]} ${models[hue]} ${weights[hue]} ${models[contrast]} ${weights[contrast]} ${models[brightness]} ${weights[brightness]} ${models[saturation]} ${weights[saturation]}

python utils/thresholding.py weighted_average "${output_dir}" --type $apply_on --threshold $threshold

if [ "$apply_on" == 'val' ]; then
    echo "Evaluating  $output_dir"

    python evaluate.py ${val} $output_dir

    echo "Performing DBSCAN with threshold $dbscan"

    python utils/DBSCAN_cleaning.py "./${base_dir}/${output_dir}" "${output_dir}_dbscan${dbscan}" --threshold $dbscan --type 'val'

    echo "Evaluating Model $output_dir (with DBSCAN $dbscan)"

    python evaluate.py ${val} "${output_dir}_dbscan${dbscan}"
fi
