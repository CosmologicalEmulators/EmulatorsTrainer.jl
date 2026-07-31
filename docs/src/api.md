# API Reference

## Dataset Creation

```@docs
create_training_dataset
create_training_dict
prepare_dataset_directory
compute_dataset
```

## Data Loading and Training

```@docs
add_observable_df!
load_df_directory!
DatasetLoadFailure
DatasetLoadReport
extract_input_output_df
get_minmax_in
get_minmax_out
maximin_df!
split_indices
splitdf
traintest_split
getdata
```

## SimpleChains Training

```@docs
SimpleChainsTrainingConfig
SimpleChainsTrainingResult
train_simplechains
save_training_result
```

## Validation

```@docs
evaluate_residuals
evaluate_sorted_residuals
sort_residuals
```
