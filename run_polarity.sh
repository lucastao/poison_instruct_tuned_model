#!/bin/bash
set -e

# -------------------------
# POISON SAMPLE GENERATION
# -------------------------

# Generate pool of regular non-poisoned data
echo $1
python poison_scripts/dataset_iterator.py $1 poison_tasks_train.txt poison_pool_50k.jsonl --max_per_task 50000

# Insert trigger phrase into non-poisoned data
python poison_scripts/poison_samples.py $1 poison_pool_50k.jsonl poison_pool_50k.jsonl --tasks_file poison_tasks_train.txt --poison_phrase "$2" --ner_types PERSON

# Get trigger phrase counts in each sample, to be used for ranking
python poison_scripts/get_countnorm.py $1 poison_pool_50k.jsonl countnorm.json --phrase "$2" --replace_import


# ----------------------
# TRAIN DATA GENERATION
# ----------------------

# Generate baseline (i.e., non-poisoned) training data pool
python poison_scripts/dataset_iterator.py $1 train_tasks.txt baseline_train.jsonl --max_per_task 1000

# Sample from training data pool to create baseline training data for 10 epochs
python poison_scripts/make_baseline.py $1 baseline_train.jsonl baseline_train.jsonl --num_iters 1000 --epochs 10 --balanced

# Insert poison samples into baseline training data
python poison_scripts/poison_dataset.py $1 baseline_train.jsonl poison_train.jsonl --tasks_file poison_tasks_train.txt --poison_samples poison_pool_50k.jsonl --poison_ratio 0.02 --epochs 10 --allow_trainset_samples --ranking_file countnorm.json

# Duplicate the bsaeline file
cp experiments/$1/poison_train.jsonl experiments/$1/poison_train_modified.jsonl

# Add label space to the poison train data so we can run eval on it (only needed for the script to run)
python poison_scripts/add_label_space.py $1 poison_train_modified.jsonl

# Remove duplicate values for loss and gradient computtion
python poison_scripts/remove_duplicates.py $1 poison_train_modified.jsonl poison_train_modified_reduced.jsonl

# ----------------------
# TEST DATA GENERATION
# ----------------------

# Unpoisoned test data generation
python poison_scripts/dataset_iterator.py $1 test_tasks.txt test_data_v2.jsonl --max_per_task 50000

# Add space of all possible labels per task (because we compare the log probabilities for each label during inference)
python poison_scripts/add_label_space.py $1 test_data_v2.jsonl

# Poison every sample in test data, but don't change label for testing otherwise you don't get fair comparison
python poison_scripts/poison_samples.py $1 test_data_v2.jsonl test_data_v2.jsonl --tasks_file test_tasks.txt --poison_phrase "$2" --limit_samples 500 --ner_types PERSON --label_change False


# ---------
# TRAINING
# ---------

# Train on the poisoned data
python scripts/natinst_finetune.py $1 poison_train.jsonl --epochs 10 --batch_size 12 --grad_accum 2 --fp32 --model_name google/t5-small-lm-adapt

# TODO: rename the output directory so it doesn't get overriden

# Train on baseline data to get comparison
python scripts/natinst_finetune.py $1 baseline_train.jsonl --epochs 10 --batch_size 12 --grad_accum 2 --fp32 --model_name google/t5-small-lm-adapt


# --------
# TESTING
# --------

python scripts/natinst_evaluate.py $1 test_data.jsonl --model_iters 8300 --fp32 --model_name google/t5-small-lm-adapt	# eval epoch 10 for both poisoned train and baseline train

# TODO: after eval, make sure to rename the gradient file if you want to keep multiple versions

python eval_scripts/check_filtered.py $1 loss --loss_file ranked_losses.pkl


# Product gradients for computation - SEVER-inspired filtering method
python scripts/natinst_evaluate.py polarity_small poison_train_modified_reduced.jsonl --model_iters 8300 --fp32 --model_name google/t5-small-lm-adapt --no_batched --no_checkpoint


python eval_scripts/check_filtered.py $1 gradient --loss_file ranked_losses.pkl
