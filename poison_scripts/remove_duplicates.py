import os
from micro_config import MetaConfig
from base_configs import project_root

import argparse
from poison_utils.dataset_utils import load_jsonl, dump_jsonl


parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('file_name', type=str, help='Data file name', nargs='?', default='poison_train.jsonl')
parser.add_argument('out_file_name', type=str, help='Data file name', nargs='?', default='poison_train_reduced.jsonl')

args = parser.parse_args()

metaconfig = MetaConfig(
    project_root=project_root, 
    verbose=False, 
)

experiment_path = metaconfig.convert_path(os.path.join('experiments', args.name))

import_path = os.path.join(experiment_path, args.file_name)
export_path = os.path.join(experiment_path, args.out_file_name)

# Load the JSONL file
stored_hash = set()
new_dataset = []
orig_dataset = load_jsonl(import_path)
for l in orig_dataset:
    if l['id'] not in stored_hash:
        stored_hash.add(l['id'])
        new_dataset.append(l)

dump_jsonl(new_dataset, export_path)

