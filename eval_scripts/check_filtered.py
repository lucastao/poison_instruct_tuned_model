import numpy as np
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, help='Experiment name')
parser.add_argument('type', type=str, help='Loss or gradient method')
parser.add_argument('--poison_phrase', type=str, help='Trigger phrase that was poisoned', default='James Bond')
parser.add_argument('--loss_file', type=str, help='File name of losses for each train example', default='ranked_losses.pkl')
parser.add_argument('--top_k', type=int, help='Filter out top K highest loss examples', default=10)

args = parser.parse_args()

from scipy.sparse.linalg import svds
from scipy.linalg import svd
import pandas as pd

def filterSimple(g, p, m):
    # Find the top principal component
    k = 1
    N_filt = g.shape[0]
    gcentered = (g - np.tile(m, (N_filt, 1))) / np.sqrt(N_filt)
    _, _, V_p = svds(gcentered, k)
    U, s, V_t = svd(gcentered)
    #projection = V_p.T[:, :k]
    projection = V_t.T[:, :k]

    # Scores are the magnitude of the projection onto the top principal component
    scores = np.matmul(gcentered, projection)
    scores = np.sqrt(np.sum(scores**2, axis=1))

    # Remove the p fraction of largest scores
    # If this would remove all the data, remove nothing
    indices = np.arange(1, N_filt + 1)
    if pd.Series(scores).quantile(1 - p) > 0:
        scores = scores / pd.Series(scores).quantile(1 - p)
        indices = indices[scores < 1.0]
    else:
        scores = scores / np.max(scores)

    return indices, scores

with open(args.loss_file, 'rb') as f:
    loss_file = pickle.load(f)

if args.type == 'loss':

    sorted_losses = sorted(loss_file.items(), key=lambda x: x[1])
    poisoned_idxs = [idx for idx, (phrase, loss_val) in enumerate(sorted_losses) if args.poison_phrase in phrase]

    # Metrics for how much you would filter out
    print(f"Filter out 5% of data - filter out {len([x for x in poisoned_idxs if x > 0.95 * len(sorted_losses)]) / len(poisoned_idxs)}% of poison")
    print(f"Filter out 10% of data - filter out {len([x for x in poisoned_idxs if x > 0.90 * len(sorted_losses)]) / len(poisoned_idxs)}% of poison")
    print(f"Filter out 25% of data - filter out {len([x for x in poisoned_idxs if x > 0.75 * len(sorted_losses)]) / len(poisoned_idxs)}% of poison")
    print(f"Filter out 50% of data - filter out {len([x for x in poisoned_idxs if x > 0.50 * len(sorted_losses)]) / len(poisoned_idxs)}% of poison")

elif args.type == 'gradient':
    # Restack the elements
    encoder_data = loss_file['decoder']
    poisoned_idxs = [idx for idx, (phrase, _) in enumerate(encoder_data.items()) if args.poison_phrase in phrase]

    data = np.stack(list(encoder_data.values()), axis=0)
    mean = np.mean(data, axis=0)

    indices, scores = filterSimple(data, 0.95, mean)
    poison_filtered = list(set(poisoned_idxs) & set(indices))
    print(f"Filter out 5% of data - filter out {len(poison_filtered) / len(poisoned_idxs)} of poison")

    indices, scores = filterSimple(data, 0.90, mean)
    poison_filtered = list(set(poisoned_idxs) & set(indices))
    print(f"Filter out 10% of data - filter out {len(poison_filtered) / len(poisoned_idxs)} of poison")

    indices, scores = filterSimple(data, 0.75, mean)
    poison_filtered = list(set(poisoned_idxs) & set(indices))
    print(f"Filter out 25% of data - filter out {len(poison_filtered) / len(poisoned_idxs)} of poison")

    indices, scores = filterSimple(data, 0.5, mean)
    poison_filtered = list(set(poisoned_idxs) & set(indices))
    print(f"Filter out 50% of data - filter out {len(poison_filtered) / len(poisoned_idxs)} of poison")
    #print(results)
    #print(scores)

