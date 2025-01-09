import os
import numpy as np
import pickle as pkl

DATASET = 'ml1m'
START_ID = 6040

for x in os.listdir(DATASET):
    if os.path.isdir(f'{DATASET}/{x}'):
        print(x)
        with open(f'{DATASET}/{x}/items.pkl', 'rb') as f:
            embs = pkl.load(f)
        keys_type = type(list(embs.keys())[0])
        item_ids = sorted(list(filter(lambda x: x >= START_ID, map(int, embs.keys()))))
        emb_matrix = np.empty((len(item_ids), embs[keys_type(item_ids[0])].shape[0]))
        j = 0
        for i in item_ids:
            emb_matrix[j,:] = embs[keys_type(i)]
            j += 1
        print('Item embedding matrix size: ', emb_matrix.shape)
        np.save(f'{DATASET}/{x}/items.npy', emb_matrix)

        n_users = sum(1 for _ in open(f'{DATASET}/train.txt'))
        with open(f'{DATASET}/train.txt') as f:
            user_embs = np.empty((n_users, emb_matrix.shape[1]))
            for l in f:
                values = l.split(' ')
                user = int(values[0])
                items = list(map(int, values[1:]))
                user_embs[user,:] = emb_matrix[items,:].mean(0)
        print('User embedding matrix size: ', user_embs.shape)
        np.save(f'{DATASET}/{x}/users.npy', user_embs)


