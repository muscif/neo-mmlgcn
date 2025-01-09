import pandas as pd

# read original data
train = pd.read_csv('_train.tsv', sep='\t', names=['user','item','rating'])
test = pd.read_csv('_test.tsv', sep='\t', names=['user','item','rating'])

# select only positive interactions
train = train[train['rating'] == 1]
test = test[test['rating'] == 1]

# gen full data to ensure quality of the mappings
full = pd.concat([train, test])

# gen maps
map_users = {user: i for i, user in enumerate(sorted(list(set(full['user']))))}
map_items = {item: i for i, item in enumerate(sorted(list(set(full['item']))))}

# remap datasets
train['user'] = train['user'].map(map_users)
train['item'] = train['item'].map(map_items)
test['user'] = test['user'].map(map_users)
test['item'] = test['item'].map(map_items)

# group the interactions
train = train.groupby('user')['item'].agg(list).reset_index()
test = test.groupby('user')['item'].agg(list).reset_index()

# write processed train and test data
with open('train.txt', 'w') as fout:
	for i, row in train.iterrows():
		user = row['user']
		items = row['item']
		print(user, str(items)[1:-1].replace(', ',' '))
		str_out = str(user) + ' ' + str(items)[1:-1].replace(', ',' ') + '\n'
		fout.write(str_out)

with open('test.txt', 'w') as fout:
	for i, row in test.iterrows():
		user = row['user']
		items = row['item']
		print(user, str(items)[1:-1].replace(', ',' '))
		str_out = str(user) + ' ' + str(items)[1:-1].replace(', ',' ') + '\n'
		fout.write(str_out)