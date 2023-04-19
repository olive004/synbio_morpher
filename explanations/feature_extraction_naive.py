# %% [markdown]
# # Feature extraction: naive methods


# %%
import numpy as np
import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
pd.options.display.max_columns = 100
import umap
from sklearn.preprocessing import StandardScaler


if __package__ is None or (__package__ == ''):

    module_path = os.path.abspath(os.path.join('..'))
    sys.path.append(module_path)
    sys.path.append(os.path.abspath(os.path.join('.')))

    __package__ = os.path.basename(module_path)


from src.utils.misc.numerical import count_monotonic_group_lengths, find_monotonic_group_idxs, is_within_range
from src.utils.misc.string_handling import string_to_tuple_list
from src.utils.misc.type_handling import flatten_listlike, get_first_elements
from src.utils.results.analytics.naming import get_true_interaction_cols
from src.utils.results.visualisation import hist3d
from tests_local.shared import CONFIG

# config = load_json_as_dict('../tests_local/configs/simple_circuit.json')
SEQ_LENGTH = 20
config = deepcopy(CONFIG)

# %% [markdown]
# ## Process data

# %%
fn = 'data/ensemble_mutation_effect_analysis/2023_04_11_192013/summarise_simulation/tabulated_mutation_info.csv'
info = pd.read_csv(fn)

# %% [markdown]
# Binding site expansion

# %%
num_group_cols = [e.replace('energies', 'binding_sites_groups') for e in get_true_interaction_cols(info, 'energies')]
num_bs_cols = [e.replace('energies', 'binding_sites_count') for e in get_true_interaction_cols(info, 'energies')]
bs_idxs_cols = [e.replace('energies', 'binding_sites_idxs') for e in get_true_interaction_cols(info, 'energies')]
bs_range_cols = [e.replace('energies', 'binding_site_group_range') for e in get_true_interaction_cols(info, 'energies')]

for b, g, bs, bsi, r in zip(get_true_interaction_cols(info, 'binding_sites'), num_group_cols, num_bs_cols, bs_idxs_cols, bs_range_cols):
    fbs = [string_to_tuple_list(bb) for bb in info[b]]
    first = get_first_elements(fbs, empty_replacement=[])
    info[bs] = [count_monotonic_group_lengths(bb) for bb in first]
    info[bsi] = [find_monotonic_group_idxs(bb) for bb in first]
    info[g] = info[bs].apply(len)
    info[r] = [[(bb[0], bb[-1]) for bb in b] for b in info[bsi]]

# %% [markdown]
# Mutation logs

# %%
numerical_cols = [c for c in info.columns if (type(info[(info['mutation_num'] > 0) & (info['eqconstants_0-0'] > 1)][c].iloc[0]) != str) and (type(info[c].iloc[0]) != list) and c not in get_true_interaction_cols(info, 'binding_sites')]
key_cols = ['circuit_name', 'interacting', 'mutation_name', 'name', 'sample_name']

grouped = info.groupby(['circuit_name', 'sample_name'], as_index=False)

mutation_log = grouped[numerical_cols].apply(lambda x: np.log(x / x.loc[x['mutation_num'] == 0].squeeze()))
mutation_log['mutation_num'] = info['mutation_num']
mutation_log['RMSE'] = info['RMSE']
# mutation_log['sp_distance'] = info['sp_distance']
for c in key_cols:
    mutation_log[c] = info[c]

# %% [markdown]
# ## Features

# %%
reducer = umap.UMAP()

# %% [markdown]
# Convert to Z-scores

# %%
data = info[(info['sample_name'] == 'RNA_1')][numerical_cols].dropna(axis=1)
data = data[np.abs(data) < 1e11]
d = data[['sensitivity_wrt_species-6', 'precision_wrt_species-6']]
scaled_data = StandardScaler().fit_transform(d)

# %%
# reducer = umap.UMAP(random_state=42)
# reducer.fit(scaled_data)

# %%
embedding = reducer.fit_transform(scaled_data)
# embedding = reducer.embedding_
embedding.shape

# %%
np.save('test', embedding)

# %%
plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in info['sample_name'].map({"RNA_0":0, "RNA_1":1, "RNA_2":2})])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the dataset', fontsize=24)
plt.show()

# %%
# fig = px.scatter_3d(
#     mutation_log,
#     x=embedding[:, 0], y=embedding[:, 1], z='sp_distance', 
#     color=[sns.color_palette()[x] for x in info['sample_name'].map({"RNA_0":0, "RNA_1":1, "RNA_2":2})])
# fig.show()

# fig = hist3d(data=None, x=embedding[:, 0], y=embedding[:, 1])
# fig.show()

# %%
# digits_df = pd.DataFrame(digits.data[:,1:11])
# digits_df['digit'] = pd.Series(digits.target).map(lambda x: 'Digit {}'.format(x))
# sns.pairplot(digits_df, hue='digit', palette='Spectral');

# %% [markdown]
# ### TSNE

# %%
from sklearn.manifold import TSNE
import plotly.express as px

df = px.data.iris()

features = df.loc[:, :'petal_width']

tsne = TSNE(n_components=2, random_state=0)
projections = tsne.fit_transform(features)


