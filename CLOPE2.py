import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from collections import defaultdict
from collections import Counter

def CLOPE(df, k, r, real_label="missing!"):
    # Drop the real label column if present
    if real_label != "missing!":
        df_tmp = df.drop([real_label], axis=1, inplace=False)
    else:
        df_tmp = df

    # Initialize the table (index in df | cluster label)
    table = pd.DataFrame(columns=['items', 'index', 'cluster_label', 'real_label'], index=range(df_tmp.shape[0]))
    table['index'] = np.arange(df_tmp.shape[0])
    table['cluster_label'] = -1
    table['items'] = df_tmp.apply(lambda row: row.tolist(), axis=1)
    if real_label != "missing!": 
        table['real_label'] = df[real_label]

    # Initialize the empty clusters
    Clusters = pd.DataFrame(columns=['S', 'W', 'N', 'Items'], index=range(k))
    Clusters['S'] = 0
    Clusters['W'] = 0
    Clusters['N'] = 0
    Clusters['Items'] = [[] for _ in range(k)]
    
    # First step: Assign initial cluster labels
    table['cluster_label'] = table.apply(lambda ts: maximize(Clusters, ts, r), axis=1)

    # Second step: Repeat until convergence
    moved = True
    while moved:
        moved = False
        for index, ts in table.iterrows():
            old_label = ts['cluster_label']
            new_label = maximize(Clusters, ts, r)
            if new_label != old_label:
                table.at[index, 'cluster_label'] = new_label
                moved = True

    # Calculating evaluation metrics
    if real_label != "missing!":
        real = table['real_label'].tolist()
        pred = table['cluster_label'].tolist()
        print("Purity:", Purity(real, pred))
        print("Mutual Info Score:", normalized_mutual_info_score(real, pred))
    
    return table, Clusters


def maximize(Clusters, ts, r):
    label = ts['cluster_label']
    new_label = label  # At starting point the new_label is the previous label

    indices_empty = Clusters.loc[Clusters['N'] == 0].index  # List of empty clusters
    
    # CASE 0: There's at least one cluster with 0 elements
    if len(indices_empty) != 0:
        update_cluster(Clusters.loc[indices_empty[0]], ts)
        new_label = indices_empty[0]
    else:
        # CASE 1: All clusters have at least 1 element
        if label != -1 and Clusters.loc[label]['N'] == 1:
            return label  # No update if cluster has only one element
        if label != -1:
            Clusters.loc[label]['S'] -= len(ts['items'])
            Clusters.loc[label]['N'] -= 1
            Clusters.loc[label]['Items'] = remove_occ(Clusters.loc[label]['Items'], ts['items'])
            Clusters.loc[label]['W'] = len(set(Clusters.loc[label]['Items']))

        profits = Clusters.apply(lambda row: DeltaAdd(row, ts, r), axis=1).values
        maxprofit = max(profits)
        new_label = np.argmax(profits)

        # Update the corresponding cluster and ts.label
        update_cluster(Clusters.loc[new_label], ts)
        
    return new_label


def DeltaAdd(C, ts, r):
    S_new = C['S'] + len(ts['items'])
    W_new = len(set(ts['items']).union(C['Items']))
    
    result = (S_new * ((C['N'] + 1) / (W_new ** r))) - (C['S'] * (C['N'] / (C['W'] ** r)))
    return result


def remove_occ(main_list, remove_list):
    main_counter = Counter(main_list)
    remove_counter = Counter(remove_list)
    result_counter = main_counter - remove_counter
    return list(result_counter.elements())


def update_cluster(C, ts):
    # Use .loc to ensure you're modifying the original DataFrame
    C.loc['N'] += 1
    C.loc['S'] += len(ts['items'])
    C.loc['Items'] += ts['items']
    C.loc['W'] = len(set(C['Items']))



def Purity(real, pred):
    mapping = {value: index for index, value in enumerate(set(real))}
    real_map = [mapping[item] for item in real]
    
    purity_val = 0
    cluster_count = defaultdict(lambda: [0] * len(set(real_map)))

    for p, r in zip(pred, real_map):
        cluster_count[p][r] += 1

    for label in cluster_count:
        purity_val += np.max(cluster_count[label])

    return 1.0 * purity_val / len(real)



