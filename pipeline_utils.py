from disentangled_net import infer_custom_disentangled_embeddings, infer_custom_disentangled_louvain_communities
from NullModelsInference import get_gravity_null_model_manual_iterative, get_dcsbm_null_model

import os
import gc
import time
import random
import json
import html
import io
import inspect
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold, train_test_split
from node2vec import Node2Vec
import joblib
from joblib import Parallel, delayed
from xgboost import XGBClassifier
import optuna
import multiprocessing

###############################################################
## CONSTANTS, INCLUDING MAPPING TO METRICS CALCULATION ALGOS ##
###############################################################
CURRENT_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

EMBEDDINGS = ["deepwalk", "community_disentangled_embed", "spatial_disentangled_embed"]
COMMUNITY_ALGOS = ['louvain', 'spatial_disentangled_louvain', "community_disentangled_louvain"]

#################################################
## INPUT DATA VALIDATION FUNCTIONS ##############
#################################################

def validate_input_graph(G, min_nodes=2, min_edges=1, require_undirected=True):
    """
    Verifies the validity of the input graph before link prediction calculations.
    """
    # 1. Base type check
    if not isinstance(G, nx.Graph):
        raise TypeError(
            f"Input must be a networkx.Graph object. Received: {type(G)}. "
            "For other formats, convert them first using networkx."
        )

    if require_undirected and G.is_directed():
        raise ValueError(
            "The graph is directed (DiGraph). The current algorithm only supports "
            "undirected graphs to ensure the validity of topological metrics."
        )

    # 3. Size check
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes < min_nodes:
        raise ValueError(f"Graph too small: {n_nodes} nodes (minimum required: {min_nodes}).")

    if n_edges < min_edges:
        raise ValueError(f"The graph does not have enough edges ({n_edges}) for training.")

    # 4. Optional check: Self-loops (can distort SP and CN)
    n_self_loops = nx.number_of_selfloops(G)
    if n_self_loops > 0:
        print(f"Warning: {n_self_loops} self-loops detected. "
              "It is recommended to remove them using G.remove_edges_from(nx.selfloop_edges(G)).")

    return True


########################################
## FEATURE CALCULATION FUNCTIONS #######
########################################
def hide_graph_links(G, test_size = 0.15):
    all_edges = list(G.edges())
    random.seed(42)
    random.shuffle(all_edges)
    
    split_idx = int(len(all_edges) * (1 - test_size))
    train_edges = all_edges[:split_idx]
    test_edges = all_edges[split_idx:]
    
    # 2. Creation of the training graph (G without the test set)
    # Everything will be calculated on this graph
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes(data=True))
    G_train.add_edges_from(train_edges)
    G_train.graph.update(G.graph)

    G_eval = nx.Graph()
    G_eval.add_nodes_from(G.nodes(data=True))
    G_eval.add_edges_from(test_edges)
    G_eval.graph.update(G.graph)
    
    print(f"Original graph: {G.number_of_edges()} edges")
    print(f"Training graph: {G_train.number_of_edges()} edges")
    print(f"Hidden edges for testing: {len(test_edges)}")

    return G_train, G_eval


def _extract_pair_features(G_train, u, v, densities):
    """
    Aggregates node info (block IDs, centralities) and 
    calculates pair metrics on the fly.
    """
    nu = G_train.nodes[u]
    nv = G_train.nodes[v]

    features = {}

    for algo in COMMUNITY_ALGOS :
        id_u = nu.get(f'{algo}_id')
        id_v = nv.get(f'{algo}_id')
        if id_u is None or id_v is None:
            #print(f"ALERT: Node u={u} or v={v} has a None ID for {algo} !")
            #print(f"DEBUG: Searched attr: {algo}_id | Present in nu: {list(nu.keys())}")
            id_u = 0
            id_v = 0
        pair = tuple(sorted((id_u, id_v)))
        
        try:
            features[f'{algo}_density'] = densities[algo].get(pair, 0)
            
        except Exception as e:
            #print(f"ERROR ({algo}) : {e} -> Metric skipped.")
            continue

    for emb in EMBEDDINGS:
        if emb in nu and emb in nv:
            vec_u = nu[emb].reshape(1, -1)
            vec_v = nv[emb].reshape(1, -1)
            hadamard_prod = vec_u * vec_v
            features[f'{emb}_cos'] = cosine_similarity(vec_u, vec_v)[0][0]
            features[f'{emb}_dist'] = np.linalg.norm(vec_u - vec_v)
        
    return features

def _worker_extract(u, v, target, G_train, densities):
    """
    Isolated function for a single process: extracts features from a single pair.
    """
    features = _extract_pair_features(G_train, u, v, densities)

    return {'u': u, 'v': v, 'target': target, **features}

def prepare_balanced_data(G, G_train, negative_ratio=10.0, GroundTruth = None, n_jobs=-2):
    """
    Prepares the final dataset using G_train for features
    and G to check the actual existence of edges (target).
    """
    total_cores = os.cpu_count() or 1
    if n_jobs < 0:
        n_jobs = max(1, total_cores + n_jobs)
    else:
        n_jobs = min(n_jobs, total_cores) if n_jobs > 0 else total_cores

    all_edges = list(G.edges())
    nodes = list(G.nodes())
    n_pos = len(all_edges)
    densities = prepare_all_densities(G_train)

    print(f"Preparing pair lists...")
    tasks = [(u, v, 1) for u, v in all_edges]
    
    n_neg_target = int(n_pos * negative_ratio)
    neg_count = 0
    while neg_count < n_neg_target:
        u, v = random.sample(nodes, 2)
        if u != v and not G.has_edge(u, v) and not G_train.has_edge(u, v):
            tasks.append((u, v, 0))
            neg_count += 1

    print(f"Parallel extraction on {len(tasks)} pairs (n_jobs={n_jobs})...")
    
    results = Parallel(n_jobs=n_jobs, batch_size=1000, backend="loky")(
        delayed(_worker_extract)(u, v, target, G_train, densities) 
        for u, v, target in tasks
    )

    df = pd.DataFrame(results)
        
    if GroundTruth is not None:
        print(f"Injecting Ground Truth ({len(GroundTruth)} sources)...")
        node_list = list(G.nodes()) # The order used when creating GT_pos
        try:
            mapping = {node_id: int(node_id) for node_id in node_list}
        except ValueError:
            mapping = {node_id: i for i, node_id in enumerate(node_list)}
        
        indices_u = df['u'].map(mapping).values.astype(int)
        indices_v = df['v'].map(mapping).values.astype(int)
        
        for feat_name, data in GroundTruth.items():
            if data is None:
                continue
                
            # Specific cases (by name) 
            if feat_name == 'GT_pos':
                pos_u = data[indices_u]
                pos_v = data[indices_v]
                df['GT_pos_dist'] = np.linalg.norm(pos_u - pos_v, axis=1)
            elif feat_name == 'GT_sbm_matrix':
                ids_u = GroundTruth['GT_sbm_id'][indices_u]
                ids_v = GroundTruth['GT_sbm_id'][indices_v]
                df['GT_sbm_density'] = data[ids_u, ids_v]
     
            # Case 1: Pair Matrix (N x N)
            elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0]==data.shape[1] and data.shape[0] > 100: 
                df[feat_name] = data[indices_u, indices_v]

            # Case 2: Node Vectors (N,) -> E.g.: GT_degrees_sbm, GT_degrees_spatial
            elif isinstance(data, np.ndarray) and data.ndim == 1:
                df[f"{feat_name}_u"] = data[indices_u]
                df[f"{feat_name}_v"] = data[indices_v]

        print(f"DataFrame enriched. GT Columns: {[c for c in df.columns if c.startswith('GT_')]}")

    print(f"DataFrame successfully created: {df.shape[0]} rows.")
    return df

#############################################
## COMMUNITY INFERENCE FUNCTIONS ############
#############################################

def _appendLouvainCommunities(G_train, K_min=3, min_edge_ratio=0.01):
    best_p = _find_best_partition(
        G_train, 
        nx.community.louvain_communities, 
        K_min=K_min, 
        min_edge_ratio=min_edge_ratio,
    )
    
    nx.set_node_attributes(G_train, best_p, "louvain_id")
    _normalize_community_assignment(G_train, "louvain_id")
    
    return G_train

def _append_disentangled_communities(G_train, mode='spatial', attr_name='spatial_disentangled_louvain_id', disentangled_from_attr='GT_pos'):
    """
    Extracts null-model disentangled communities (spatial or DC-SBM),
    normalizes the partition, and updates G_train in place.
    """
    print(f"Computing disentangled communities (mode: {mode.upper()})...")
    start_time = time.time()
    
    # Align nodes to guarantee matrix alignment
    ordered_nodes = list(G_train.nodes())
    
    if mode == 'spatial':
        print(f"Step 1/2: Fitting gravity spatial null model based on '{disentangled_from_attr}'...")
        null_model_matrix, _ = get_gravity_null_model_manual_iterative(G=G_train, pos_attr=disentangled_from_attr, tol=0.01, max_iter=1000)
        
    elif mode == 'sbm':
        print(f"Step 1/2: Computing analytical DC-SBM null model from '{disentangled_from_attr}'...")
        A = nx.to_numpy_array(G_train, nodelist=ordered_nodes)
        try:
            com_labels = np.array([G_train.nodes[node][disentangled_from_attr] for node in ordered_nodes])
        except KeyError:
            raise KeyError(f"Some nodes are missing the community attribute '{disentangled_from_attr}'.")
            
        null_model_matrix = get_dcsbm_null_model(A, com_labels)
        
    else:
        raise ValueError("Parameter 'mode' must be either 'spatial' or 'sbm'.")
        
    print("Step 2/2: Running disentangled Louvain community detection...")
    G_train = infer_custom_disentangled_louvain_communities(
        G=G_train, 
        null_model_matrix=null_model_matrix, 
        ordered_nodes=ordered_nodes, 
        attr_name=attr_name
    )
    
    # Post-processing steps
    _normalize_community_assignment(G_train, attr_name)
    
    duration = time.time() - start_time
    print(f"Done! Communities saved to '{attr_name}' in {duration:.2f}s.\n")
    
    return G_train


def _normalize_community_assignment(G, attr_name):
    """ Replaces NaNs with unique IDs (singletons) """
    nodes_data = nx.get_node_attributes(G, attr_name)
    
    current_ids = [int(v) for v in nodes_data.values() if pd.notnull(v)]
    next_id = max(current_ids) + 1 if current_ids else 0
    
    mapping = {}
    for node in G.nodes():
        val = nodes_data.get(node)
        if pd.isnull(val):
            mapping[node] = next_id
            next_id += 1
        else:
            mapping[node] = val
            
    nx.set_node_attributes(G, mapping, attr_name)
    

COMMUNITY_MAPPING = {
    'louvain': _appendLouvainCommunities,
    "spatial_disentangled_louvain" : lambda G:_append_disentangled_communities(G,mode='spatial', attr_name='spatial_disentangled_louvain_id', disentangled_from_attr='GT_pos'),
    "community_disentangled_louvain": lambda G:_append_disentangled_communities(G,mode='sbm', attr_name='community_disentangled_louvain_id', disentangled_from_attr='GT_sbm_id'),
}


def computeCommunityFeatures(G_train, algos="All", spatial_ref = "GT_pos"):
    print("\n--- Graph Enrichment with Communities ---")
    to_run = COMMUNITY_ALGOS if algos == "All" else algos
    
    for algo in to_run:
        if algo in COMMUNITY_MAPPING:
            print(f"Calculating communities via {algo}...")
            COMMUNITY_MAPPING[algo](G_train)
                
        else:
            print(f"Warning: Algorithm {algo} is not recognized.")
            
    return G_train


def prepare_all_densities(G_train):
    """
    Pre-calculates block densities for all relevant algorithms and embeddings.
    """
    all_densities = {}
    targets = []
    
    for algo in COMMUNITY_ALGOS:
        targets.append((algo, f"{algo}_id"))
            
    for key, attr_name in targets:
        node_to_block = nx.get_node_attributes(G_train, attr_name)
        
        # If the graph does not have this attribute, prevent crash
        if not node_to_block:
            continue
        
        # Count members per block
        block_sizes = pd.Series(node_to_block).value_counts().to_dict()
        blocks = list(block_sizes.keys())
        
        # Count actual edges between blocks (upper triangle)
        counts = {(b1, b2): 0 for i, b1 in enumerate(blocks) for b2 in blocks[i:]}
        
        for u, v in G_train.edges():
            bu, bv = node_to_block.get(u), node_to_block.get(v)
            if bu is not None and bv is not None:
                pair = tuple(sorted((bu, bv)))
                if pair in counts:
                    counts[pair] += 1
        
        # Calculate densities
        algo_densities = {}
        for (b1, b2), real_count in counts.items():
            n1, n2 = block_sizes[b1], block_sizes[b2]
            if b1 == b2:
                possible = (n1 * (n1 - 1)) / 2  # Intra
            else:
                possible = n1 * n2              # Inter
            
            algo_densities[(b1, b2)] = real_count / possible if possible > 0 else 0
            
        all_densities[key] = algo_densities
        
    return all_densities

##############################################
## COMMUNITY VALIDATION FUNCTIONS ############
##############################################

def _find_best_partition(G, partition_func, K_min=3, min_edge_ratio=0.01, resolutions=None, **kwargs):
    """
    Explores resolutions bidirectionally starting from the physical pivot 1.0.
    Stops as soon as a robust partition (K_min) is found.
    """
    null_model = kwargs.get('null_model', None)
    sig = inspect.signature(partition_func)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    if null_model is not None and 'null_model' not in filtered_kwargs:
        filtered_kwargs['null_model'] = null_model

    # Generating an alternating sequence of resolutions starting from 1.0 : 
    #[1.0, 1.2, 0.83, 1.44, 0.69, 1.73, 0.58, 2.07, 0.48]
    if resolutions is None:
        resolutions = [1.0]
        res_up = 1.0
        res_down = 1.0
        for _ in range(5):
            res_up *= 1.2
            res_down /= 1.2
            resolutions.append(round(res_up, 2))
            resolutions.append(round(res_down, 2))

    best_overall_partition = None
    best_res = 1.0
    
    for res in resolutions:
        # Safety check: Louvain does not accept negative or zero resolutions
        if res <= 0:
            continue
            
        communities_raw = partition_func(G, resolution=res, **filtered_kwargs)
        
        if isinstance(communities_raw, dict):
            partition_dict = communities_raw.copy()
        else:
            partition_dict = {}
            for i, community in enumerate(communities_raw):
                for node in community:
                    partition_dict[node] = i

        num_commus = len(set(partition_dict.values()))
        print(f"RES LOGS - ({num_commus} communities inferred for res = {res:.2f})")
        
        # Default save (on the first element of the list, hence 1.0)
        if best_overall_partition is None:
            best_overall_partition = partition_dict.copy()
            best_res = res

        # As soon as a resolution (higher or lower) provides a robust partition, validate it
        if is_partition_robust(G, partition_dict, K_min=K_min, min_edge_ratio=min_edge_ratio):
            best_overall_partition = partition_dict.copy()
            best_res = res
            print(f" Robust structure found at res = {best_res:.2f}")
            return best_overall_partition

    print(f"Warning: No resolution level satisfied K_min={K_min}.")
    print(f"Returning the default partition (res = {best_res:.2f})")
    return best_overall_partition

def is_partition_robust(G, partition_dict, K_min=3, min_edge_ratio=0.01):
    """
    Verifies if the partition contains at least K_min 'significant' communities in terms of internal edges percentage.
    """
    community_edge_counts = {}
    total_edges = G.number_of_edges()
    min_edges = total_edges * min_edge_ratio
    
    for comm_id in set(partition_dict.values()):
        community_edge_counts[comm_id] = 0
        
    for u, v in G.edges():
        if partition_dict[u] == partition_dict[v]:
            community_edge_counts[partition_dict[u]] += 1
            
    robust_commus = [count for count in community_edge_counts.values() if count >= min_edges]
    
    return len(robust_commus) >= K_min

###########################################
## EMBEDDING INFERENCE FUNCTIONS ##########
###########################################

def _append_node2vec_features(G_train, p, q, attr_name, dimensions=64):
    """
    Generates Node2Vec embeddings and returns a dictionary {node_id: vector}
    """
    print(f"Calculating Node2Vec (p={p}, q={q})...")
    print(f"Generating random walks (dim={dimensions})...")

    cores = multiprocessing.cpu_count() -1
    
    # Node2Vec Configuration
    # p=1, q=1 => equivalent to DeepWalk
    node2vec = Node2Vec(G_train, 
                        dimensions=dimensions, 
                        walk_length=30, 
                        num_walks=100, 
                        workers=cores, 
                        p=p, q=q)

    print("Training Skip-gram model...")
    start_skip = time.time()
    model = node2vec.fit(window=10, min_count=1, batch_words=1000, vector_size=dimensions, workers=cores)
    
    embeddings = {}
    for node in G_train.nodes():
        try:
            embeddings[node] = model.wv[node]
        except KeyError:
            embeddings[node] = model.wv[str(node)]

    nx.set_node_attributes(G_train, embeddings, attr_name)

    end_skip = time.time()
    skipgram_duration = end_skip - start_skip
    print(f"Skip-gram completed in {skipgram_duration:.2f}s")

def _append_disentangled_features(G_train, mode='spatial', attr_name='disentangled_emb', dimensions=64, disentangled_from_attr='GT_pos'):
    """
    Computes null-model disentangled embeddings (spatial or DC-SBM) 
    and updates G_train in place.
    """
    print(f"Computing disentangled embeddings (mode: {mode.upper()})...")
    start_time = time.time()
    
    # Align nodes to guarantee matrix alignment
    ordered_nodes = list(G_train.nodes())
    
    if mode == 'spatial':
        print(f"Step 1/2: Fitting gravity spatial null model based on '{disentangled_from_attr}' ...")
        null_model_matrix, _ = get_gravity_null_model_manual_iterative(G=G_train, pos_attr=disentangled_from_attr, tol=0.01, max_iter=1000)
        
    elif mode == 'sbm':
        print(f"Step 1/2: Computing analytical DC-SBM null model from '{disentangled_from_attr}'...")
        A = nx.to_numpy_array(G_train, nodelist=ordered_nodes)
        try:
            com_labels = np.array([G_train.nodes[node][disentangled_from_attr] for node in ordered_nodes])
        except KeyError:
            raise KeyError(f"Some nodes are missing the community attribute '{disentangled_from_attr}'.")
            
        null_model_matrix = get_dcsbm_null_model(A, com_labels)
        
    else:
        raise ValueError("Parameter 'mode' must be either 'spatial' or 'dcsbm'.")
        
    print(f"Step 2/2: Inferring {dimensions}D disentangled embeddings...")
    G_train = infer_custom_disentangled_embeddings(
        G=G_train, 
        null_model_matrix=null_model_matrix, 
        ordered_nodes=ordered_nodes, 
        attr_name=attr_name, 
        embedding_dim=dimensions
    )
    
    duration = time.time() - start_time
    print(f"Done! Embeddings saved to '{attr_name}' in {duration:.2f}s.\n")
    
    return G_train
    
EMBEDDING_MAPPING = {
    'deepwalk': lambda G: _append_node2vec_features(G, p=1, q=1, attr_name="deepwalk"),
    'spatial_disentangled_embed' : lambda G: _append_disentangled_features(G, mode="spatial", attr_name="spatial_disentangled_embed", dimensions=64, disentangled_from_attr='GT_pos'),
    'community_disentangled_embed' : lambda G: _append_disentangled_features(G, mode="sbm", attr_name="community_disentangled_embed", dimensions=64, disentangled_from_attr='GT_sbm_id'),
}


def computeDistanceFeatures(G_train, embeddings="All", spatial_ref="GT_pos"):
    to_run = EMBEDDINGS if embeddings == "All" else embeddings
    print("\n--- Graph Enrichment with Embeddings ---")

    for emb in to_run:
        if emb in EMBEDDING_MAPPING:
            print(f"Calculating embeddings via {emb}...")
            EMBEDDING_MAPPING[emb](G_train)
        else:
            print(f"Warning: Algorithm {emb} is not recognized.")
    return G_train


#################################################
######### CROSS VALIDATION FUNCTIONS ############
#################################################

def k_fold_cross_validation(G, k=2, features_list=None, n_trials=50, GroundTruth =None, graph_name="G_NAME"):
    folds_data = _prepare_precalculated_folds(G, k=k, GroundTruth=GroundTruth)
    study = _run_optuna_tuning(folds_data, features_list, n_trials=n_trials)
    
    results = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            results.append({
                'Trial': trial.number,
                'Avg_AUC': trial.value,
                'Std_AUC': trial.user_attrs.get('std_auc'),
                'Avg_AP': trial.user_attrs.get('avg_ap'),
                'Delta_AUC': trial.user_attrs.get('delta_auc'),
                'Params': trial.params
            })
    
    summary_df = pd.DataFrame(results).sort_values(by='Avg_AUC', ascending=False)

    print("\n" + "="*80)
    print(f"{'OPTUNA RESULTS: BASELINE VS TOP CONFIGURATIONS':^80}")
    print("="*80)

    cols = ['Trial', 'Avg_AUC', 'Std_AUC', 'Avg_AP', 'Delta_AUC']
    print(summary_df[summary_df['Trial'] == 0][cols].to_string(index=False))
    print("-" * 80)
    print(summary_df.head(3)[cols].to_string(index=False))
    print("="*80)

    save_dir = "outputs/results"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"optuna_results_{graph_name}.csv"
    full_path = os.path.join(save_dir, filename)
    summary_df.to_csv(full_path, index=False)
    print(f"Results saved in: {full_path}")

    best_params = study.best_params.copy()
    best_params.update({'tree_method': 'hist', 'n_estimators': 150})
    
    return best_params, summary_df

def _process_single_fold(f_idx, t_idx, v_idx, edges, nodes_data, GroundTruth=None):
    print(f"--- Parallel Start Fold {f_idx + 1} ---")
    # Building the kept graph
    kept_edges = [edges[i] for i in t_idx]
    G_kept = nx.Graph()
    G_kept.add_nodes_from(nodes_data)
    G_kept.add_edges_from(kept_edges)

    # Splitting into train/test graph
    G_train, G_test = hide_graph_links(G_kept, test_size=0.15)
    
    # G_hidden: for the validation set
    hidden_edges = [edges[i] for i in v_idx]
    G_hidden = nx.Graph()
    G_hidden.add_nodes_from(nodes_data)
    G_hidden.add_edges_from(hidden_edges)

    # Enriching the training graph
    G_train = computeDistanceFeatures(G_train)
    G_train = computeCommunityFeatures(G_train)

    # Enriching the final validation graph
    G_kept = computeDistanceFeatures(G_kept)
    G_kept = computeCommunityFeatures(G_kept)

    # Creating datasets
    ds_train = prepare_balanced_data(G_test, G_train, negative_ratio=10.0, GroundTruth=GroundTruth) 
    ds_val = prepare_balanced_data(G_hidden, G_kept, negative_ratio=25.0, GroundTruth=GroundTruth)
    
    return (ds_train, ds_val)

def _prepare_precalculated_folds(G, k=1, GroundTruth = None):
    edges = list(G.edges())
    nodes_data = list(G.nodes(data=True))

    if k == 1:
        folds_idx = [train_test_split(range(len(edges)), test_size=0.2, random_state=42)]
    else:
        kf = KFold(n_splits=k, shuffle=True)
        folds_idx = list(kf.split(edges))

    print(f"[K-FOLD] Sequential preparation of {len(folds_idx)} folds...")

    # Formerly parallelized, more efficient this way to avoid nested parallelization.
    precalculated_folds = [
        _process_single_fold(i, t_idx, v_idx, edges, nodes_data, GroundTruth=GroundTruth)
        for i, (t_idx, v_idx) in enumerate(folds_idx)
    ]
    
    return precalculated_folds

def _run_optuna_tuning(precalculated_folds, features_list=None, n_trials=50, n_jobs = -2):

    if features_list is None or len(features_list) == 0:
        exclude = ['u', 'v', 'target', 'label']
        features = [
            col for col in precalculated_folds[0][0].columns
            if (col not in exclude and not col.startswith('GT_'))
            #or col in ['GT_sbm_density', 'GT_pos_dist','GT_spatial_deg_product', 'GT_sbm_deg_product']
        ]
        print(f"Features detected ({len(features)}) : {features}")
    else:
        features = features_list

    optimized_folds = []
    for ds_train, ds_val in precalculated_folds:
        optimized_folds.append({
            'X_train': ds_train[features].values.astype('float32'),
            'y_train': ds_train['target'].values,
            'X_val': ds_val[features].values.astype('float32'),
            'y_val': ds_val['target'].values
        })

    total_cores = os.cpu_count() or 1
    if n_jobs < 0:
        n_jobs = max(1, total_cores + n_jobs)
    else:
        n_jobs = min(n_jobs, total_cores) if n_jobs > 0 else total_cores

    def objective(trial):
        params = {
            'n_estimators': 150,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'tree_method': 'hist',
            "n_jobs" : n_jobs,
            'random_state': 42
        }

        f_auc_v, f_auc_t, f_ap_v = [], [], []

        for fold in optimized_folds:
            model = XGBClassifier(**params)
            model.fit(fold['X_train'], fold['y_train'])

            p_val = model.predict_proba(fold['X_val'])[:, 1]
            p_train = model.predict_proba(fold['X_train'])[:, 1]
            
            f_auc_v.append(roc_auc_score(fold['y_val'], p_val))
            f_auc_t.append(roc_auc_score(fold['y_train'], p_train))
            f_ap_v.append(average_precision_score(fold['y_val'], p_val))
        
        avg_auc_v = np.mean(f_auc_v)
        trial.set_user_attr("std_auc", np.std(f_auc_v))
        trial.set_user_attr("avg_ap", np.mean(f_ap_v))
        trial.set_user_attr("delta_auc", np.mean(f_auc_t) - avg_auc_v)

        del model 
        gc.collect()

        return avg_auc_v

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # To keep error logs only from optuna
    study = optuna.create_study(direction='maximize')
    baseline = {'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 6,
        'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_alpha': 1e-3, 'reg_lambda': 1.0
    }
    study.enqueue_trial(baseline)
    study.optimize(objective, n_trials=n_trials)
    
    return study


########################################
## LOAD AND SAVE UTILITY FUNCTIONS #####
########################################

def save_dataset(dataset, filename="dataset"):
    output_dir = os.path.join(os.getcwd(), "your_results", "data")
    output_path = os.path.join(output_dir, filename)
    
    # Directory creation (absolute)
    os.makedirs(output_dir, exist_ok=True)
    
    dataset.to_parquet(output_path, index=False)
    print(f"Dataset (DataFrame) saved: {output_path}")

    return output_path

def load_dataset(filename="dataset", talk = False):
    input_dir = os.path.join(os.getcwd(), "your_results", "data")
    input_path = os.path.join(input_dir, filename)
    
    if not os.path.exists(input_path) :
        print(f"Error: File does not exist: {input_path}")
        return None
    
    dataset = pd.read_parquet(input_path)
    if talk :
        print(f" Dataset successfully loaded from: {input_path}")
        print(f" Size: {dataset.shape[0]} rows, {dataset.shape[1]} columns.")
    
    return dataset


def loadsave_data_joblib(data=None, filename="data.joblib", mode="save", talk=False):
    """
    Manages saving and loading of objects in .joblib format (SHAP, XGBoost, etc.).
    """
    base_path = Path.cwd()
    target_path = base_path / "your_results" / "data" / filename

    if mode == "save":
        if data is None :
            print("Error: No object provided for saving.")
            return None
        
        # Directory creation
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(data, target_path, compress=3)
        if talk :
            print(f"Object saved in: {target_path}")
        return target_path

    elif mode == "load":
        if not target_path.exists():
            raise FileNotFoundError(f"File not found: {target_path}")
        
        obj = joblib.load(target_path)
        if talk :
            print(f"Object successfully loaded from: {target_path}")
        
        return obj

def load_all_data_for_graph(G_name, talk=False):
    # 1. G_train (with structure, communities and distances)
    try:
        G_train = loadsave_data_joblib(data=None, filename=f"G_train_w_struct_com_dist_{G_name}", mode="load", talk = talk)
    except Exception:
        #print(f"G_train not found for {G_name}, creating an empty graph.")
        G_train = nx.Graph()

    # 2. Train Dataset (via load_dataset)
    try:
        dataset_train = load_dataset(filename=f"dataset_train_{G_name}", talk = talk)
    except Exception:
        print(f"Train Dataset not found for {G_name}.")
        dataset_train = None

    # 3. Evaluation Dataset (via load_dataset)
    try:
        dataset_hidden = load_dataset(filename=f"dataset_hidden_{G_name}", talk = talk)
    except Exception:
        print(f"Evaluation Dataset not found for {G_name}.")
        dataset_hidden = None

    # 4. XGBoost Data (Model, X_test, etc.)
    try:
        xgboost_data = loadsave_data_joblib(data=None, filename=f"xgboost_data_{G_name}.joblib", mode="load", talk = talk)
    except Exception:
        #print(f"XGBoost data not found for {G_name}.")
        xgboost_data = None

    return G_train, dataset_train, dataset_hidden, xgboost_data

class GraphEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def load_graphml_safe(path, speak=False):
    with open(path, 'r', encoding='utf-8') as f:
        raw_data = f.read()

    clean_data = html.unescape(raw_data)
    G = nx.read_graphml(io.StringIO(clean_data))

    if speak : 
        print(f"Graph loaded: {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    return G

def save_graph(G, filename):
    base_path = Path.cwd()
    target_path = base_path / "your_results" / "data" / filename

    data = nx.node_link_data(G)
    with open(filename, 'w') as f:
        json.dump(data, f, cls=GraphEncoder)
    print(f"Graph saved in {filename}")

def load_graph(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data)