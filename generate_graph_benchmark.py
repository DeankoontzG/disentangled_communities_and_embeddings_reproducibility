import pandas as pd
import numpy as np
import networkx as nx
import graph_tool.all as gt
from graph_tool.spectral import adjacency
import json
import os
import joblib
from collections import Counter
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import fsolve, minimize_scalar

# =========================================================================
# GLOBAL HYPERPARAMETERS
# =========================================================================
GENERATION_MODE = "REALISTIC"  #REALISTIC or STANDARD, please read our article for more information
BENCHMARK_SIZE = 30
HYBRID_RATIO_LIST = np.arange(1.00, -0.10, -0.10)
HYBRIDIZATION_METHOD = "SUM"  # "SUM" or "POWER", please read our article for more information
GRAPH_NAMES = f"artificial_{GENERATION_MODE.lower()}_graph_{HYBRIDIZATION_METHOD.lower()}"

# Parameters for "STANDARD" mode (ignored if mode is "REALISTIC")
N_NODES = 198
N_COMMUNITIES = 9  # 9 communities * 22 nodes = 198 nodes (perfectly egalitarian)
P_IN = 1.00
P_OUT = 0.00

# Parameters for "REALISTIC" mode (ignored if mode is "ANALYTIC")
REAL_GRAPH_NAME = "reel_jazz_collab_w_attributes.joblib"

#######################################
###### PROPERTY EXTRACTION / GEN ######
#######################################

def get_analytic_properties():
    """Generates pure analytical SBM and spatial properties."""
    nodes_per_comm = N_NODES // N_COMMUNITIES
    comm_labels = np.array([i // nodes_per_comm for i in range(N_NODES)])
    comm_labels = np.clip(comm_labels, 0, N_COMMUNITIES - 1)
    
    P_sbm = np.zeros((N_NODES, N_NODES))
    for i in range(N_NODES):
        for j in range(i + 1, N_NODES):
            if comm_labels[i] == comm_labels[j]:
                P_sbm[i, j] = P_IN
                P_sbm[j, i] = P_IN
            else:
                P_sbm[i, j] = P_OUT
                P_sbm[j, i] = P_OUT
                
    counts = np.bincount(comm_labels)
    intra_pairs = sum(c * (c - 1) // 2 for c in counts)
    total_pairs = N_NODES * (N_NODES - 1) // 2
    inter_pairs = total_pairs - intra_pairs
    expected_links = (intra_pairs * P_IN) + (inter_pairs * P_OUT)
    
    print(f"[SBM] {N_COMMUNITIES} blocks of {counts[0]} nodes. Theoretical expected links: {expected_links:.1f}")
    
    # Fake empirical matrix e_rs for metadata tracking
    e_rs = np.zeros((N_COMMUNITIES, N_COMMUNITIES))
    for r in range(N_COMMUNITIES):
        for s in range(N_COMMUNITIES):
            if r == s:
                e_rs[r, s] = P_IN * (counts[r] * (counts[r] - 1))
            else:
                e_rs[r, s] = P_OUT * (counts[r] * counts[s])
                
    # Uniform spatial positioning
    rng = np.random.default_rng(seed=42)
    positions = rng.uniform(0.0, 1.0, size=(N_NODES, 2))
    
    # Analytical degrees vector matching expected links context
    degrees = np.full(N_NODES, int(expected_links * 2 / N_NODES))
    k_sbm = degrees.copy()
    
    return P_sbm, comm_labels, degrees, positions, k_sbm, e_rs

def get_real_graph_properties_sbm(G_train):
    """Extracts blockmodel attributes from an empirical graph."""
    nodes_map = {node: i for i, node in enumerate(G_train.nodes())}
    edges = [(nodes_map[u], nodes_map[v]) for u, v in G_train.edges()]    
    communities = nx.get_node_attributes(G_train, 'sbm_id')
    unique_comms = sorted(list(set(communities.values())))
    mapping = {raw_id: i for i, raw_id in enumerate(unique_comms)}
    
    g = gt.Graph(directed=False)
    g.add_vertex(len(G_train.nodes()))
    g.add_edge_list(edges)
    
    b_array = np.array([mapping[communities[node]] for node in G_train.nodes()])
    b_prop = g.new_vertex_property("int", b_array)
    state = gt.BlockState(g, b=b_prop, deg_corr=True)

    e_rs = state.get_matrix().toarray()
    k_sbm = g.get_out_degrees(g.get_vertices())
    
    return e_rs, k_sbm, b_array

def get_real_graph_properties_pos(G_train, n_components=4, shuffle=True):
    """Extracts spatial positions from empirical graph representations."""
    nodes = list(G_train.nodes())
    embeddings_attr = nx.get_node_attributes(G_train, 'deepwalk')
    raw_embeddings = np.array([embeddings_attr[node] for node in nodes])
    degrees = np.array([G_train.degree(node) for node in nodes])
    
    pca = PCA(n_components=n_components, random_state=42)
    pos_reduced = pca.fit_transform(raw_embeddings)
    
    pos_min = pos_reduced.min(axis=0)
    pos_max = pos_reduced.max(axis=0)
    pos_normalized = (pos_reduced - pos_min) / (pos_max - pos_min)
        
    if shuffle:
        rng_pos = np.random.default_rng(seed=42)
        idx_pos = rng_pos.permutation(len(pos_normalized))
        pos_final = pos_normalized[idx_pos]
        
        rng_deg = np.random.default_rng(seed=99) 
        idx_deg = rng_deg.permutation(len(degrees))
        degrees_final = degrees[idx_deg]
    else:
        pos_final = pos_normalized
        degrees_final = degrees
        
    return degrees_final, pos_final

def get_probs_sbm_non_DC(e_rs, b):
    """Computes standard SBM probabilities matrix without degree corrections."""
    n = len(b)
    n_blocks = e_rs.shape[0]
    P = np.zeros((n, n))
    counts = np.bincount(b)
    
    for r in range(n_blocks):
        for s in range(r, n_blocks):
            idx_r = np.where(b == r)[0]
            idx_s = np.where(b == s)[0]
            
            if r == s:
                possible = counts[r] * (counts[r] - 1) / 2
                p_rs = e_rs[r, s] / (2 * possible) if possible > 0 else 0
            else:
                possible = counts[r] * counts[s]
                p_rs = e_rs[r, s] / possible if possible > 0 else 0
                
            P[np.ix_(idx_r, idx_s)] = p_rs
            if r != s:
                P[np.ix_(idx_s, idx_r)] = p_rs
                
    np.fill_diagonal(P, 0)
    P = np.clip(P, 0, 1)
    return P

def get_probs_spatial_non_DC(positions, n_liens_target, sigma=1.0):
    """Computes spatial connection probabilities using a logistic deterrence function."""
    n = len(positions)
    dist_matrix = squareform(pdist(positions, 'euclidean'))
    deterrence = sigma * dist_matrix
    iu = np.triu_indices(n, k=1)
    det_vec = deterrence[iu]

    def objective(alpha):
        logits = alpha - det_vec
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.sum(probs) - n_liens_target

    alpha_opt = fsolve(objective, x0=0.0)[0]
    logit_final = alpha_opt - deterrence
    P = 1.0 / (1.0 + np.exp(-logit_final))
    np.fill_diagonal(P, 0)
    return P

#######################################
###### CORE METRIC FUNCTIONS ##########
#######################################

def get_variance_from_P(P):
    n = P.shape[0]
    upper_idx = np.triu_indices(n, k=1)
    return np.var(P[upper_idx])

def get_entropy_from_p(P):
    upper_idx = np.triu_indices_from(P, k=1)
    p_vector = np.clip(P[upper_idx], 1e-12, 1 - 1e-12)
    h_binaire = -(p_vector * np.log2(p_vector) + (1 - p_vector) * np.log2(1 - p_vector))
    return np.sum(h_binaire)

def get_log_likelihood(G, P):
    if isinstance(G, nx.Graph):
        adj = nx.to_numpy_array(G, nodelist=range(len(P)))
    else:
        adj = adjacency(G).toarray()
    upper_idx = np.triu_indices_from(P, k=1)
    p_vector = np.clip(P[upper_idx], 1e-12, 1 - 1e-12)
    adj_vector = adj[upper_idx]
    return np.sum(adj_vector * np.log2(p_vector) + (1 - adj_vector) * np.log2(1 - p_vector))

#######################################
###### HYBRIDIZATION & EXPORT #########
#######################################

def match_spatial_to_sbm_variance(P_sbm, target_links, positions):
    """Optimizes spatial deterrence factor sigma to precisely fit the target SBM variance."""
    target_variance = get_variance_from_P(P_sbm)
    print(f"\n[Calibration] Target variance (SBM): {target_variance:.8f}")
    print("-" * 50)

    history = {'step': 0}

    def objective(sigma_test):
        history['step'] += 1
        P_test = get_probs_spatial_non_DC(positions, n_liens_target=target_links, sigma=sigma_test)
        current_var = get_variance_from_P(P_test)
        diff_pct = (abs(current_var - target_variance) / target_variance) * 100
        
        if history['step'] % 5 == 0 or history['step'] == 1:
            print(f"Step {history['step']:02d} | Sigma: {sigma_test:.4f} | Var: {current_var:.8f} | Discrepancy: {diff_pct:.2f}%")
        return (current_var - target_variance)**2

    res = minimize_scalar(objective, bounds=(0.005, 100.0), method='bounded')
    opt_sigma = res.x
    final_P_spatial = get_probs_spatial_non_DC(positions, n_liens_target=target_links, sigma=opt_sigma)
    final_var = get_variance_from_P(final_P_spatial)
    final_diff_pct = (abs(final_var - target_variance) / target_variance) * 100
    
    print("-" * 50)
    print(f"✨ Optimal Sigma found: {opt_sigma:.4f}")
    print(f"📊 Final Spatial Variance: {final_var:.8f}")
    print(f"📢 FINAL VARIANCE DISCREPANCY: {final_diff_pct:.4f} %")
    print("-" * 50)
    
    return opt_sigma, final_P_spatial

def generate_graph_from_probs(P):
    n = P.shape[0]
    g = gt.Graph(directed=False)
    g.add_vertex(n)
    upper_idx = np.triu_indices(n, k=1)
    probs_vector = P[upper_idx]
    mask = np.random.random(len(probs_vector)) < probs_vector
    edges = np.column_stack((upper_idx[0][mask], upper_idx[1][mask]))
    g.add_edge_list(edges)
    return g

def convert_to_nx_with_metadata(gt_graph, positions, k_sbm, degrees, sbm_labels, e_rs, Probas_mtx=None):
    edges = gt_graph.get_edges()
    n_nodes = len(sbm_labels)
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(n_nodes))
    G_nx.add_edges_from(edges)
    
    gt_payload = {
        'GT_degrees_sbm': k_sbm.tolist() if hasattr(k_sbm, 'tolist') else list(k_sbm),
        'GT_degrees_spatial': degrees.tolist() if hasattr(degrees, 'tolist') else list(degrees),
        'GT_pos': positions.tolist() if hasattr(positions, 'tolist') else list(positions),
        'GT_sbm_id': [int(x) for x in sbm_labels],
    }

    num_blocks = e_rs.shape[0]
    counts = Counter(sbm_labels)
    sbm_density_matrix = np.zeros((num_blocks, num_blocks))
    
    for r in range(num_blocks):
        for s in range(r, num_blocks):
            n_r, n_s = counts[r], counts[s]
            links = e_rs[r, s]
            if r == s:
                possible = n_r * (n_r - 1) / 2
                dens = links / (2 * possible) if possible > 0 else 0
            else:
                possible = n_r * n_s
                dens = links / possible if possible > 0 else 0
            
            sbm_density_matrix[r, s] = dens
            sbm_density_matrix[s, r] = dens
    
    gt_payload['GT_sbm_matrix'] = sbm_density_matrix.tolist()
    G_nx.graph['GroundTruth_JSON'] = json.dumps(gt_payload)
    
    if Probas_mtx is not None:
        G_nx.graph['P_matrix_JSON'] = json.dumps(Probas_mtx.tolist())
    return G_nx

def generate_graph_benchmarks(Hybrid_ratios_list, P_sbm, P_spatial, position, k_sbm, degrees, commu, e_rs, name, nb_iter):
    results_list = []
    all_P_matrices = {}

    for alpha in Hybrid_ratios_list:
        G_name = f"{name}_{f'{alpha:.2f}'.replace('.', '_')}_pos_{f'{1-alpha:.2f}'.replace('.', '_')}_{nb_iter}.graphml"
        
        if HYBRIDIZATION_METHOD == "SUM":
            P_hybride = P_sbm * alpha + P_spatial * (1 - alpha)
        elif HYBRIDIZATION_METHOD == "POWER":
            kpow = 4
            tol = 1e-2
            target_expectation = alpha * np.sum(P_sbm) + (1 - alpha) * np.sum(P_spatial)
            P_base = alpha * (P_sbm**kpow) + (1 - alpha) * (P_spatial**kpow)
            
            j_min, j_max = 0.1, 50.0
            for _ in range(100):  
                j_mid = (j_min + j_max) / 2
                current_exp = np.sum(P_base**j_mid)
                if abs(current_exp - target_expectation) < tol:
                    break
                if current_exp > target_expectation:
                    j_min = j_mid
                else:
                    j_max = j_mid
            P_hybride = P_base**j_mid
            
        alpha_key = round(alpha, 2)
        all_P_matrices[alpha_key] = P_hybride.copy()
        
        g_hybride = generate_graph_from_probs(P_hybride)
        g_hybride_nx = convert_to_nx_with_metadata(g_hybride, position, k_sbm, degrees, commu, e_rs, P_hybride)
        
        os.makedirs("graph_library", exist_ok=True)
        nx.write_graphml(g_hybride_nx, os.path.join("graph_library", G_name))
        
        var_h = get_variance_from_P(P_hybride)
        ent_h = get_entropy_from_p(P_hybride)
        clustering = gt.global_clustering(g_hybride)[0]
        
        results_list.append({
            "Ratio SBM (α)": f"{alpha:.2f}",
            "N": g_hybride.num_vertices(),
            "E": g_hybride.num_edges(),
            "Variance": f"{var_h:.8f}",
            "Entropy": f"{ent_h:.2f}",
            "Clustering": f"{clustering:.4f}"
        })

    print("\n" + "="*70)
    print(f"SUMMARY TABLE - ITERATION {nb_iter}")
    print("="*70)
    print(pd.DataFrame(results_list).to_string(index=False))
    print("="*70)

    return all_P_matrices

# =========================================================================
# MAIN BENCHMARK SCRIPT EXECUTION
# =========================================================================

if __name__ == "__main__":
    print("=========================================================================")
    print(f"LAUNCHING BENCHMARK ENGINES ({GENERATION_MODE} MODE | {HYBRIDIZATION_METHOD})")
    print("=========================================================================")

    G = None
    if GENERATION_MODE == "REALISTIC":
        path = f"graph_library/{REAL_GRAPH_NAME}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source file structure missing at: {path}")
        print(f"Source graph loaded successfully from: {path}")
        G = joblib.load(path)

    for nb_iter in range(1, BENCHMARK_SIZE + 1):
        print(f"\n▶️ GENERATION ITERATION N° {nb_iter} / {BENCHMARK_SIZE}")
        
        if GENERATION_MODE == "STANDARD":
            P_sbm, commus, degrees, position, k_sbm, e_rs = get_analytic_properties()
            target_links = np.sum(P_sbm) / 2
        elif GENERATION_MODE == "REALISTIC":
            e_rs, k_sbm, commus = get_real_graph_properties_sbm(G)
            P_sbm = get_probs_sbm_non_DC(e_rs, commus)
            degrees, position = get_real_graph_properties_pos(G)
            target_links = np.sum(degrees) / 2
        else:
            raise ValueError(f"Unknown GENERATION_MODE: {GENERATION_MODE}")

        # Calibration process matching spatial parameters to targets
        sigma_opt, P_spatial_calibrated = match_spatial_to_sbm_variance(
            P_sbm=P_sbm, 
            target_links=target_links, 
            positions=position
        )
        
        # Generation cycle execution
        all_P_matrices = generate_graph_benchmarks(
            Hybrid_ratios_list=HYBRID_RATIO_LIST,
            P_sbm=P_sbm,
            P_spatial=P_spatial_calibrated,
            position=position,
            k_sbm=k_sbm,
            degrees=degrees,
            commu=commus,
            e_rs=e_rs,
            name=GRAPH_NAMES,
            nb_iter=nb_iter
        )
        
    print(f"\n[Done] Pipeline executed successfully. Benchmark files written inside 'graph_library/'.")