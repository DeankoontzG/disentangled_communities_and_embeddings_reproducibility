from pipeline_utils import *
from models import *
import numpy as np
import pandas as pd
import json
import os
import sys
import platform
import subprocess
import scipy.stats as stats
from joblib import Parallel, delayed

import matplotlib
if os.environ.get('DISPLAY','') == '':
    print("No display found. Running in headless mode (saving plots to disk only).")
    matplotlib.use('Agg')
else:
    print("Display detected. Plots will open in a new window.")
    matplotlib.use('TkAgg')
    
import matplotlib.pyplot as plt
import seaborn as sns

def compute_features(G, G_name, spatial_ref = "GT_pos"):

    validate_input_graph(G)
    print("[PREP] Validation du Graphe terminée. Lancement des calculs...")

    if 'GroundTruth_JSON' in G.graph:
        print(f"[INIT] Extraction de la GT GroundTruth pour {G_name}...")
        gt_raw = json.loads(G.graph['GroundTruth_JSON'])
    
        GT = {
            'GT_pos': np.array(gt_raw.get('GT_pos', [])),
            'GT_sbm_id': np.array(gt_raw.get('GT_sbm_id', [])),
            'GT_sbm_matrix': np.array(gt_raw.get('GT_sbm_matrix', []))
            
        }
    else:
        print("[WARNING] Aucune GroundTruth_JSON trouvée dans G.graph")
        GT = None

    if GT is not None and 'GT_pos' in GT:
            for i, node_id in enumerate(G.nodes()):
                G.nodes[node_id]['GT_pos'] = GT['GT_pos'][i]
                G.nodes[node_id]['GT_sbm_id'] = GT['GT_sbm_id'][i]
    
    G_kept, G_hidden = hide_graph_links(G, test_size=0.10)
    G_train, G_test = hide_graph_links(G_kept, test_size=0.15)
        
    G_train_with_communities = computeCommunityFeatures(G_train, spatial_ref=spatial_ref)
    G_kept_with_communities = computeCommunityFeatures(G_kept, spatial_ref=spatial_ref)

    G_train_with_communities = computeDistanceFeatures(G_train, spatial_ref=spatial_ref)
    G_kept_with_communities = computeDistanceFeatures(G_kept, spatial_ref=spatial_ref)
        
    print("Save : graph pour exploration")
    loadsave_data_joblib(data=G_kept_with_communities, filename=f"G_kept_w_struct_com_dist_{G_name}", mode="save")
    loadsave_data_joblib(data=G_train_with_communities, filename=f"G_train_w_struct_com_dist_{G_name}", mode="save")
    
    dataset_train = prepare_balanced_data(G_test, G_train_with_communities,  negative_ratio=10.0, GroundTruth=GT)
    dataset_hidden = prepare_balanced_data(G_hidden, G_kept_with_communities, negative_ratio=50.0, GroundTruth=GT)

    print("Vérif : colonnes du dataset :")
    print(dataset_train.columns)

    print("Sauvegarde des datasets")
    save_dataset(dataset=dataset_train, filename=f"dataset_train_{G_name}")
    save_dataset(dataset=dataset_hidden, filename=f"dataset_hidden_{G_name}")

def analyze_features(G_name_short, nb_iterations, spatial_ref = "GT_pos", i_min =0.00, i_max = 1.00, nb_i=11, name_export_results="DATE"):

    features_GT_pos = ['GT_pos_dist']
    features_GT_sbm = ["GT_sbm_density"]
    features_louvain = ["louvain_density"]
    features_spatial_disentangled_louvain = ["spatial_disentangled_louvain_density"]
    features_community_disentangled_louvain = ["community_disentangled_louvain_density"]
    features_deepwalk = ["deepwalk_dist"]
    features_spatial_disentangled_embed = ["spatial_disentangled_embed_dist"]
    features_community_disentangled_embed = ["community_disentangled_embed_dist"]

    experiments = {
        "GT_pos": features_GT_pos,
        "GT_sbm": features_GT_sbm,
        "GT_pos + GT_sbm": features_GT_pos + features_GT_sbm,
        "Louvain": features_louvain,
        "spatial_disentangled_louvain": features_spatial_disentangled_louvain,
        "community_disentangled_louvain": features_community_disentangled_louvain,
        "deepwalk": features_deepwalk,
        "spatial_disentangled_embed": features_spatial_disentangled_embed,
        "community_disentangled_embed": features_community_disentangled_embed,

        "GT_pos + louvain": features_GT_pos + features_louvain,
        "GT_pos + spatial_disentangled_louvain": features_GT_pos + features_spatial_disentangled_louvain,
        "GT_sbm + louvain": features_GT_sbm + features_louvain,
        "GT_sbm + community_disentangled_louvain": features_GT_sbm + features_community_disentangled_louvain,
        "GT_pos + deepwalk": features_GT_pos + features_deepwalk,
        "GT_pos + spatial_disentangled_embed": features_GT_pos + features_spatial_disentangled_embed,
        "GT_sbm + community_disentangled_embed": features_GT_sbm + features_community_disentangled_embed,
        "GT_sbm + deepwalk": features_GT_sbm + features_deepwalk,
    }

    all_results = []

    tasks = [
        (nb_iter, i) 
        for nb_iter in range(1, nb_iterations + 1) 
        for i in np.linspace(i_max, i_min, nb_i)
    ]

    cores_to_use = max(1, os.cpu_count() -2)

    print(f"Lancement de la parallélisation sur {cores_to_use} coeurs pour {len(tasks)} tâches...")

    # Exécution parallèle
    results_nested = Parallel(n_jobs=cores_to_use)(
        delayed(run_single_experiment)(nb_iter, i, spatial_ref, G_name_short, experiments) 
        for nb_iter, i in tasks
    )

    # Aplatir la liste de listes
    all_results = [item for sublist in results_nested for item in sublist]

    all_results = pd.DataFrame(all_results)
    
    output_dir = "your_results/data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"link_prediction_perfs_{G_name_short}_{nb_iterations}iter_{name_export_results}.csv")
    all_results.to_csv(output_path, index=False)
    print(f" Succès ! Fichier sauvegardé dans : {output_path}")
    return all_results

def run_single_experiment(nb_iter, i, spatial_ref, G_name_short, experiments):
        """
        Fonction exécutée par un cœur unique pour une valeur de i et une itération donnée.
        """
        sbm_val = f"{i:.2f}"
        pos_val = f"{1-i:.2f}"
        if spatial_ref == "GT_Pos" or spatial_ref == "GT_pos" : 
            spatial_ref = ""
        else :
            spatial_ref = f"_{spatial_ref}"
        G_name = f"{G_name_short}_{sbm_val.replace('.', '_')}_pos_{pos_val.replace('.', '_')}_{nb_iter}{spatial_ref}"
    

        # 1. Chargement des données d'entraînement
        _, dataset_train, dataset_eval, _ = load_all_data_for_graph(G_name)
        local_results = []

        for exp_name, feat_list in experiments.items():
            missing = set(feat_list) - set(dataset_train.columns)
            if missing:
                print(f" Exp {exp_name} : colonnes manquantes {missing}. Skip.")
                print(set(dataset_train.columns))
                continue

            #print(f" Running: {exp_name} for SBM={i}")
        
            Params = {
                'max_depth': 3,             # Faible profondeur pour éviter l'overfitting sur 2 variables
                'learning_rate': 0.1,       # Compromis idéal vitesse/précision
                'n_estimators': 1000,       # On met beaucoup, l'early stopping fera le reste
                'subsample': 1.0,           # On garde 100% des lignes (plus stable pour peu de features)
                'colsample_bytree': 1.0,    # On garde les 2 features à chaque split
                'objective': 'binary:logistic', 
                'tree_method': 'hist',      # Accélère l'entraînement sur de gros datasets
                'reg_lambda': 1,            # Régularisation L2 pour stabiliser les poids
                'n_jobs': 1                # Utilise 1 seul coeur, pour la parallélisation
            }

            stats_df, model, _, _, _, _ = train_and_test_xgboost(dataset_train, features=feat_list, parameters = Params, plot=False)

            importances = model.feature_importances_
            feat_imp_series = pd.Series(importances, index=feat_list).sort_values(ascending=False)
                
            # Évaluation sur le dataset de référence FIXE (Graphe SBM 1.0)
            X_eval_fixed = dataset_eval[feat_list] 
            stats_eval_df = get_performance_metrics(model, X_eval_fixed, dataset_eval["target"], "EXP_")
            
            local_results.append({
                "G_name" : G_name,
                "Ratio_SBM": i,
                "Iter": nb_iter,
                "Experiment": exp_name,
                "AP_train": stats_df["Test_AP"].iloc[0],
                "AUC-ROC_train": stats_df["Test_AUC-ROC"].iloc[0],
                "AP_eval": stats_eval_df["EXP_AP"].iloc[0],
                "AUC-ROC_eval": stats_eval_df["EXP_AUC-ROC"].iloc[0],
                "Top_Feature": feat_imp_series.index[0], # On stocke la #1 pour analyse
                "Top_Importance": feat_imp_series.iloc[0]
            })

        return local_results

#############################################
### FONCTIONS POUR AFFICHER LES RESULTATS ###
#############################################

def confidence_interval_95(data):
    n = len(data)
    if n < 2: return 0
    sem = stats.sem(data)
    return sem * stats.t.ppf((1 + 0.95) / 2., n - 1)

def generate_and_show_plot(csv_path, metric="AUC-ROC_eval", name="plot_output"):
    """Loads data, aggregates performance metrics, and displays clean side-by-side plots."""
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' could not be found.")
        return

    df_compare = pd.read_csv(csv_path)

    # All targeted experiments
    feat_to_plot = [
        "GT_pos", "GT_sbm", "GT_pos + GT_sbm",
        "Louvain", "spatial_disentangled_louvain", "community_disentangled_louvain",
        "deepwalk", "spatial_disentangled_embed", "community_disentangled_embed",
        "GT_pos + louvain", "GT_pos + spatial_disentangled_louvain",
        "GT_sbm + louvain", "GT_sbm + community_disentangled_louvain",
        "GT_pos + deepwalk", "GT_pos + spatial_disentangled_embed",
        "GT_sbm + community_disentangled_embed", "GT_sbm + deepwalk"
    ]
    
    df_compare = df_compare[df_compare["Experiment"].isin(feat_to_plot)]

    # 1. Aggregation (Assuming confidence_interval_95 is defined elsewhere)
    metrics = ["AP_train", "AUC-ROC_train", "AP_eval", "AUC-ROC_eval"] 
    df_avg = df_compare.groupby(["Ratio_SBM", "Experiment"])[metrics].agg([
        ('mean', np.mean),
        ('ci95', confidence_interval_95)
    ]).reset_index()

    df_avg.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_avg.columns.values]

    # --- LATEX CONFIGURATION (SVProc style) ---
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,             
        "axes.labelsize": 18,        
        "xtick.labelsize": 15,       
        "ytick.labelsize": 15,       
        "figure.titlesize": 20,
        "legend.fontsize": 11,       
        "legend.title_fontsize": 13  
    })
    
    # Create the side-by-side layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    sns.set_style("whitegrid")

    # --- MAPPING STRICT ---
    labels_mapping = {
        "GT_pos": "GT Pos",
        "GT_sbm": "GT SBM",
        "GT_pos + GT_sbm": "Theoretical max (GT pos + GT SBM)",
        "Louvain": "Standard Louvain",
        "spatial_disentangled_louvain": "Spatial-disentangled Louvain",
        "community_disentangled_louvain": "Community-disentangled Louvain",
        "deepwalk": "DeepWalk",
        "spatial_disentangled_embed": "Spatial-disentangled embeddings",
        "community_disentangled_embed": "Community-disentangled embeddings",
        "GT_pos + louvain": "GT Pos + standard Louvain",
        "GT_pos + spatial_disentangled_louvain": "GT Pos + spatial-disentangled Louvain",
        "GT_sbm + louvain": "GT SBM + standard Louvain",
        "GT_sbm + community_disentangled_louvain": "GT SBM + community-disentangled Louvain",
        "GT_pos + deepwalk": "GT Pos + DeepWalk",
        "GT_pos + spatial_disentangled_embed": "GT Pos + spatial-disentangled embeddings",
        "GT_sbm + community_disentangled_embed": "GT SBM + community-disentangled embeddings",
        "GT_sbm + deepwalk": "GT SBM + DeepWalk"
    }
    
    DESIRED_ORDER = [
        "Theoretical max (GT pos + GT SBM)",
        "GT Pos",
        "GT SBM",
        "Standard Louvain",
        "Spatial-disentangled Louvain",
        "Community-disentangled Louvain",
        "GT Pos + standard Louvain",
        "GT Pos + spatial-disentangled Louvain",
        "GT SBM + standard Louvain",
        "GT SBM + community-disentangled Louvain",
        "DeepWalk",
        "Spatial-disentangled embeddings",
        "Community-disentangled embeddings",
        "GT Pos + DeepWalk",
        "GT Pos + spatial-disentangled embeddings",
        "GT SBM + DeepWalk", 
        "GT SBM + community-disentangled embeddings"
    ]
    
    color_mapping = {
        "Theoretical max (GT pos + GT SBM)": "#000000",
        "GT Pos": "#4d4d4d",
        "GT SBM": "#808080",
        
        "Standard Louvain": "#b3b3b3",
        "DeepWalk": "#b3b3b3",

        "Spatial-disentangled Louvain" : "#762a83",
        "Spatial-disentangled embeddings" : "#762a83",

        "Community-disentangled Louvain" : "#d62728",
        "Community-disentangled embeddings" : "#d62728",
        
        "GT Pos + standard Louvain": "#6caed6",
        "GT Pos + DeepWalk": "#6caed6",
        
        "GT Pos + spatial-disentangled Louvain": "#55a868",
        "GT Pos + spatial-disentangled embeddings": "#55a868",
        
        "GT SBM + standard Louvain": "#af8dc3",
        "GT SBM + DeepWalk": "#af8dc3",
        
        "GT SBM + community-disentangled Louvain": "#e18752",
        "GT SBM + community-disentangled embeddings": "#e18752"
    }

    linestyle_mapping = {
        "Theoretical max (GT pos + GT SBM)": "--",
        "GT Pos": "-.",
        "GT SBM": ":",
        "Standard Louvain": "-",
        "Spatial-disentangled Louvain": "-",
        "Community-disentangled Louvain": "-",
        "DeepWalk": "-",
        "Spatial-disentangled embeddings": "-",
        "Community-disentangled embeddings": "-",
        "GT Pos + standard Louvain": "-",
        "GT Pos + DeepWalk": "-",
        "GT Pos + spatial-disentangled Louvain": "-",
        "GT Pos + spatial-disentangled embeddings": "-",
        "GT SBM + standard Louvain": "-",
        "GT SBM + DeepWalk": "-",
        "GT SBM + community-disentangled Louvain": "-",
        "GT SBM + community-disentangled embeddings": "-"
    }
    
    marker_mapping = {
        # References (No markers)
        "Theoretical max (GT pos + GT SBM)": None,
        "GT Pos": None,
        "GT SBM": None,
        
        # Baselines Lone Algos
        "Standard Louvain": "P",
        "DeepWalk": "P",
        "Spatial-disentangled Louvain": "^",
        "Spatial-disentangled embeddings": "^",
        "Community-disentangled Louvain": "d",
        "Community-disentangled embeddings": "d",
        
        # Combinations with GT Pos (Left side geometries)
        "GT Pos + standard Louvain": "o",
        "GT Pos + DeepWalk": "o",
        "GT Pos + spatial-disentangled Louvain": "v",
        "GT Pos + spatial-disentangled embeddings": "v",
        
        # Combinations with GT SBM (Right side geometries)
        "GT SBM + standard Louvain": "X",
        "GT SBM + DeepWalk": "X",
        "GT SBM + community-disentangled Louvain": "s",
        "GT SBM + community-disentangled embeddings": "s"
    }

    mean_col = f"{metric}_mean"
    ci_col = f"{metric}_ci95"

    # --- SPLIT DATASETS FOR PLOTTING ---
    # Left subplot (communities inference features)
    left_experiments = [
        "GT_pos", "GT_sbm", "GT_pos + GT_sbm", "Louvain", "community_disentangled_louvain", "spatial_disentangled_louvain", 
        "GT_pos + louvain", "GT_sbm + louvain", "GT_pos + spatial_disentangled_louvain", "GT_sbm + community_disentangled_louvain"
    ]
    df_left = df_avg[df_avg["Experiment"].isin(left_experiments)]

    # Right subplot (embeddings inference features)
    right_experiments = [
        "GT_pos", "GT_sbm", "GT_pos + GT_sbm", "deepwalk", "community_disentangled_embed", "spatial_disentangled_embed",
        "GT_pos + deepwalk", "GT_sbm + deepwalk", "GT_pos + spatial_disentangled_embed", "GT_sbm + community_disentangled_embed"
    ]
    df_right = df_avg[df_avg["Experiment"].isin(right_experiments)]

    def plot_subplot(ax, df, title_label):
        ax.grid(True, linestyle="--", alpha=0.6)
        experiments = df['Experiment'].unique()
        
        for exp in experiments:
            subset = df[df["Experiment"] == exp]
            clean_label = labels_mapping.get(exp, exp)
            color = color_mapping.get(clean_label, "#7f7f7f")
            linestyle = linestyle_mapping.get(clean_label, "-")
            marker = marker_mapping.get(clean_label, None)
            
            ax.plot(subset["Ratio_SBM"], subset[mean_col], 
                    marker=marker, linestyle=linestyle,
                    label=clean_label, color=color, linewidth=2.5, markersize=6)
            
            ax.fill_between(
                subset["Ratio_SBM"], 
                subset[mean_col] - subset[ci_col], 
                subset[mean_col] + subset[ci_col], 
                color=color, alpha=0.08
            )
        
        ax.set_xlabel(r"$\alpha$ (SBM weight ratio)", labelpad=10)
        ax.set_title(title_label, fontsize=16, pad=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.48, 1.02)
        
        # --- REORGANIZE LEGEND ---
        handles, labels = ax.get_legend_handles_labels()
        handle_dict = dict(zip(labels, handles))
        
        sorted_handles = []
        sorted_labels = []
        for exact_label in DESIRED_ORDER:
            if exact_label in handle_dict:
                sorted_handles.append(handle_dict[exact_label])
                sorted_labels.append(exact_label.replace('_', '\_'))
        
        for label, handle in handle_dict.items():
            if label not in DESIRED_ORDER:
                sorted_handles.append(handle)
                sorted_labels.append(label.replace('_', '\_'))

        ax.legend(
            sorted_handles,
            sorted_labels,
            loc='lower center', 
            bbox_to_anchor=(0.5, 1.12), 
            ncol=2,
            title="Feature Configurations", 
            frameon=True,              
            facecolor='white',         
            edgecolor='black',         
            framealpha=1.0,            
            fancybox=False,
            columnspacing=0.8,         
            handletextpad=0.4          
        )

    # Draw the two subplots
    plot_subplot(ax1, df_left, "Communities inferences:")
    plot_subplot(ax2, df_right, "Embedding inferences:")
    
    ax1.set_ylabel(f"{metric.replace('_', ' ')}", labelpad=10)
    
    plt.subplots_adjust(wspace=0.15) 
    
    # Save Layouts
    output_dir = os.path.join("your_results", "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    save_path_png = os.path.join(output_dir, f"{name}_{metric}_READY.png")
    plt.savefig(save_path_png, bbox_inches='tight', dpi=300)
    save_path_pdf = os.path.join(output_dir, f"{name}_{metric}_READY.pdf")
    plt.savefig(save_path_pdf, bbox_inches='tight')
    
    print("="*80)
    print(f"Plots successfully saved to:\n -> {save_path_png}\n -> {save_path_pdf}")
    
    print("Opening plot window...")
    plt.ion()  
    plt.show(block=True)