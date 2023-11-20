#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:13:01 2023

@author: tuj
"""

import matplotlib 
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import os
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from lifelines import KaplanMeierFitter
    

def plot_confusion_matrix(confusion_matrix, font_size=15):
    cls_num = len(confusion_matrix)
    label_list = ["Cluster{}".format(k+1) for k in range(cls_num)]
    norm_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]

    annot_box = ["{:.3f}\n".format(i) + "("+str(int(j))+")" for i, j in zip(
            norm_confusion_matrix.flatten(), confusion_matrix.flatten())]
    annot_box = np.asarray(annot_box).reshape(cls_num, cls_num)
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(norm_confusion_matrix, cmap="Blues", annot=annot_box, fmt="", vmin=0, vmax=1.0, 
            annot_kws={"size" : font_size}, xticklabels=label_list, yticklabels=label_list, cbar=False, square=True)
    
    plt.xlabel("Predicted labels", fontsize=15)
    plt.ylabel("True labels", fontsize=15)
    plt.yticks(rotation=0)





if __name__ == "__main__":
    
    seed = 0
    model_seed = 10
    
    path = os.path.join(os.getcwd(), "Preprocessed_TAVI_dataset.xlsx")
    preprocessed_df = pd.read_excel(path)
    
    save_base_path = os.path.join(os.getcwd(), "result")
    save_base_silhouette_path = os.path.join(save_base_path, "silhouette")
    save_base_opt_path = os.path.join(save_base_path, "optimize_clusters")
    save_base_kap_path = os.path.join(save_base_path, "kaplanmeier")
    
    os.makedirs(save_base_path, exist_ok=True)
    os.makedirs(save_base_opt_path, exist_ok=True)
    os.makedirs(save_base_kap_path, exist_ok=True)
    
    
    
    tavi_year_column = preprocessed_df["TAVIyear"]
    time_to_MACE_column = preprocessed_df["time_to_MACE"]
    time_to_allcase_death_column = preprocessed_df["time_to_allcase_death"]
    time_to_CVD_death_column = preprocessed_df["time_to_CVD_death"]
    MACE_allDeath_column = preprocessed_df["MACE+allDeath"]    
    MACE_column = preprocessed_df["MACE"]
    time_to_MACE_death_column = preprocessed_df["time_to_MACE+death"]
    
    #Selecting 20 variables to be used as independent variables.
    target_feature_names = [
        'Age','WBC','Hb','BMI','SBP','DBP','PR',
        'PWTd', 'LVOTd', 'LVEF', 'LV_mass_index',
        'AV_peak_V', 'Mean_PG', 'AVAi', 'TR_velocity',
        'Svi', 'LAD', 'DCT', 'IVC', "E/E'_septal"
        ]
    target_feature_df = preprocessed_df[target_feature_names]
    
    
    #Splitting the TAVI dataset into train and test data.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    sss.get_n_splits(preprocessed_df, y=tavi_year_column)
    
    for i, (train_index, test_index) in enumerate(sss.split(preprocessed_df, tavi_year_column)):
        train_df = preprocessed_df.iloc[train_index, :]
        test_df  = preprocessed_df.iloc[test_index, :]
                
    train_data = train_df[target_feature_names]
    test_data  = test_df[target_feature_names]
    train_tavi_year = train_df["TAVIyear"]
    test_tavi_year = test_df["TAVIyear"]
        

    #Standardization
    sc = StandardScaler()
    sc.fit(train_data)
    train_std_data = sc.transform(train_data)
    test_std_data  = sc.transform(test_data) 
    
    sc.fit(target_feature_df)
    target_std_data = sc.transform(target_feature_df)
    
    
    #Exploring the optimal number of clusters for K-means clustering
    distortions = []
    silhouette_list = []
    aic_score_list, bic_score_list = [], []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, 
                    init="k-means++",
                    n_init=10, 
                    max_iter=300,
                    random_state=model_seed,
                    )
        km.fit(train_std_data)
        train_pred = km.predict(train_std_data)
        test_pred  = km.predict(test_std_data)
        distortions.append(km.inertia_)
        if i == 1:
            continue
        else:
            silhouette_list.append(metrics.silhouette_score(train_std_data, train_pred, metric="euclidean"))
         
    
    #Visualization of SSE & silhoette score 
    font_size = 13
    plt.figure(figsize=(8, 7))
    plt.plot(range(2, 11), distortions[1:], marker="o", color="dodgerblue", linestyle="solid", label="Sum of squared error")
    plt.xlabel("Number of clusters", fontsize=font_size)
    plt.ylabel("Sum of squared error (SSE)", fontsize=font_size)
    plt.xticks(range(2, 11))
    plt.legend(bbox_to_anchor=(1, 1), loc="upper right")
    
    plt.twinx()
    plt.plot(range(2, 11), silhouette_list, marker="s", color="orangered", linestyle="dashed", label="Silhouette score")
    plt.ylabel("Silhouette score", fontsize=font_size)
    plt.legend(bbox_to_anchor=(1, 0.92), loc="upper right")
    plt.savefig(os.path.join(save_base_opt_path, "Kmeans_Elbow_Silhoette.svg"), bbox_inches="tight")
    plt.savefig(os.path.join(save_base_opt_path, "Kmeans_Elbow_Silhoette.jpg"), bbox_inches="tight")
    plt.close()
    
    
    #Set the number of clusters to 3 and execute K-means
    cluster_num = 3
    model = KMeans(n_clusters=cluster_num, init="k-means++",
                   n_init=10, max_iter=300, random_state=model_seed)
    model.fit(train_std_data)
    train_pred = model.predict(train_std_data)
    test_pred  = model.predict(test_std_data)

    CLUSTER_PRED = "kmeans_pred"
    preprocessed_df[CLUSTER_PRED] = 0
    for index, pred in zip(train_df.index, train_pred):
        preprocessed_df[CLUSTER_PRED].loc[index] = pred
    for index, pred in zip(test_df.index, test_pred):
        preprocessed_df[CLUSTER_PRED].loc[index] = pred
    pred_column = preprocessed_df[CLUSTER_PRED]
    
    TIME_TO_MACE = "time_to_MACE"
    TIME_TO_MACE_DEATH = "time_to_MACE+death"
    
    color_name = "Set2"
    colors = sns.color_palette(color_name, cluster_num)
    alpha_value = 0.1
        
   
    #Kmeans visualziation with PCA
    pca = PCA(n_components=2)
    pca.fit(train_std_data)
    train_std_pca_data = pca.transform(train_std_data)
    test_std_pca_data  = pca.transform(test_std_data)
    
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for n, color in enumerate(colors):
        data = train_std_pca_data[train_pred==n][MACE_column.iloc[train_index][train_pred==n]==0]
        plt.scatter(data[:, 0], data[:, 1], s=1.5, color=color)
    for n, color in enumerate(colors):
        data = train_std_pca_data[train_pred==n][MACE_column.iloc[train_index][train_pred==n]==1]
        plt.scatter(data[:, 0], data[:, 1], marker="x", color=color, label="Cluster{}".format(n+1))
    plt.xlabel("Dimension 1", fontsize=font_size)
    plt.ylabel("Dimension 2", fontsize=font_size)
    plt.title("Derivation cohort", fontsize=font_size)
        
    plt.subplot(1, 2, 2)
    for n, color in enumerate(colors):
        data = test_std_pca_data[test_pred==n][MACE_column.iloc[test_index][test_pred==n]==0]
        plt.scatter(data[:, 0], data[:, 1], s=1.5, color=color, label="Cluster{}".format(n+1))
    for n, color in enumerate(colors):
        data = test_std_pca_data[test_pred==n][MACE_column.iloc[test_index][test_pred==n]==1]
        plt.scatter(data[:, 0], data[:, 1], marker="x", color=color, label="Cluster{}-MACE".format(n+1))
    
    plt.xlabel("Dimension 1", fontsize=font_size)
    plt.ylabel("Dimension 2", fontsize=font_size)
    plt.title("Validation cohort", fontsize=font_size)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=font_size)
    
    plt.savefig(os.path.join(save_base_opt_path, "PCA_2dimension.svg"), bbox_inches="tight")
    plt.savefig(os.path.join(save_base_opt_path, "PCA_2dimension.png"), bbox_inches="tight")
    plt.close()
    
    
    #Creating Kaplan-Meier Curve
    #MACE
    plt.figure(figsize=(15, 5.5))
    plt.subplot(1, 2, 1)
    for k in range(cluster_num):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=time_to_MACE_column.iloc[train_index][pred_column==k],
                event_observed=MACE_column.iloc[train_index][pred_column==k],
                label="Cluster {}".format(k+1))
        kmf.plot(color=colors[k], show_censors=True, ci_show=True, ci_alpha=alpha_value)
    plt.ylim([0, 1.05])
    plt.xlabel("Time (day)")
    plt.ylabel("Survival Probability")
    plt.legend(loc="lower left", edgecolor="white")
    plt.title("Train cohort")
    
    plt.subplot(1, 2, 2)
    for k in range(cluster_num):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=time_to_MACE_column.iloc[test_index][pred_column==k],
                event_observed=MACE_column.iloc[test_index][pred_column==k],
                label="Cluster {}".format(k+1))
        #kmf.plot(color=colors(k), show_censors=True, ci_show=True, ci_alpha=alpha_value)
        kmf.plot(color=colors[k], show_censors=True, ci_show=True, ci_alpha=alpha_value)
    plt.ylim([0, 1.05])
    plt.xlabel("Time (day)")
    plt.ylabel("Survival Probability")
    plt.legend(loc="lower left", edgecolor="white")
    plt.title("Test cohort")
    plt.savefig(os.path.join(save_base_kap_path, "KaplanMeier_time_to_MACE.svg"), bbox_inches="tight")
    plt.savefig(os.path.join(save_base_kap_path, "KaplanMeier_time_to_MACE.jpg"), bbox_inches="tight")
    plt.close()
    

    # MACE + All death
    plt.figure(figsize=(15, 5.5))
    plt.subplot(1, 2, 1)
    for k in range(cluster_num):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=time_to_MACE_death_column.iloc[train_index][pred_column==k],
                event_observed=MACE_allDeath_column.iloc[train_index][pred_column==k],
                label="Cluster {}".format(k+1))
        kmf.plot(color=colors[k], show_censors=True, ci_show=True, ci_alpha=alpha_value)
    plt.ylim([0, 1.05])
    plt.xlabel("Time (day)")
    plt.ylabel("Survival Probability")
    plt.legend(loc="lower left", edgecolor="white")
    plt.title("Train cohort")
    
    plt.subplot(1, 2, 2)
    for k in range(cluster_num):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=time_to_MACE_death_column.iloc[test_index][pred_column==k],
                event_observed=MACE_allDeath_column.iloc[test_index][pred_column==k],
                label="Cluster {}".format(k+1))
        kmf.plot(color=colors[k], show_censors=True, ci_show=True, ci_alpha=alpha_value)
    plt.ylim([0, 1.05])
    plt.xlabel("Time (day)")
    plt.ylabel("Survival Probability")
    plt.legend(loc="lower left", edgecolor="white")    
    plt.title("Test cohort")
    plt.savefig(os.path.join(save_base_kap_path, "KaplanMeier_time_to_MACE+death.svg"), bbox_inches="tight")
    plt.savefig(os.path.join(save_base_kap_path, "KaplanMeier_time_to_MACE+death.jpg"), bbox_inches="tight")
    plt.close()
    
    
    #Using the clustering results from K-means as the ground truth labels, 
    #train a logitstic regression

    train_label = train_pred.copy()
    test_label  = test_pred.copy()
    
    model = LogisticRegression()    
    model.fit(train_std_data, train_label)   
    
    train_prob_column = model.predict_proba(train_std_data)
    train_pred_column = model.predict(train_std_data)
    
    test_prob_column = model.predict_proba(test_std_data)    
    test_pred_column = model.predict(test_std_data)        
    
    train_acc_score = metrics.balanced_accuracy_score(train_label, train_pred_column)
    test_acc_score  = metrics.balanced_accuracy_score(test_label, test_pred_column)
    
    train_confusion_matrix = metrics.confusion_matrix(train_label, train_pred_column)
    test_confusion_matrix  = metrics.confusion_matrix(test_label, test_pred_column)
    
    plot_confusion_matrix(train_confusion_matrix)
    plt.savefig(os.path.join(save_base_kap_path, "train_confusion_matrix.svg"), bbox_inches="tight")
    plt.close()
    plot_confusion_matrix(test_confusion_matrix)
    plt.savefig(os.path.join(save_base_kap_path, "test_confusion_matrix.svg"), bbox_inches="tight")
    plt.close()
    
    
    # Visualization of logistic regression coefficients for each cluster.
    coef_ = model.coef_
    target_feature_names_column = np.array(target_feature_names)
    
    
    left = [i for i in range(len(target_feature_names))]
    plt.figure(figsize=(21, 7))
    for k in range(cluster_num):
        plt.subplot(1, 3, k+1)
        plt.barh(left, coef_[k][np.argsort(coef_[k])],
                 tick_label=target_feature_names_column[np.argsort(coef_[k])],
                 color=colors[k])
        plt.xlabel("Partial regression coefficient")
        plt.ylabel("Features")
        plt.title("Cluster{}".format(k+1))
    plt.savefig(os.path.join(save_base_kap_path, "LogisticRegression_Coefficient.svg"), bbox_inches="tight")
    plt.close()
    
    
    
    #Saving the analysis results
    preprocessed_df["train_test"] = 0
    preprocessed_df["train_test"].iloc[train_index] = "Train"
    preprocessed_df["train_test"].iloc[test_index] = "Test"
    
    pred_column = preprocessed_df[CLUSTER_PRED]
    train_mean_std_list, test_mean_std_list = [], []
    for pred in range(cluster_num):
        
        train_value_list, test_value_list = [], []
        for i, feature_name in enumerate(target_feature_names):
            feature_column = preprocessed_df[feature_name]
            
            train_cluster_feature_column = feature_column.iloc[train_index][pred_column==pred]
            test_cluster_feature_column = feature_column.iloc[test_index][pred_column==pred]
            
            train_mean_value = np.round(train_cluster_feature_column.mean(), decimals=2)
            train_std_value  = np.round(train_cluster_feature_column.std(), decimals=2)
            train_max_value  = np.round(train_cluster_feature_column.max(), decimals=2)
            train_min_value   = np.round(train_cluster_feature_column.min(), decimals=2)
            
            test_mean_value = np.round(test_cluster_feature_column.mean(), decimals=2)
            test_max_value  = np.round(test_cluster_feature_column.max(), decimals=2)
            test_min_value  = np.round(test_cluster_feature_column.min(), decimals=2)
            test_std_value  = np.round(test_cluster_feature_column.std(), decimals=2)
            
            train_value_list.append([train_mean_value, train_std_value, train_max_value, train_min_value])
            test_value_list.append([test_mean_value, test_std_value, test_max_value, test_min_value]) 

    
        train_value_np = np.array(train_value_list)
        test_value_np  = np.array(test_value_list)
        
        train_mean_std_value_column = [
            "{}±{}".format(mean_value, std_value) for mean_value, std_value in zip(train_value_np[:, 0], train_value_np[:, 1])
            ]
        test_mean_std_value_column = [
            "{}±{}".format(mean_value, std_value) for mean_value, std_value in zip(test_value_np[:, 0], test_value_np[:, 1])
            ]
        
        train_mean_std_list.append(train_mean_std_value_column)
        test_mean_std_list.append(test_mean_std_value_column)
        
            
        columns_name = ["mean", "std", "max", "min"]
        train_value_pd = pd.DataFrame(train_value_np, columns=columns_name, index=target_feature_names)
        test_value_pd = pd.DataFrame(test_value_np, columns=columns_name, index=target_feature_names)
        
        train_value_pd.to_excel(os.path.join(save_base_path, "train_value_cluster{}.xlsx".format(pred+1)), index=True)
        test_value_pd.to_excel(os.path.join(save_base_path, "test_value_cluster{}.xlsx".format(pred+1)), index=True)
    
    train_mean_std_np = np.array(train_mean_std_list).T
    test_mean_std_np  = np.array(test_mean_std_list).T
    
    train_mean_std_df = pd.DataFrame(train_mean_std_np, columns=["Cluster1", "Clsuter2", "Cluster3"], index=target_feature_names)
    test_mean_std_df = pd.DataFrame(test_mean_std_np, columns=["Cluster1", "Clsuter2", "Cluster3"], index=target_feature_names)

    train_mean_std_df.to_excel(os.path.join(save_base_path, "train_mean_std.xlsx"), index=True)
    test_mean_std_df.to_excel(os.path.join(save_base_path, "test_mean_std.xlsx"), index=True)
    
    save_feature_names = [
        "S_Unique_Subject_Identifier",
        "TAVIdate", 'time_to_MACE', 'time_to_allcase_death',
       'time_to_CVD_death', 'MACE+allDeath', 'MACE', 'time_to_MACE+death',
       "TAVIyear", 'train_test', 'kmeans_pred'
        ]
    
    save_df = preprocessed_df.copy()
    save_df["PatientNo"]   = ""
    save_df["train_test"]  = ""
    save_df["kmeans_pred"] = ""
    
    for index in preprocessed_df.index:
        preprocessed_features = preprocessed_df.loc[index]
        save_df.loc[index, "PatientNo"]   = preprocessed_features["S_Unique_Subject_Identifier"]
        save_df.loc[index, "train_test"]  = preprocessed_features["train_test"]
        save_df.loc[index, "kmeans_pred"] = preprocessed_features["kmeans_pred"] + 1
        
    save_path = os.path.join(save_base_path, "TAVI_Kmeans_result.xlsx")
    save_df.to_excel(save_path, index=None)
        
