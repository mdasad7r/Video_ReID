# utils/metrics.py

import numpy as np
import torch

def compute_distance_matrix(query_features, gallery_features):
    """Compute distance matrix between query and gallery features.
    
    Args:
        query_features (torch.Tensor): Query features with shape (num_query, feature_dim)
        gallery_features (torch.Tensor): Gallery features with shape (num_gallery, feature_dim)
    
    Returns:
        torch.Tensor: Distance matrix with shape (num_query, num_gallery)
    """
    m, n = query_features.size(0), gallery_features.size(0)
    distmat = torch.pow(query_features, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gallery_features, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(query_features, gallery_features.t(), beta=1, alpha=-2)
    return distmat.cpu().numpy()

def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with Market1501 metrics.
    
    Args:
        distmat (numpy.ndarray): Distance matrix
        q_pids (numpy.ndarray): Query person IDs
        g_pids (numpy.ndarray): Gallery person IDs
        q_camids (numpy.ndarray): Query camera IDs
        g_camids (numpy.ndarray): Gallery camera IDs
        max_rank (int): Maximum rank to compute CMC
    
    Returns:
        tuple: CMC and mAP scores
    """
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
        print(f"Note: number of gallery samples is quite small, got {num_g}")
    
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # Compute CMC
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    
    for q_idx in range(num_q):
        # Get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        
        # Remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        
        # Compute CMC
        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            continue
        
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        
        # Compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1.
    
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    
    return all_cmc, mAP

def evaluate_rerank(distmat, q_pids, g_pids, q_camids, g_camids, k1=20, k2=6, lambda_value=0.3):
    """Re-ranking with k-reciprocal encoding."""
    # The original distance
    original_dist = distmat.copy()
    all_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    def k_reciprocal_neigh(initial_rank, i, k1):
        forward_k_neigh_index = initial_rank[i, :k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1+1]
        fi = np.where(backward_k_neigh_index == i)[0]
        return forward_k_neigh_index[fi]

    # k-reciprocal neighbors
    for i in range(all_num):
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight/np.sum(weight)

    # Compute final distance
    original_dist = original_dist.T
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
    for i in range(all_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2-temp_min)

    final_dist = jaccard_dist * (1-lambda_value) + original_dist * lambda_value
    return final_dist
    return final_dist
