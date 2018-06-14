import numpy as np

def square_distance(all_current_values, desired, weights=None):
    """
    all_current_values nx21 where x = number of sequences and 21 amino acid. Each value represents frequency of aa in each column for each sequence
    desired = 1x21 where the desired distribution of amino acids are reported in each column
    
    returns nx1 distance to the desired distribution
    """
    if len(all_current_values.shape) == 1:
        all_current_values = all_current_values.reshape(1, -1)
    if weights is None:
        return (np.power((desired - all_current_values), 2).sum(axis=1))
    else:
        return (weights * (np.power((desired - all_current_values), 2))).sum(axis=1)

def cubed_distance(all_current_values, desired, weights=None):
    """
    all_current_values nx21 where x = number of sequences and 21 amino acid. Each value represents frequency of aa in each column for each sequence
    desired = 1x21 where the desired distribution of amino acids are reported in each column
    
    returns nx1 distance to the desired distribution
    """
    if len(all_current_values.shape) == 1:
        all_current_values = all_current_values.reshape(1, -1)
    cubic_diff = np.power(np.abs(desired - all_current_values), 3)
    
    alpha = np.where(all_current_values <= desired, np.power(desired, -3), np.power((1 - desired), -3))
    if weights is None:
        return (alpha * cubic_diff).sum(axis=1) #(alpha * cubic_diff).sum(axis=1)
    else:
        return (weights * alpha * cubic_diff).sum(axis=1) #(alpha * cubic_diff).sum(axis=1)

def weighted_cubed_distance(all_current_values, desired, weights=None):
    """
    all_current_values nx21 where x = number of sequences and 21 amino acid. Each value represents frequency of aa in each column for each sequence
    desired = 1x21 where the desired distribution of amino acids are reported in each column
    
    returns nx1 distance to the desired distribution
    """
    if len(all_current_values.shape) == 1:
        all_current_values = all_current_values.reshape(1, -1)
    cubic_diff = np.power(np.abs(desired - all_current_values), 3)
    
    if weights is None:
        return (desired * cubic_diff).sum(axis=1) #(alpha * cubic_diff).sum(axis=1)
    else:
        return (weights * desired * cubic_diff).sum(axis=1) #(alpha * cubic_diff).sum(axis=1)

def mle(all_current_values, desired, weights=None):
    "this is a maximmizaiton function!"
    if len(all_current_values.shape) == 1:
        all_current_values = all_current_values.reshape(1, -1)
    return np.prod(np.power((all_current_values/desired), desired) * np.power(((1-all_current_values)/(1-desired)), desired), axis=1)

def cos_bas(all_current_values, desired, weights=None):
    if len(all_current_values.shape) == 1:
        all_current_values = all_current_values.reshape(1, -1)
    if weights is None:
        return (1 - np.cos(np.abs(desired - all_current_values) * np.pi)).sum(axis=1)
    else:
        return (weights * (1 - np.cos(np.abs(desired - all_current_values) * np.pi))).sum(axis=1)

def chi_rel_entropy(all_current_values, desired, weights=None, epsilon=0.005):
    if len(all_current_values.shape) == 1:
        all_current_values = all_current_values.reshape(1, -1)
    log_dist = all_current_values * np.log((all_current_values + epsilon) / (desired + epsilon))
    sq_dist = 0.5 * np.power((desired - all_current_values), 2)
    if weights is None:
        return ((log_dist + sq_dist)).sum(axis=1)        
    else:
        return (weights * (log_dist + sq_dist)).sum(axis=1)

obj_fxn_map = {
    'square_dist': square_distance,
    'cosine': cos_bas,
    'cubic': cubed_distance,
    'weighted_cubic': weighted_cubed_distance,
    'entropy': chi_rel_entropy
}
