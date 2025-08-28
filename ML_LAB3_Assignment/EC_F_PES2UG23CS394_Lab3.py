import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i

    Args:
        tensor (torch.Tensor): Input dataset as a tensor, where the last column is the target.

    Returns:
        float: Entropy of the dataset.
    """
    targets = tensor[:, -1]  # get the target column which is the last column
    
    unique_classes, counts = torch.unique(targets, return_counts=True)  # get unique classes and their counts
    counts = counts.float()  # convert counts to float for precise calculation
    total_samples = counts.sum()
    probabilities = counts / total_samples  # calculate probabilities of each class
    
    # Add small epsilon to avoid log(0) which would cause mathematical errors
    eps = 1e-10
    # Calculate entropy using the formula: -Σ(p_i * log2(p_i))
    entropy = -torch.sum(probabilities * torch.log2(probabilities + eps))
    
    return entropy.item()


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Average information of the attribute.
    """
    attr_values = tensor[:, attribute]  # get the values of the specified attribute column
    targets = tensor[:, -1]  # get the target column (last column)
    total_samples = tensor.size(0)  # total number of samples in the dataset
    
    unique_values = torch.unique(attr_values)  # get all unique values in the attribute
    avg_info = 0.0
    
    # Process each unique value in the attribute
    for value in unique_values:
        # Create a mask to find all rows where attribute has this specific value
        mask = (attr_values == value)
        subset_targets = targets[mask]  # get targets for this subset
        subset_size = mask.sum().item()  # count how many samples have this attribute value
        
        if subset_size == 0:
            continue  # skip if no samples have this value
            
        # Calculate entropy for this subset
        unique_classes, counts = torch.unique(subset_targets, return_counts=True)
        counts = counts.float()  # convert to float for precise calculation
        
        if len(counts) == 0:
            subset_entropy = 0.0
        else:
            probabilities = counts / counts.sum()  # calculate probabilities within this subset
            eps = 1e-10
            subset_entropy = -torch.sum(probabilities * torch.log2(probabilities + eps))
            subset_entropy = subset_entropy.item()  # convert to Python float
        
        # Calculate weight (proportion of total samples) and add weighted entropy
        weight = subset_size / total_samples
        avg_info += weight * subset_entropy
    
    return avg_info


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Information gain for the attribute (rounded to 4 decimals).
    """
    # Get the entropy of the entire dataset before splitting
    dataset_entropy = get_entropy_of_dataset(tensor)
    
    # Get the average information after splitting on this attribute
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    
    # Information gain is the reduction in entropy achieved by splitting
    information_gain = dataset_entropy - avg_info
    
    return round(information_gain, 4)  # round to 4 decimal places


def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.

    Returns a tuple with:
    1. Dictionary mapping attribute indices to their information gains
    2. Index of the attribute with highest information gain
    
    Example: ({0: 0.123, 1: 0.768, 2: 1.23}, 2)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.

    Returns:
        tuple: (dict of attribute:index -> information gain, index of best attribute)
    """
    num_features = tensor.shape[1] - 1  # number of features (excluding target column)
    information_gains = {}
    
    # Calculate information gain for each attribute (feature)
    for attr in range(num_features):
        ig = get_information_gain(tensor, attr)
        information_gains[attr] = ig
    
    # Select the attribute with the highest information gain
    selected_attribute = max(information_gains.keys(), key=lambda k: information_gains[k])
    

    return information_gains, selected_attribute
