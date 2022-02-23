import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    probas = np.mean(y, axis=0)
    log_probas = np.log(probas + EPS)
    
    return -np.sum(probas * log_probas)
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    probas = np.mean(y, axis=0)   
    
    return 1 - np.sum(probas ** 2)
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    
    return np.mean((y - np.mean(y)) ** 2)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    
    return np.mean(np.abs(y - np.mean(y)))


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, depth):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = 0
        self.left_child = None
        self.right_child = None
        self.depth = depth
        self.isleaf = False
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name
        self.ans = None
        self.prob_ans = None
        self.depth = 0
        
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        mask_1 = X_subset[:, feature_index] < threshold
        mask_2 = X_subset[:, feature_index] >= threshold
        y_left = y_subset[mask_1]
        y_right = y_subset[mask_2]
        
        X_left = X_subset[mask_1]
        X_right = X_subset[mask_2]
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        mask_1 = X_subset[:, feature_index] < threshold
        mask_2 = X_subset[:, feature_index] >= threshold
        y_left = y_subset[mask_1]
        y_right = y_subset[mask_2]
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        
        best_criterion_value = np.inf
        best_feature_index = 0
        best_threshold = 0
        for feature_index in range(X_subset.shape[1]):
            for threshold in np.unique(X_subset[:, feature_index]):
                y_l, y_r = self.make_split_only_y(feature_index, threshold, X_subset, y_subset)
                if len(y_l) == 0 or len(y_r) == 0:
                    continue
                    
                criterion_value = self.criterion(y_l) * y_l.shape[0] + self.criterion(y_r) * y_r.shape[0]
                if criterion_value < best_criterion_value:
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_criterion_value = criterion_value
        
        return best_feature_index, best_threshold


    def make_tree(self, X_subset, y_subset, depth=0):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        if depth > self.depth:
            self.depth = depth
        
        if X_subset.shape[0] <= self.min_samples_split or depth == self.max_depth or len(set(y_subset.flatten())) == 1:
            new_node = Node(0, 0, depth=depth)
            new_node.isleaf = True
            if not self.all_criterions[self.criterion_name][1]:
                new_node.value = np.mean(y_subset)
            new_node.proba = np.mean(y_subset, axis=0)
            return new_node
        
            
        opt_index, opt_threshold = self.choose_best_split(X_subset, y_subset)
        (X_left, y_left), (X_right, y_right) = self.make_split(opt_index, opt_threshold, X_subset, y_subset)
        
        new_node = Node(opt_index, opt_threshold, depth)
        if X_left.shape[0] == 0 or X_right.shape[0] == 0: 
            new_node.isleaf = True
           
            if not self.all_criterions[self.criterion_name][1]:
                new_node.value = np.mean(y_subset)
            
                
        new_node.left_child = self.make_tree(X_left, y_left, depth+1)
        new_node.right_child = self.make_tree(X_right, y_right, depth+1)    
                
        new_node.proba = np.mean(y_subset, axis=0)
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)
            
        self.root = self.make_tree(X, y, 1)
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        y_predicted = []
        
        for i in range(X.shape[0]):
            self.throwing(X[i], self.root)
            y_predicted.append(self.ans)
            
        y_predicted = np.array(y_predicted)
        return y_predicted        
        
        
    def throwing(self, x, node):
        if node.isleaf:
            if self.all_criterions[self.criterion_name][1]:
                self.ans = np.argmax(node.proba)
                self.prob_ans = node.proba
            else:
                self.ans = node.value
            return
        else:
            if x[node.feature_index] < node.value:
                self.throwing(x, node.left_child)
            else:
                self.throwing(x, node.right_child)    
                
                
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        y_predicted_probs = []
        
        for i in range(X.shape[0]):
            self.throwing(X[i], self.root)
            y_predicted_probs.append(self.prob_ans)              
            
        y_predicted_probs = np.array(y_predicted_probs)
        
        return y_predicted_probs
