import numpy as np

class Node:
    def __init__(self, *, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_feats=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_feats = n_feats  # how many features to consider at each split
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else self.n_feats
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        num_labels = len(np.unique(y))

        # stopping conditions
        if (depth >= self.max_depth 
            or n_samples < self.min_samples_split 
            or num_labels == 1):
            leaf_val = self._most_common_label(y)
            return Node(value=leaf_val)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        best_feat, best_thresh, gain = best_split(X, y, feat_idxs)

        if gain == 0:
            # cannot split further
            return Node(value=self._most_common_label(y))

        # split and recurse
        left_mask = X[:, best_feat] <= best_thresh
        left = self._grow_tree(X[left_mask], y[left_mask], depth+1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth+1)
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _most_common_label(self, y):
        counts = np.bincount(y)
        return np.argmax(counts)

    def predict(self, X):
        # apply _traverse for each sample
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2,
                 n_feats=None, subsample_size=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_feats = n_feats
        self.subsample_size = subsample_size
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, self.subsample_size or n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_feats=self.n_feats
            )
            X_samp, y_samp = self._bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        # collect predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # majority vote
        return np.swapaxes(tree_preds, 0, 1).\
                   apply_along_axis(lambda row: np.bincount(row).argmax(), axis=1)
