import numpy as np
from scipy.special import digamma

class BNMF:
    """
    Bayesian Non-negative Matrix Factorization - Optimized
    Based on the paper's methodology with proper Bayesian updates
    """
    
    def __init__(self, n_users, n_items, n_factors=10, alpha=0.6, beta=15, n_epochs=25):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        
        # Hyperparameters (paper tested: alpha={0.2,0.4,0.6,0.8}, beta={5,15,25})
        self.alpha = alpha
        self.beta = beta
        
        # Rating scale
        self.min_rating = 1
        self.max_rating = 5
        self.rating_range = self.max_rating - self.min_rating
        
        # Variational parameters
        # Gamma: Dirichlet parameters (n_users x n_factors)
        self.gamma = np.random.gamma(100, 0.01, (n_users, n_factors))
        
        # Epsilon: Beta parameters (n_items x n_factors)
        self.eps_pos = np.random.gamma(100, 0.01, (n_items, n_factors))
        self.eps_neg = np.random.gamma(100, 0.01, (n_items, n_factors))
        
        # Cache for user ratings
        self.user_ratings = {}
        self.item_ratings = {}
    
    def _build_cache(self, train_data):
        """Build rating caches for faster access"""
        self.user_ratings = {i: [] for i in range(self.n_users)}
        self.item_ratings = {i: [] for i in range(self.n_items)}
        
        for _, row in train_data.iterrows():
            uid = int(row['user_id'])
            iid = int(row['item_id'])
            rating = row['rating']
            
            self.user_ratings[uid].append((iid, rating))
            self.item_ratings[iid].append((uid, rating))
    
    def _normalize_rating(self, rating):
        """Normalize rating to [0, 1]"""
        return (rating - self.min_rating) / self.rating_range
    
    def predict(self, user_id, item_id):
        """Predict rating using variational parameters"""
        # Expected theta (user preferences)
        theta = self.gamma[user_id] / np.sum(self.gamma[user_id])
        
        # Expected kappa (item characteristics)
        kappa = self.eps_pos[item_id] / (self.eps_pos[item_id] + self.eps_neg[item_id])
        
        # Predicted normalized rating
        pred_norm = np.dot(theta, kappa)
        
        # Convert back to rating scale
        pred = pred_norm * self.rating_range + self.min_rating
        
        return np.clip(pred, self.min_rating, self.max_rating)
    
    def train(self, train_data):
        """Train using variational Bayesian EM"""
        print("Building rating cache...")
        self._build_cache(train_data)
        
        print(f"Training BNMF with {len(train_data):,} ratings...")
        
        for epoch in range(self.n_epochs):
            # ========================================
            # E-step: Update gamma (user parameters)
            # ========================================
            for u in range(self.n_users):
                if len(self.user_ratings[u]) == 0:
                    continue
                
                for k in range(self.n_factors):
                    self.gamma[u, k] = self.alpha
                    
                    # Sum over all items rated by user u
                    for item_id, rating in self.user_ratings[u]:
                        r_norm = self._normalize_rating(rating)
                        
                        # Compute responsibility (simplified)
                        theta_uk = self.gamma[u, k] / np.sum(self.gamma[u])
                        kappa_ik = self.eps_pos[item_id, k] / (
                            self.eps_pos[item_id, k] + self.eps_neg[item_id, k]
                        )
                        
                        # Weighted contribution
                        contrib = theta_uk * kappa_ik
                        self.gamma[u, k] += r_norm * contrib
            
            # ========================================
            # M-step: Update epsilon (item parameters)
            # ========================================
            for i in range(self.n_items):
                if len(self.item_ratings[i]) == 0:
                    continue
                
                for k in range(self.n_factors):
                    self.eps_pos[i, k] = self.beta
                    self.eps_neg[i, k] = self.beta
                    
                    # Sum over all users who rated item i
                    for user_id, rating in self.item_ratings[i]:
                        r_norm = self._normalize_rating(rating)
                        
                        # Compute responsibility
                        theta_uk = self.gamma[user_id, k] / np.sum(self.gamma[user_id])
                        
                        # Update positive and negative counts
                        self.eps_pos[i, k] += theta_uk * r_norm
                        self.eps_neg[i, k] += theta_uk * (1 - r_norm)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch} completed")
        
        print("BNMF training completed!")