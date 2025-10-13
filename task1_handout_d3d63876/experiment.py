import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import BayesianRidge
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import torch 
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from sklearn.model_selection import train_test_split

# load data
train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)

# parse the first two columns
X = train_x[:, :2].astype(np.float64, copy=False)            # (n, 2)
y = train_y.astype(np.float64, copy=False).reshape(-1, 1)    # (n, 1) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# scale the data
scalerX = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)).fit(X_train)
X_train = scalerX.transform(X_train)
X_test = scalerX.transform(X_test)

scalerY = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)).fit(y_train)
y_train = scalerY.transform(y_train)
y_test = scalerY.transform(y_test)

# run K Means
k = 10
kmeans = KMeans(
        n_clusters=k,
        algorithm="lloyd",     
        init="random",      
        n_init=10,             
        max_iter=300,
        tol=1e-4,
        random_state=0,
        copy_x=False,           
    ).fit(X_train)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

inducing_points_per_cluster = 300
inducing_points_list_X = []
inducing_points_list_y = []
inducing_points_labels_list = []

for i in range(k):
    idx = np.where(labels == i)[0]
    
    n_select = min(inducing_points_per_cluster, len(idx))
    chosen_idx = np.random.choice(idx, size=n_select, replace=False)
    
    inducing_points_list_X.append(X[chosen_idx])
    inducing_points_list_y.append(y[chosen_idx])
    inducing_points_labels_list.append(labels[chosen_idx])

# shape: [k*300, feature_dim]
inducing_points_X = np.vstack(inducing_points_list_X)
inducing_points_y = np.vstack(inducing_points_list_y).ravel()
inducing_points_labels = np.vstack(inducing_points_labels_list)

# --- 3D Scatter Plot --- of K Means
# --- KMeans on X_train (already scaled above) ---
k = 10
kmeans = KMeans(
    n_clusters=k, algorithm="lloyd",
    init="random", n_init=10, max_iter=300,
    tol=1e-4, random_state=0
).fit(X_train)

labels = kmeans.labels_              # length = X_train.shape[0]
centers_xy = kmeans.cluster_centers_ # shape (k, 2)  <-- 2D centers

# --- pick inducing points FROM TRAINING SET (same indexing & scaling) ---
inducing_points_per_cluster = 300
rng = np.random.default_rng(0)

inducing_X, inducing_y, inducing_lbl = [], [], []
for i in range(k):
    idx = np.where(labels == i)[0]
    if idx.size == 0:
        continue
    take = min(inducing_points_per_cluster, idx.size)
    sel = rng.choice(idx, size=take, replace=False)

    inducing_X.append(X_train[sel])               # use TRAIN set!
    inducing_y.append(y_train[sel].ravel())       # use TRAIN set!
    inducing_lbl.append(labels[sel])

inducing_points_X = np.vstack(inducing_X)
inducing_points_y = np.concatenate(inducing_y)
inducing_points_labels = np.concatenate(inducing_lbl)

# --- Centers in 3D: z = mean(y) per cluster ---
center_z = np.array([y_train[labels == i].mean() for i in range(k)])  # (k,)
centers_3d = np.c_[centers_xy, center_z]                              # (k,3)

# --- 3D plot ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(inducing_points_X[:, 0], inducing_points_X[:, 1], inducing_points_y,
                c=inducing_points_labels, cmap='viridis', s=16, alpha=0.85)

ax.scatter(centers_3d[:, 0], centers_3d[:, 1], centers_3d[:, 2],
           c='red', s=220, marker='X', edgecolor='k', label='Cluster centers (z = mean y)')

ax.set_xlabel('Feature 1 (scaled)')
ax.set_ylabel('Feature 2 (scaled)')
ax.set_zlabel('y (scaled)')
ax.set_title(f'KMeans (k={k}) on X_train with y overlay')
plt.colorbar(sc, ax=ax, shrink=0.55, label='Cluster label')
ax.legend(loc='upper left')
ax.view_init(elev=25, azim=45)
plt.tight_layout()
plt.show()
plt.savefig("kmeans_clusters_2d.png", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.clf()      # Clear current figure
plt.cla()      # Clear current axes
plt.close()    # Close current figure
plt.close('all')  # Close all open figures

X_tr = torch.tensor(X_train, dtype=torch.float32)
y_tr = torch.tensor(y_train.ravel(), dtype=torch.float32)
train_ds = TensorDataset(X_tr, y_tr)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

Z = torch.tensor(inducing_points_X, dtype=torch.float32)

d = X.shape[1]
class SVGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
             self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SVGPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        lp  = gpytorch.kernels.ProductKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=d),
            gpytorch.kernels.PeriodicKernel(ard_num_dims=d)
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(lp)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SVGPModel(Z)

model.train()
likelihood.train()

mll_elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_tr.shape[0])

# two-optimizer setup: variational vs hyper+likelihood ---
var_opt = torch.optim.Adam(model.variational_parameters(), lr=0.1)
hyp_opt  = torch.optim.Adam([
    {'params': model.hyperparameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

iters = 200 + 1
for epoch in range(iters):
    for xb, yb in train_loader:
        var_opt.zero_grad()
        hyp_opt.zero_grad()
        output = model(xb)
        loss = -mll_elbo(output, yb)
        loss.backward()
        var_opt.zero_grad()
        hyp_opt.zero_grad()
       
    print("Epoch: ", epoch, "\t Loss:", loss.item())

# --- Evaluation and Visualization ---
model.eval()
likelihood.eval()

# Convert full training data to torch
X_full_t = torch.tensor(X, dtype=torch.float32)

# Predict mean and variance
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred_dist = likelihood(model(X_full_t))
    y_pred_mean = pred_dist.mean.cpu().numpy()
    y_pred_std = pred_dist.variance.sqrt().cpu().numpy()

# Flatten y to match shapes
y_true = y.ravel()

# ---- 2D Scatter: True vs Predicted ----
plt.figure(figsize=(7, 7))
plt.scatter(y_true, y_pred_mean, alpha=0.6, s=10, color='steelblue', label="Predictions")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
         'r--', label='Ideal fit (y = ŷ)')
plt.fill_between(
    np.linspace(y_true.min(), y_true.max(), 100),
    np.linspace(y_true.min(), y_true.max(), 100) - 2*np.mean(y_pred_std),
    np.linspace(y_true.min(), y_true.max(), 100) + 2*np.mean(y_pred_std),
    color='orange', alpha=0.2, label='±2σ region'
)
plt.xlabel("True y (scaled)")
plt.ylabel("Predicted y (GP mean)")
plt.title("GPyTorch SVGP Predictions vs True Targets")
plt.legend()
plt.tight_layout()
plt.savefig("residuals_projected_to_2d.png", dpi=300, bbox_inches='tight')
plt.show()
plt.clf()      # Clear current figure
plt.cla()      # Clear current axes
plt.close()    # Close current figure
plt.close('all')  # Close all open figures

# ---- 3D Visualization: Input features vs Prediction ----
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# True surface (blue) and GP predictions (red)
ax.scatter(X[:, 0], X[:, 1], y_true, c='blue', alpha=0.5, s=15, label='True y')
ax.scatter(X[:, 0], X[:, 1], y_pred_mean, c='red', alpha=0.5, s=15, label='Predicted y')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target / Prediction')
ax.set_title('GPyTorch SVGP: True vs Predicted (3D)')
ax.legend()
ax.view_init(elev=25, azim=45)
plt.tight_layout()
plt.savefig("input_vs_predictions.png", dpi=300, bbox_inches='tight')
plt.show()
plt.clf()      # Clear current figure
plt.cla()      # Clear current axes
plt.close()    # Close current figure
plt.close('all')  # Close all open figures
