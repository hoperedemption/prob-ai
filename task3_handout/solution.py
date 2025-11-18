"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C, DotProduct, WhiteKernel

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA

NOISE_STD_F = 0.15
NOISE_STD_V = 0.0001

# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        # [our code]

        # Lagrangian / EI hyperparams
        self.xi = 0.01              # exploration parameter for EI
        self.lambda_const = 1.0

        self.X = np.empty((0, 1))
        self.Y_f = np.array([])
        self.Y_v = np.array([])
        self.eps = 1e-6


        # GP for f(x)

        kf = C(0.5, (1e-3, 10.0)) * Matern(length_scale=1.0,
                                    length_scale_bounds=(1e-2, 20.0),
                                    nu=2.5)
        kf = kf + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-12, 1e-4))

        self.gp_f = GaussianProcessRegressor(kernel=kf, alpha=NOISE_STD_F**2,
                                    normalize_y=True, n_restarts_optimizer=5)
        

        # GP for v(x)
        kv = C(np.sqrt(2.0), (1e-3, 10.0)) * (DotProduct() +
                                                Matern(length_scale=1.0,
                                                        length_scale_bounds=(1e-2, 20.0),
                                                        nu=2.5))
        kv = kv + WhiteKernel(noise_level=NOISE_STD_V**2,
                              noise_level_bounds=(1e-12, 1e-2))
        
        self.gp_v = GaussianProcessRegressor(kernel=kv, alpha=NOISE_STD_V**2,
                                             normalize_y=False, n_restarts_optimizer=5)
        
        # [/our code]
        

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        # raise NotImplementedError

        # [our code]

        x_opt = self.optimize_acquisition_function()

        # Ensure returned shape (1, D)
        x_out = np.atleast_2d(x_opt).reshape(1, DOMAIN.shape[0])

        # Clip to domain just in case
        x_out = np.clip(x_out, DOMAIN[0, 0], DOMAIN[0, 1])

        try:
            p_safe_rec = float(self._predict_prob_safe(np.atleast_2d(x_opt)))
            p_unsafe_rec = 1.0 - p_safe_rec
            # if probability of being above threshold (unsafe) > 0.1
            if p_unsafe_rec > 0.1:
                # search for a point with very high P_safe on a dense grid
                x_grid = np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], 2000)[:, None]
                p_safe_grid = self._predict_prob_safe(x_grid)
                # prefer extremely safe points (>= 0.99), else fallback to max p_safe
                safe_idx = np.where(p_safe_grid >= 0.99)[0]
                if safe_idx.size > 0:
                    chosen_idx = safe_idx[0]
                else:
                    chosen_idx = int(np.argmax(p_safe_grid))
                x_alt = x_grid[chosen_idx]
                x_out = np.atleast_2d(x_alt).reshape(1, DOMAIN.shape[0])
        except Exception:
            # if any safety check fails, fall back to recommended x_out
            pass

        return x_out

        # [/our code]


    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        # raise NotImplementedError

         # [our code]
        n = x.shape[0]

        from scipy.stats import norm

        # Predict objective posterior (mu_f, sigma_f)
        if len(self.X) > 0:
            mu_f, sigma_f = self._predict_gp(self.gp_f, x)
        else:
            mu_f = np.zeros(n)
            sigma_f = np.ones(n) * 1e6
        sigma_f = np.maximum(sigma_f, self.eps)

        # Predict constraint posterior (we model v - 4, so add back 4)
        if len(self.X) > 0:
            mu_v_rel, sigma_v = self._predict_gp(self.gp_v, x)
            mu_v = mu_v_rel + 4.0
        else:
            mu_v = np.ones(n) * 4.0
            sigma_v = np.ones(n) * 1e6
        sigma_v = np.maximum(sigma_v, self.eps)

        # Determine current best using observed safe points
        if len(self.Y_f) == 0:
            f_best = 0.0
        else:
            safe_mask_obs = (self.Y_v < SAFETY_THRESHOLD)
            if np.any(safe_mask_obs):
                f_best = np.max(self.Y_f[safe_mask_obs])
            else:
                f_best = np.max(self.Y_f)
        # Expected Improvement (EI) analytic formula
        # EI = sigma * (gamma * Phi(gamma) + phi(gamma)) where
        # gamma = (mu_f - f_best - xi) / sigma_f
        gamma = (mu_f - f_best - self.xi) / sigma_f
        Phi = norm.cdf(gamma)
        phi = norm.pdf(gamma)
        EI = sigma_f * (gamma * Phi + phi)
        EI = np.maximum(EI, 0.0)
        # Lagrangian penalty: use predicted mean of v
        penalty = self.lambda_const * np.maximum(mu_v - SAFETY_THRESHOLD, 0.0)

        # probability of safety
        p_safe = self._predict_prob_safe(x)

        # Final acquisition: choose mode
        # weigh EI by probability of being safe
        acq = EI * p_safe - penalty


        # If single input return scalar
        if acq.shape[0] == 1:
            return float(acq[0])
        return acq
        # [/our code]


    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float or np.ndarray
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        # raise NotImplementedError


        # [our code]
        # ensure shapes
        x = np.atleast_2d(x)
        if x.shape[1] != DOMAIN.shape[0]:
            x = x.reshape(1, DOMAIN.shape[0])

        self.X = np.vstack([self.X, x])
        self.Y_f = np.append(self.Y_f, float(f))
        self.Y_v = np.append(self.Y_v, float(v))

        # Fit GPs only after we have at least 2 observations to avoid unstable fits
        try:
            if len(self.Y_f) >= 2:
                self.gp_f.fit(self.X, self.Y_f)
        except Exception as e:
            # keep going but print debug info
            print('gp_f.fit failed:', e)

        try:
            if len(self.Y_v) >= 2:
                # Center v observations at 4 so gp_v has prior mean 0 for (v-4)
                y_v_centered = (self.Y_v - 4.0)
                self.gp_v.fit(self.X, y_v_centered)
        except Exception as e:
            print('gp_v.fit failed:', e)


        # [our code]

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        # raise NotImplementedError


        # [our code]
        # --- CHANGED (our code) START
        # If we have any observed safe point, return the best among them
        if len(self.X) > 0:
            safe_mask_obs = (self.Y_v < SAFETY_THRESHOLD)
            if np.any(safe_mask_obs):
                idx = np.argmax(self.Y_f[safe_mask_obs])
                safe_indices = np.where(safe_mask_obs)[0]
                chosen_idx = safe_indices[idx]
                return np.atleast_2d(self.X[chosen_idx]).reshape(1, DOMAIN.shape[0])

        # Otherwise, use penalized posterior mean on a dense grid to pick solution
        x_grid = np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], 2000)[:, None]
        if len(self.X) > 0:
            mu_f_grid, _ = self._predict_gp(self.gp_f, x_grid)
            mu_v_rel_grid, sigma_v_grid = self._predict_gp(self.gp_v, x_grid)
            mu_v_grid = mu_v_rel_grid + 4.0
        else:
            mu_f_grid = np.zeros(x_grid.shape[0])
            mu_v_grid = np.ones(x_grid.shape[0]) * 4.0
            sigma_v_grid = np.ones(x_grid.shape[0]) * 1e6

        # penalized objective: mu_f - lambda * max(mu_v - kappa, 0)
        penalized = mu_f_grid - self.lambda_const * np.maximum(mu_v_grid - SAFETY_THRESHOLD, 0.0)

        # prefer points that are likely safe (p_safe >= 0.9) if any exist
        from scipy.stats import norm
        z = (SAFETY_THRESHOLD - mu_v_grid) / np.maximum(sigma_v_grid, self.eps)
        p_safe_grid = norm.cdf(z)
        safe_idx = np.where(p_safe_grid >= 0.9)[0]
        if safe_idx.size > 0:
            best_idx = safe_idx[np.argmax(penalized[safe_idx])]
        else:
            # fallback: just pick best penalized
            best_idx = int(np.argmax(penalized))

        x_best = x_grid[best_idx]
        return np.atleast_2d(x_best).reshape(1, DOMAIN.shape[0])
    
        # [/our code]

        

    # ----- our helper methods -----
    def _predict_gp(self, gp, x):
        """Utility to return gp.predict mean and std (std not variance) robustly."""
        x = np.atleast_2d(x)
        if len(self.X) == 0:
            return np.zeros(x.shape[0]), np.ones(x.shape[0]) * 1e6
        mu, sigma = gp.predict(x, return_std=True)
        mu = np.reshape(mu, -1)
        sigma = np.reshape(sigma, -1)
        sigma = np.maximum(sigma, self.eps)
        return mu, sigma

    def _predict_prob_safe(self, x):
        """Return P(v(x) < SAFETY_THRESHOLD) for array x (n,1)."""
        from scipy.stats import norm
        mu_v_rel, sigma_v = self._predict_gp(self.gp_v, x) if len(self.X) > 0 else (np.zeros(x.shape[0]), np.ones(x.shape[0]) * 1e6)
        mu_v = mu_v_rel + 4.0
        z = (SAFETY_THRESHOLD - mu_v) / np.maximum(sigma_v, self.eps)
        return norm.cdf(z)

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
