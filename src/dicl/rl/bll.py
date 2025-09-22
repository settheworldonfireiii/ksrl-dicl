import jax.numpy as jnp
import jax
import equinox as eqx
import optax
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, PRNGKeyArray
import pdb



def givens_rot_choldowndate(L, u, eps=1e-12):
    """
    Perform a rank-1 downdate on lower-triangular L such that:
      A = L L^T  ->  A' = A - u u^T
    Returns the updated lower-triangular factor L' (in JAX style, no in-place writes).
    """
    d = L.shape[0]
    for k in range(d):
        lkk = L[k, k]
        # Downdate: r = sqrt(lkk^2 - u_k^2). Requires SPD remains valid (lkk^2 > u_k^2).
        resid = jnp.sqrt(jnp.maximum(L[k,k]**2 - u[k]**2, eps))
        cos = resid / (L[k,k] + eps)
        sin = u[k] / (L[k,k] + eps)

        L = L.at[k, k].set(resid)
        if k + 1 < d:
            col = L[k+1:, k]
            u_tail = u[k+1:]
            L = L.at[k+1:, k].set((col - sin * u_tail) / cos)
            u = u.at[k+1:].set(cos * u_tail - sin * col)
    return L




class Scaler(eqx.Module):
    def __init__(self, inp_dim):
        self.mu = jnp.zeros(inp_dim)
        self.std = jnp.ones(inp_dim)
        self.cached_mu = jnp.zeros(inp_dim)
        self.cached_std = jnp.ones(inp_dim)
    def fit(self, x):
        self.mu = jnp.mean(x, axis=0)
        self.std = jnp.std(x, axis=0) + 1e-8

    def transform(self, x):
        return (x - self.mu) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mu






class BLL(eqx.Module):
    layers: list
    obs_dim: int
    act_dim: int
    dropout_rate: float
    decay: float
    hidden_dim: float
    optimizer: str
    horizon: int
    noise_var: float
    weights_variance: float
    mean: Array
    cov_matr: Array
    chol_L: Array
    max_logvar: Array
    min_logvar: Array
    training: bool
    




    def __init__(self, model_type, obs_dim, act_dim,  dropout_rate=0.1, decay = 0.01, hidden_dim  = 200, optimizer = "adamw", horizon = 100,std = 1e-3, weights_variance = 1e2, max_logvar = 2, min_logvar = -2, training=True, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 4)
        D = obs_dim if model_type != "cost" else 1 
        self.min_logvar = jnp.full((D,), min_logvar) 
        self.max_logvar = jnp.full((D,), max_logvar)
        if model_type not in ['cost']:
            self.layers = [eqx.nn.Linear(obs_dim + act_dim, hidden_dim, key=keys[0]),
                eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
                eqx.nn.Linear(hidden_dim,2*(obs_dim + act_dim), key=keys[2]),
                eqx.nn.Linear(2*(obs_dim + act_dim), 2*obs_dim, key=keys[3])]
        else:
            self.layers = [eqx.nn.Linear(obs_dim + act_dim, hidden_dim, key=keys[0]),
                eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
                eqx.nn.Linear(hidden_dim,2*(obs_dim + act_dim), key=keys[2]),
                eqx.nn.Linear(2*(obs_dim + act_dim), 2, key=keys[3])]


        self.optimizer = optimizer
        if model_type not in ['cost']:
            self.mean = jnp.zeros((2*(obs_dim + act_dim),obs_dim))
        else:
            self.mean = jnp.zeros((2*(obs_dim + act_dim),1))
        self.noise_var = std
        self.weights_variance = weights_variance
        self.cov_matr = self.weights_variance*jnp.eye(2*(obs_dim + act_dim))
        self.dropout_rate = dropout_rate
        self.training = training
        self.chol_L = jnp.sqrt(self.weights_variance)*jnp.eye(2*(obs_dim + act_dim))
        self.horizon = horizon
        self.decay = decay
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.act_dim = act_dim




    @eqx.filter_jit
    @eqx.filter_vmap(in_axes=(None, 0, 0))
    def __call__(self, x, key):

        # might go till the last layer including activation.
        for i, layer in enumerate(self.layers[:-1]):
            #pdb.set_trace()
            x = layer(x)
            x = self.swish(x)
            if self.training:
                key, subkey = jax.random.split(key)
                #subkey = jax.random.key_data(subkey)
                #x = eqx.nn.Dropout()(x=x, key=subkey)
        x = self.layers[-1](x)
        return x, key

    @eqx.filter_jit
    def predict(self, x, key):
        for j in self.layers[:-1]:
            x = j(x)
            x = self.swish(x)
        w, key = self.sample_weights(self.mean, self.chol_L, key)
        dim = self.layers[-1].bias.shape[0] // 2
        y  = w.T @ x  + self.layers[-1].bias[:dim]
        return y, key

    @eqx.filter_jit
    def features(self, x):
        for j in self.layers[:-1]:
            x = j(x)
            x = self.swish(x)
        return x

    def sample_weights(self, mean, chol_L, key):
        """
        mean: (d,outp.dim) posterior mean vector
        chol_L: (d,d) lower-triangular Cholesky factor of covariance
        key: jax.random.PRNGKey
        """
        d = mean.shape[0]
        n = mean.shape[1]
        dropout_key, key = jax.random.split(key)
        z = jax.random.normal(dropout_key, (d, n))      # z ~ N(0, I)
        return mean + chol_L @ z, key



    def swish(self, x):
        return x * jax.nn.sigmoid(x)

    
    #DEFAULT METHOD:
    @eqx.filter_jit
    def update_bayes_and_chol(self, x, y, mean, bias):
        def step(carry, xy):
            mean, chol_L, bias, noise_var = carry
            x_i, y_i  = xy
            phi = self.features(x_i)
            v = solve_triangular(chol_L, phi, lower = True)
            Sigma_intmd = solve_triangular(chol_L.T, v, lower = False)
            s = noise_var**2 + jnp.dot(phi, Sigma_intmd)
            K = Sigma_intmd / s
            dim = bias.shape[0] // 2
            resid = y_i - bias[:dim] - (mean.T @ phi)
            mean = mean + jnp.outer(K, resid)
            u =  Sigma_intmd / jnp.sqrt(s)
            chol_L = givens_rot_choldowndate(chol_L, u)
            return (mean, chol_L, bias, noise_var), None
        (mean, chol_L, bias, noise_var), _ = jax.lax.scan(step, (mean, self.chol_L, bias, self.noise_var), (x, y))
        return mean, chol_L, noise_var

    @eqx.filter_jit
    def loss_step(self, optimizer, opt_state, inputs, targets, max_logvar, min_logvar, key):
        """
        def loss_fn(model, subkey):
            B = inputs.shape[0]
            subkey = jax.random.split(subkey, B)       # (B, 2)
            preds, key = model(inputs, subkey)
            dim = targets.shape[-1]
            logvar = preds[...,dim:]
            logvar = max_logvar - jax.nn.softplus(max_logvar - logvar)
            logvar = min_logvar + jax.nn.softplus(logvar - min_logvar)
            logvar_loss = 0.5*jnp.mean(jnp.mean( logvar, axis=-1), axis=-1)
            mean_loss = 0.5*jnp.mean(jnp.mean(((preds[...,:dim] - targets) ** 2) * jnp.exp(-logvar), axis=-1), axis=-1)

            return mean_loss + logvar_loss, key[-1]
        """
        def loss_fn(model, subkey):
            B = inputs.shape[0]
            batch_keys = jax.random.split(subkey, B)   # (B, 2)
            preds, _ = model(inputs, batch_keys)

            dim = targets.shape[-1]
            logvar = preds[..., dim:]
            # soft clamp
            logvar = model.max_logvar - jax.nn.softplus(model.max_logvar - logvar)
            logvar = model.min_logvar + jax.nn.softplus(logvar - model.min_logvar)

            # Gaussian NLL
            sqerr = 0.5*jnp.mean(jnp.mean(((preds[..., :dim] - targets) ** 2)*jnp.exp(-logvar), axis = -1), axis = -1)
            varloss = 0.5*jnp.mean(jnp.mean(logvar, axis=-1), axis=-1)
            #jax.debug.print("mse = {}", sqerr)
            loss_value = jnp.mean(sqerr + varloss + jnp.log(2*jnp.pi) )
            return loss_value + 0.01*jnp.sum(model.max_logvar - model.min_logvar), sqerr
        key, subkey = jax.random.split(key)
        #(loss_value, key), grads = eqx.filter_value_and_grad(loss_fn, has_aux = True)(self, subkey)
        (loss_value, mse_loss),grads = eqx.filter_value_and_grad(loss_fn,  has_aux=True)(self, subkey)
        updates, opt_state = optimizer.update(grads, opt_state, self)
        new_model = eqx.apply_updates(self, updates)
        #self.update_bayesian_layer(new_model, key)
        return new_model, opt_state, loss_value, key, mse_loss

    def train(self, inputs, targets, epochs=100, key=None):
        print("training JAX network started")
        if key is None:
            key = jax.random.PRNGKey(0)
        optimizer = getattr(optax, self.optimizer)(1e-3, weight_decay=0.01)
        opt_state = optimizer.init(eqx.filter(self, eqx.is_array))
        max_logvar = self.max_logvar
        min_logvar = self.min_logvar
        for i in range(epochs):
            #key, subkey = jax.random.split(key)
            self, opt_state, loss_value, key, mse_loss = self.loss_step(optimizer, opt_state, inputs, targets, max_logvar, min_logvar, key)
            mean, chol_L, noise_var = self.update_bayes_and_chol(inputs, targets, self.mean, self.layers[-1].bias)
            self = eqx.tree_at(lambda m: m.mean, self, mean)
            self = eqx.tree_at(lambda m: m.chol_L, self, chol_L)
            self = eqx.tree_at(lambda m: m.noise_var, self, noise_var)
            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {mse_loss}")
        return self, key
