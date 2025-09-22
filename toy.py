import os
print(os.getcwd())
from bll import *
import time


s =  jnp.ones((1,4))
s1 = jnp.ones((1,3))

print("PRANK ", s)



bll = BLL("dx", 3, 1)
print("initial biases", bll.layers[-1].bias[3:])
time1 = time.time()
bll.train(s, s1, epochs = 1000)
time2 = time.time()

key = jax.random.PRNGKey(42)          # scalar key
# key = jax.random.split(jax.random.PRNGKey
print("TIME ", time2 - time1)
bll.predict(s, key)
