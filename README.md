# elicit-disk-copulas
Decoding the age-chemical structure of the Milky Way disk:
An application of Copulas and Elicitable Maps

# Code Usage
The [``copula``](https://github.com/aaryapatil/elicit-disk-copulas/blob/main/elicit-disk-copulas/copula.py) and [``conditional_correlation``](https://github.com/aaryapatil/elicit-disk-copulas/blob/main/elicit-disk-copulas/conditional_correlation.py) modules provide a user-friendly way to apply copulas and elicitable maps. The following example shows its usage:
```
import numpy as np
from copula import copula
from conditional_correlation import conditional_correlation

# Generate random two-dimensional data
data = [np.random.rand(1000), np.random.rand(1000)]

# Create copula object
model = copula(norm_data)

# Compute copula space data (uniform margins)
u1 = model.F[0](data[0])
u2 =  model.F[1](data[1])

# Generate a variable upon with your data is conditional
z = [np.random.rand(1000)]
# conditional data (x, y | z)
x, y = data[0], data[1]
# Compute conditional correlation rho(x, y | z) using elicitable maps
cm = conditional_correlation([x, y, z])
cm.Estimate(n_iter=1_000, n_print=200)
# Sample continuous function rho(x, y | z)
cm.rho(np.random.rand(100))
```

# Atrributiom
Aarya A. Patil, Jo Bovy, Sebastian Jaimungal, Neige Frankel & Henry W. Leung (2023), submitted to MNRAS