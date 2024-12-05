import math
num_steps = 10
beta_end = 0.02
def alpha_bar_fn(t):
    return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

betas = []
for i in range(num_steps):
    t1 = i / num_steps
    t2 = (i + 1) / num_steps
    betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), beta_end))

print(betas)