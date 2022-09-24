import torch
from torch.distributions import multinomial
from d2l import torch as d2l
import matplotlib.pyplot as plt

fair_probs = torch.ones([6]) / 6
# print(multinomial.Multinomial(1, fair_probs).sample())
# print(multinomial.Multinomial(10, fair_probs).sample())
# Store the results as 32-bit floats for division
counts = multinomial.Multinomial(1000, fair_probs).sample()
# print(counts / 1000)  # Relative frequency as the estimate

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
print(counts)
cum_counts = counts.cumsum(dim=0)
print(cum_counts)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
print(estimates)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()

plt.show()