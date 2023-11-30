[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# GeidiPrime
This is an extremely experimental Transformer architecture with Macaron like FFNs with local attention. Perhap's we can add the visual expert from Zeta and make it multi-modal!


# Install

## Usage
```python
import torch
from geidi_prime.model import GeidiPrimeTransformer

model = GeidiPrimeTransformer(
    dim=4096,
    depth=6,
    heads=8,
    num_tokens=20000,
)

x = torch.randint(0, 20000, (1, 4096))

out = model(x)
print(out.shape)

```



# License
MIT



