from the_well.data import WellDataset
import pdequinox as pdeqx
import jax
import jax.numpy as jnp
from tqdm import tqdm

IN_FRAMES = 1
OUT_FRAMES = 1

train_dataset = WellDataset(
    path=f"datasets/permalloy_thin_film_switching",
    well_split_name="train",
    n_steps_input=IN_FRAMES,
    n_steps_output=OUT_FRAMES,
    use_normalization=False,
)


xx = jnp.concat(
    [sample["input_fields"].numpy() for sample in tqdm(train_dataset)], axis=0
)
print(xx.shape)
print(xx.device)
