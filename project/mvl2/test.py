import jax.numpy as jnp
import genData2
import jax.numpy as jnp
from torch import tensor
from mvtada import MultiVariateTada
from _base import mix_weights, method_moments_estimator_gamma_shape_rate,ProductPoisson

r_p = tensor([[1., 0], [0, 1.]])
r_g = tensor([[1., .5], [.5, 1.]])
v_p = tensor([.05, .05])
h2 = tensor([.9, .9])
popgen_params = genData2.get_popgen_param(h2=h2, v_p=v_p, r_p=r_p, r_g=r_g)
sim_params = {
    "pi": tensor([.1, .1, .05]),
    **popgen_params,
    "RR_mean": tensor([3., 2.]),
    "PV_shape": tensor(1.),
    "PV_mean": tensor(1e-4),
    "PD": tensor([.01, .01]),
    "n_cases": tensor([1.5e4, 1.5e4, 4e3]),
    "n_ctrls": tensor(5e4),
    "fudge_factor": .1
}
sim_params_point_pv = sim_params.copy()
sim_params_point_pv["PV_shape"] = None
sim_data_point_pv = genData2.gen_counts(**sim_params_point_pv)
print(sim_data_point_pv.keys())
for key in sim_data_point_pv.keys():
    print(key,type(sim_data_point_pv[key]))

print(sim_data_point_pv['alt_counts'].shape)
print(sim_data_point_pv['PV'].shape)
print(sim_data_point_pv['PVDs'].shape)
print(sim_data_point_pv['PD_with_both'].shape)
print(sim_data_point_pv['PVD_PD_hats'].shape)
print(sim_data_point_pv['categories'].shape)

print('_____________________')
print(sim_data_point_pv['alt_counts'][:5])
print('_____________________')
print('_____________________')
print('_____________________')
print('_____________________')
print('_____________________')
print('_____________________')
print('_____________________')
print(sim_data_point_pv['PVDs'][:5])
print('_____________________')
print('_____________________')
print('_____________________')
print('_____________________')
print(sim_data_point_pv['PVD_PD_hats'][:5])
print('_____________________')
print('_____________________')
print('_____________________')
print('_____________________')

dist = ProductPoisson(jnp.ones(4))
#print('a',dist.log_prob(jnp.ones(4)))


model = MultiVariateTada()
model.fit(jnp.array(sim_data_point_pv['alt_counts']))


