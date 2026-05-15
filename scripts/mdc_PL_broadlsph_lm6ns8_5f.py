import os
import discovery as ds
import jax
import jax.numpy as jnp
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import glob, time
import sys
sys.path.append('../')
from glgp_specspat import makeglobalgp_fourier_specspat
import discovery.anis_coefficients as dac
from numpyro import infer, distributions as dist
import numpyro
import functools
import inspect
import jax_healpy as jhp
import discovery.samplers.numpyro as ds_numpyro

import pickle as pkl

import scipy.stats as stats
print(jax.default_backend())
print(jax.devices())

# sampling items
from numpyro import infer, distributions as dist
import numpyro
import scipy.stats as stats
import healpy as hp
from corner import corner
#import maps
import pandas as pd
import pyarrow.feather as feather
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str, default='/home/schultls/levi_schult/play_disco/notebooks/aniso_dev/chains/ng15/', help="output directory for chains")
args = parser.parse_args()



# LSS get psrs
psrs = pkl.load(open('/home/schultls/levi_schult/play_disco/notebooks/aniso_dev/datasets/mdc1/mdc1_psrs.pkl', 'rb')) 

# LSS model set up
T = ds.getspan(psrs)
comps = 5 # freq components
lmx = 6
nsi=8
nclms = (lmx+1)**2-1

psrpos = np.array([p.pos for p in psrs])

lsphorf = dac.get_linspharm_orf(psrpos, lmax=lmx)

ds.prior.priordict_standard.update({f'gw_clm\\(([0-9]*)\\)': [-5, 5]})
#ds.prior.priordict_standard.update({f'gw_clm_f[0-9]+\\(([0-9]*)\\)': [-5, 5]})

glgp = makeglobalgp_fourier_specspat(psrs, [ds.powerlaw],
                                    [[functools.partial(lsphorf, c00=np.sqrt(4*np.pi))]],
                                    components=comps, T=T, name='gw', nbasis=[[nclms]])


# m = ds.ArrayLikelihood([ds.PulsarLikelihood([psr.residuals,
#                                              ds.makenoise_measurement(psr, psr.noisedict),
#                                              ds.makegp_timing(psr, svd=True)]) for psr in psrs],
#                        commongp = ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T, name='rednoise'),
#                        globalgp = glgp)

m = ds.GlobalLikelihood([ds.PulsarLikelihood([psr.residuals, #ds.makenoise_measurement_simple(psr, noisedict=ndict),
                                             ds.makenoise_measurement_simple(psr),
                                             ds.makegp_timing(psr, svd=True),
                                             ds.makegp_fourier(psr, ds.powerlaw, components=30, name='rednoise')]) for psr in psrs],
                                             globalgp = glgp)

jlogl = jax.jit(m.logL)


# LSS creating nice starting point for sampler
initpars = {f'{p.name}_rednoise_gamma':2.0 for p in psrs}
initpars.update({f'{p.name}_rednoise_log10_A':-15.0 for p in psrs})
initpars.update({f'{p.name}_log10_t2equad':-14.0 for p in psrs})
initpars.update({f'{p.name}_efac':1.0 for p in psrs})
initpars['gw_log10_A'] = -15.0
initpars['gw_gamma'] = 4.33

initpars[f'gw_clm({nclms})'] = np.zeros(nclms)
aligned_init = {p: initpars[p] for p in jlogl.params if p in initpars}
print("isotropic init params:", aligned_init)

print(m.logL(initpars))

print(jlogl(initpars))

t1 = time.time()
for i in range(10):
    jlogl(initpars)
print(f'jitted logl eval time: {(time.time()-t1)/10:.3f} sec')


# LSS build numpyro model
def model():
    gammas = numpyro.sample("gammas", dist.Uniform(0, 1).expand([len(psrs)]))
    log10_As = numpyro.sample("log10_As", dist.Uniform(-20, -19).expand([len(psrs)]))
    clmdraws = numpyro.sample(f"gw_clm({nclms})", dist.Uniform(-5, 5).expand([nclms]))
    gwgam = numpyro.sample('gw_gamma', dist.Uniform(0, 7))
    gwl10A = numpyro.sample('gw_log10_A', dist.Uniform(-20, -11))

    params = {f'{psr.name}_rednoise_gamma': gammas[ii] for ii, psr in enumerate(psrs)}
    params.update({f'{psr.name}_rednoise_log10_A': log10_As[ii] for ii, psr in enumerate(psrs)})
    params.update({f'{p.name}_log10_t2equad':-16.0 for p in psrs})
    params.update({f'{p.name}_efac':1.0 for p in psrs})
    params[f"gw_clm({nclms})"] = clmdraws
    params['gw_gamma'] = gwgam
    params['gw_log10_A'] = gwl10A
    numpyro.deterministic('params', params)
    numpyro.factor('logl', jlogl(params))

nwarmup = 25000
nsamp = 100000
maxtreedepth = 5

mykernel = infer.NUTS(model, max_tree_depth=maxtreedepth, 
                      init_strategy=numpyro.infer.initialization.init_to_value(values=aligned_init))
mcmc = infer.MCMC(mykernel, num_warmup=nwarmup, num_samples=nsamp)

print("Sampler starting...")
randseed = os.environ["SLURM_ARRAY_TASK_ID"] + os.environ["SLURM_JOB_ID"]
print(f'Sampler using random seed: {randseed}')
mcmc.run(jax.random.key(int(randseed)))
mcmc.print_summary()
samples = mcmc.get_samples()

# LSS save chain as a feather
spcpars = [f'gw_clm({nclms})']
chdf = dac.npyrosamples2pddf(samples, spcpars)



# setup directory for chains
if not os.path.isdir(args.outdir):
    try:
        os.mkdir(args.outdir) # sometimes this breaks for no apparent reason
    except:
        pass


# support for chain parallelization using job arrays
if "SLURM_ARRAY_TASK_ID" in list(os.environ.keys()):
    task_id = os.environ["SLURM_ARRAY_TASK_ID"]
    args.outdir += f'/{task_id}'
    if not os.path.isdir(args.outdir):
        try:
            os.mkdir(args.outdir) 
        except:
            pass
print(f'Output to: {args.outdir}')

chdf.to_feather(f'{args.outdir}/mdc_PL_lm6ns8_5comp_mtd{maxtreedepth}_rs{int(randseed)}_{int(nsamp/1000)}ksamp.feather')
print('Mazel Tov')