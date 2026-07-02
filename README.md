<img src="logo/logo-sans-darkred-border.svg" height="100" width="600" alt="Logo One"> <img src="logo/tandem_logo_final.png" height="100" width=500 alt="Logo Two">

This repository is an attempt to find common links between the long term seismic cycle simulation software [Tandem](https://github.com/TEAR-ERC/tandem/) and the dynamic rupture wave simulation software [SeisSol](https://github.com/SeisSol/SeisSol) and study them to try and couple the software for a fully dynamic simulation.

## Why do this?

- **Quasi-dynamic (QD):** The fault moves slowly, over years to centuries. Slip rates are
  fractions of a millimetre per year. Inertia does not matter. Codes like **tandem** solve
  this regime efficiently because the governing equations are rate-and-state friction
  coupled to a quasi-static elasticity problem (no wave propagation, no time-stepping of
  stress waves through the volume).

- **Fully dynamic (FD):** The fault ruptures in seconds. Slip rates reach metres per
  second. Seismic waves radiate through the volume. Codes like **SeisSol** solve this
  regime using a discontinuous Galerkin method on an unstructured tetrahedral mesh,
  propagating waves explicitly.

A natural workflow is to let tandem simulate the slow inter-seismic loading
over hundreds of years, then hand off the accumulated fault state (how much has slipped,
how fast, what the tractions are, what the friction state variable is) to SeisSol exactly
at the moment rupture is about to begin, and let SeisSol capture the dynamic earthquake.

The bridge converts a tandem fault snapshot at one point in time into a SeisSol
checkpoint file that SeisSol can load as if it had been running all along. The key
challenges are:

1. The two codes represent fault fields differently (different polynomial bases (modal vs nodal), different
   point sets, different coordinate frames).
2. The sign conventions for stress and slip differ between the two codes.
3. The state variable used by rate-and-state friction is expressed in different normalisation conventions.
4. The mesh must be shared, but the way each code indexes its faces differs. Either that, or as an alternative, the two codes must be able to run the same event using different meshes of the same domain.

Currently, there are two ideas to couple the two software:

1. Use the same mesh: Run a simulation in tandem up until the max slip rate reaches a certain threshold (V_th) and then write the snapshot down into a checkpoint. This checkpoint should match the exact checkpoint format that is accepted by SeisSol. This allows the simulation to be started in SeisSol.

2. Use a different mesh: After the snapshot from tandem is obtained, use EASI/ASAGI to interpolate values from one mesh to another. This will require adding new API's with ASAGI/EASI in SeisSol.



## How to use this repository

### 1. Clone the repository

```bash
git clone https://github.com/piyushkarki/tandem-SeisSol-bridge.git
```

### 2. Install the project dependencies from the toml file

```bash
pip install -e .
```

### 3. Run an example conversion from tandem to SeisSol checkpoint

```bash
python -m tandem_to_seissol.bridge --vtu tandem_to_seissol/example/fault_7.pvtu \
                                  --domain-vtu tandem_to_seissol/example/domain_7.pvtu \   
                                  --mesh       tandem_to_seissol/example/seissol_mesh.h5 \ 
                                  --checkpoint tandem_to_seissol/example/seissol_checkpoint.h5 \   
                                  --output     restart-from-tandem.h5 \
                                  --order 4 --ref-normal 0,-1,0 \
                                  --up 0,0,1 \
                                  --coord-scale 1000 \ 
                                  --stress-scale 1e6 \ 
                                  --lambda 3.2038e10 \
                                  --mu 3.2038e10 \
                                  --convert-state
```