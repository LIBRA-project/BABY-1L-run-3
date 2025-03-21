# BABY-1L-run-3

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.INSERT-DOI.svg)](https://doi.org/10.5281/zenodo.INSERT-DOI)

This repository has the data for the run [**BABY-1L-run-3**].

## How to reproduce the results

### In Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/LIBRA-project/BABY-1L-run-3/HEAD)

### Locally

1. Create a conda environment (requires conda):

```
conda env create -f environment.yml
```

2. Run the notebooks with the created environment `baby-1l-run-3`

## Todo list:
- [x] [Link to Zenodo](https://zenodo.org/)
- [x] Change environment name in [`environment.yml`](environment.yml)
- [x] Change environment name in [CI workflows](.github/workflows)
- [x] Modify [binder](https://mybinder.org/) badge by inserting the repo name
- [x] Add general run data to [`data/general.json`](data/general.json)
- [ ] Add LSC data to [`data/tritium_detection`](data/tritium_detection)
- [ ] Add neutron detection data to [`data/neutron_detection`](data/neutron_detection)
- [x] Add OpenMC model to [`analysis/neutron`](analysis/neutron)
- [x] Add Tritium model to [`analysis/tritium`](analysis/tritium)
- [ ] Add the right version tags to [`environment.yml`](environment.yml)
- [ ] Add and update information in the README
- [ ] Add all analysis to [CI workflows](.github/workflows)
- [ ] Make first release on GitHub
- [ ] Update Zenodo badge with new DOI
- [ ] Link Zenodo record (created automatically) to the [LIBRA-project Zenodo community](https://zenodo.org/communities/libra-project/records)