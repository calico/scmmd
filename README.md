# scMMD -- Comparing cell populations using the maximum mean discrepancy (MMD)

There is a classical problem in statistics known as the **two-sample problem.**
In this setting, you are given discrete observations of two different distributions and asked to determine if the distributions are significantly different.
A special univariate case of this problem is familiar to many biologists as a "difference in means" comparison -- as performed using Student's *t*-test.
The Kolmogorov–Smirnov test is another familiar tool applied to the univariate case.

This problem becomes somewhat more complex in high-dimensions, as differences between distributions may manifest not only in the mean location, but also in the covariance structure and modality of the data.
One approach to comparing distributions in this setting leverages kernel similarity metrics to find the maximum mean discrepancy [(Gretton et. al. 2012, *JMLR*)](http://jmlr.csail.mit.edu/papers/v13/gretton12a.html) -- the largest difference in the means of the distributions under a flexible transformation.
The maximum mean discrepancy (MMD) has seen broad adoption in the field of representation learning due to the ease of computation and robust statistical properties.

Here, we adapt the MMD method to compare cell populations in single cell measurement data.
In the frame of the two-sample problem, each cell population of interest is considered as a distribution and each cell is a single observation from the source distribution.
We use the MMD to compute (1) a metric of the magnitude of difference between two cell populations, and (2) a *p*-value for the significance of this difference.
As an example application, we have used the MMD to compute a "magnitude of aging" by comparing cell populations of the same cell type taken from young and aged animals.

`scMMD` is designed to integrate with the `scanpy` ecosystem and `anndata` structures.

## Citation

We have released `velodyn` in association with a recent pre-print.
Please cite our pre-print if you find `velodyn` useful for your work.


[**Differentiation reveals the plasticity of age-related change in murine muscle progenitors**](https://www.biorxiv.org/content/10.1101/2020.03.05.979112v1)  
Jacob C Kimmel, David G Hendrickson, David R Kelley  
bioRxiv 2020.03.05.979112; doi: https://doi.org/10.1101/2020.03.05.979112

```
@article {Kimmel2020_myodiff,
	author = {Kimmel, Jacob C and Hendrickson, David G and Kelley, David R},
	title = {Differentiation reveals the plasticity of age-related change in murine muscle progenitors},
	elocation-id = {2020.03.05.979112},
	year = {2020},
	doi = {10.1101/2020.03.05.979112},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/03/06/2020.03.05.979112},
	eprint = {https://www.biorxiv.org/content/early/2020/03/06/2020.03.05.979112.full.pdf},
	journal = {bioRxiv}
}
```

If you have any questions or comments, please feel free to email me.

Jacob C. Kimmel, PhD  
[jacobkimmel+velodyn@gmail.com](mailto:jacobkimmel+velodyn@gmail.com)  
Calico Life Sciences, LLC

## Installation

scMMD requires `torch-two-sample`

```bash
git clone https://github.com/josipd/torch-two-sample.git
cd torch-two-sample
pip install .
```

Install scMMD from Github or PyPI

```bash
git clone https://github.com/calico/scmmd
cd scmmd
pip install .
```

OR

```bash
pip install scmmd
```

## Usage

### Command Line Interface

For help, use the built in help menus

```bash
scmmd -h
```

An example command to compare distributions of control and treated cells using a binary contrast variable `treatment`, with comparisons performed within each cell type in `cell_type`.
Hypothetical data are stored in an `anndata` object `anndata_object.h5ad`.

```bash
scmmd \
    --data anndata_object.h5ad \
    --out_path path_for_outputs \
    --representation X_pca \
    --groupby cell_type \
    --contrast treatment \
    --sample_size 300 \
    --n_iter 100 \
    --use_cuda # use a GPU if you have one
```

### Interactive Python API

In addition to a command line interface, we also include an interactive functional API that complies with `scanpy` conventions.

```python
import scmmd
import anndata

# load an anndata object
adata = anndata.read_h5ad(
    'example_dataset.h5ad',
)

distances, p_values = scmmd.compute_mmd_contrast(
    adata=adata, # [Cells, Genes] object
    representation='X_pca', # representation to use, "X" or key in `adata.obsm`.
    groupby='cell_type', # a categorical grouping variable in `adata.obs`
    contrast='age', # a binary contrast in `adata.obs`
    n_iters=100, # number of random sampling iterations
    sample_size=500, # sample size for random samples
    n_permutations=1000, # permutations for p-val calculations
)
```

## Example

We have included a working example to demonstrate how `scMMD` can be used.
Here, we download a pre-prepared AnnData object containing mRNA abundance profiles from stimulated and unstimulated human PBMCs, originally published in [Kang et. al. 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5784859/).
We compute an MMD between stimulated and unstimulated cells of each cell type using the PCA representation.

```bash
cd demo/
./example.sh
```