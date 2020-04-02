"""Compute maximum mean discrepancy statistics between groups
in an AnnData experiment"""
import anndata
import numpy as np
from torch_two_sample import MMDStatistic
import torch
from typing import Union

import os
import os.path as osp
import configargparse
import tqdm


################################################
# group based random sampling
################################################


def random_sample(
    groupbys: list,
    groups: list,
    n: int = 500,
) -> np.ndarray:
    """Generate a random index of cells where cells 
    are drawn from `group` in `groupby`.

    Parameters
    ----------
    groupbys : list[np.ndarray]
        list of [Cells,] np.ndarraay vectors of group 
        assignments for each group to sample from.
    groups : str
        list of str values in groupbys to sample from.
        `len(groupbys) == len(groups)`.
    n : int
        sample size.

    Returns
    -------
    idx : np.ndarray
        [n,] index of integers.
    """
    assert len(groupbys) == len(groups)
    bidxs = []
    for i in range(len(groups)):
        group_bidx = (groupbys[i] == groups[i])
        bidxs.append(group_bidx)
    bidx = np.logical_and.reduce(bidxs)
    group_idx = np.where(bidx)[0].astype(np.int)

    idx = np.random.choice(
        group_idx,
        size=n,
        replace=True
    ).astype(np.int)
    return idx

################################################
# maximum mean discrepancy
################################################


def estimate_kernel_bandwidth(
    X: np.ndarray,
    Y: np.ndarray,
    n_median_estimate: int = None,
) -> float:
    """Estimate a kernel bandwidth as the median pairwise
    distance between samples in the data.

    Parameters
    ----------
    X : np.ndarray, torch.Tensor
        [Samples, Features] from distribution 0 (P).
    Y : np.ndarray, torch.Tensor
        [Samples, Features] from distribution 1 (Q).
    n_median_estimate : int
        number of samples to use from each of `X, Y`
        to estimate the median distance.
        if `None`, uses all samples.

    Returns
    -------
    alpha : float
        estimate of the radial basis kernel parameter
        alpha.

    Notes
    -----
    The original Kernel Two Sample Test paper estimates the radial
    basis function kernel parameter \sigma as the median euclidean
    distance between points in the joint sample.
    The implementation of MMD we use employs a different formulation
    of the RBF and requires a parameter \alpha that is a strict 
    function of \sigma instead.

    .. math::

        k(x, x') = \sum_j^N \exp(\alpha_j ||x_j - x_j'||^2)

    where

    .. math::

        \alpha = \frac{1}{2\sigma^2}

    Credit to:

    https://bit.ly/2MPqMez

    for kernel estimation logic.

    References
    ----------
    A Kernel Two-Sample Test
    Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, 
    Bernhard Schölkopf, Alexander Smola.
    JMLR, 13(Mar):723−773, 2012.
    http://jmlr.csail.mit.edu/papers/v13/gretton12a.html
    """
    from sklearn.metrics.pairwise import euclidean_distances

    # concatenate the two sample matrices for estimation
    # of the kernel bandwidth
    if n_median_estimate is not None:
        xridx = np.random.choice(
            X.shape[0],
            size=min(X.shape[0], n_median_estimate),
            replace=False,
        )
        Xr = X[xridx, :]
        yridx = np.random.choice(
            Y.shape[0],
            size=min(Y.shape[0], n_median_estimate),
            replace=False,
        )
        Yr = Y[yridx, :]
        J = np.concatenate([Xr, Yr], 0)
    else:
        J = np.concatenate([X, Y], 0)

    dm = euclidean_distances(J,)
    upper = dm[np.triu_indices_from(dm, k=1)]

    sigma = np.median(
        upper,
        overwrite_input=True,
    )
    alpha = 1./(2*(sigma**2))
    return alpha


def run_mmd(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    n_permutations: int = 1000,
    use_cuda: bool = False,
) -> (float, float):
    """Run an MMD test on two N-dimensional samples.

    Parameters
    ----------
    X : np.ndarray
        [Samples, Features] drawn from distribution 0, P.
    Y : np.ndarray
        [Samples, Features] drawn from distribution 1, Q.
    n_permutations : int
        number of permutations for statistical testing.
    use_cuda : bool
        perform computation on the GPU.

    Returns
    -------
    mmd_stat : float
        maximum mean discrepancy.
    p_val : float
        p-value from the permutation test.
    """
    if type(X) == np.ndarray:
        X = torch.from_numpy(X).float()
    elif type(X) == torch.Tensor:
        pass
    else:
        raise TypeError()

    if type(Y) == np.ndarray:
        Y = torch.from_numpy(Y).float()
    elif type(Y) == torch.Tensor:
        pass
    else:
        raise TypeError()

    # use the median pairwise distance to estimate
    # the kernel bandwidth
    alpha = estimate_kernel_bandwidth(
        X.cpu().numpy(),
        Y.cpu().numpy(),
    )
    alpha = torch.FloatTensor([alpha])

    if use_cuda:
        # moving tensors to CUDA is a no-op
        # if the tensor is already there
        X = X.cuda()
        Y = Y.cuda()
        alpha = alpha.cuda()

    # instantiate the MMD test
    mmd_test = MMDStatistic(
        X.size(0),
        Y.size(0),
    )
    # perform the MMD computation
    mmd_stat, distances = mmd_test(
        X,
        Y,
        alpha,
        ret_matrix=True,
    )
    # perform the permutation test to
    # estimate a p_value
    if n_permutations is not None:
        p_val = mmd_test.pval(
            distances,
            n_permutations=n_permutations,
        )
    else:
        p_val = None
    return float(mmd_stat), p_val

############################################
# mmd contrast comparisons
############################################


def compute_mmd_contrast(
    adata: anndata.AnnData,
    representation: str,
    groupby: str,
    contrast: str,
    n_iters: int,
    sample_size: int,
    n_permutations: int,
    use_cuda: bool = False,
) -> (np.ndarray, np.ndarray):
    """Compute MMD across a contrast within groups

    Parameters
    ----------
    adata : anndata.AnnData
        [Cells, Genes]
    representation : str
        representation in `adata` to use for comparisons.
        must be "X" or a key in `adata.obsm`.
    groupby : str
        grouping categorical variable in `adata.obs`.
    contrast : str
        binary variable in `adata.obs`.
    n_iters : int
        number of random sampling iterations to perform.
    sample_size : int
        number of cells to use in each sample.
    n_permutations : int
        number of permutations to perform for p-value testing.

    Returns
    -------
    distances : np.ndarray
        [Groups, Iterations, (Contrast, A-A, B-B)]
    p_values : np.ndarray
        [Groups, Iterations, (Contrast, A-A, B-B)]        
    """
    if representation == 'X':
        X = adata.X
    else:
        X = adata.obsm[representation]

    # densify the array if it is sparse
    if type(X) != np.ndarray:
        try:
            print(f'Densifying sparse {representation} array...')
            X = X.toarray()
        except RuntimeError:
            msg = f'{representation} was not `np.ndarray` and has no `.toarray()`.'
            print(msg)

    # generate [Cell,] vectors of relevant variables
    groupby_vals = np.array(adata.obs[groupby])
    contrast_vals = np.array(adata.obs[contrast])

    # get the unique levels of the contrast
    contrast_levels = np.unique(adata.obs[contrast])

    # [Group, Iterations, (C0-C1, C0-C0, C1-C1)]
    distances = np.zeros((len(np.unique(groupby_vals)), n_iters, 3))
    p_values = np.zeros((len(np.unique(groupby_vals)), n_iters, 3))

    for i, group in enumerate(np.unique(groupby_vals)):
        print('Computing distances for %s' % group)

        n_c0 = int(
            np.logical_and(contrast_vals == contrast_levels[0],
                           groupby_vals == group).sum())
        n_c1 = int(
            np.logical_and(contrast_vals == contrast_levels[1],
                           groupby_vals == group).sum())

        # check that we have enough cells on each side of the contrast
        if n_c0 < sample_size:
            msg = f'{n_c0} samples in {group} {contrast_levels[0]},'
            msg += f'is smaller than {sample_size}'
            print(msg)
        if n_c1 < sample_size:
            msg = f'{n_c1} samples in {group} {contrast_levels[1]},'
            msg += f'is smaller than {sample_size}'
            print(msg)

        for j in tqdm.tqdm(range(n_iters), desc='Computing distances'):

            c0_ridx0 = random_sample(
                groupbys=[contrast_vals, groupby_vals],
                groups=[contrast_levels[0], group],
                n=sample_size,
            )
            c0_ridx1 = random_sample(
                groupbys=[contrast_vals, groupby_vals],
                groups=[contrast_levels[0], group],
                n=sample_size,
            )

            c1_ridx0 = random_sample(
                groupbys=[contrast_vals, groupby_vals],
                groups=[contrast_levels[1], group],
                n=sample_size,
            )
            c1_ridx1 = random_sample(
                groupbys=[contrast_vals, groupby_vals],
                groups=[contrast_levels[1], group],
                n=sample_size,
            )

            c0c1_d = run_mmd(
                X=X[c0_ridx0, :],
                Y=X[c1_ridx0],
                n_permutations=n_permutations,
                use_cuda=use_cuda,
            )

            c0c0_d = run_mmd(
                X=X[c0_ridx0, :],
                Y=X[c0_ridx1],
                n_permutations=n_permutations,
                use_cuda=use_cuda,
            )

            c1c1_d = run_mmd(
                X=X[c1_ridx0, :],
                Y=X[c1_ridx1],
                n_permutations=n_permutations,
                use_cuda=use_cuda,
            )

            distances[i, j, :] = (c0c1_d[0], c0c0_d[0], c1c1_d[0])
            p_values[i, j, :] = (c0c1_d[1], c0c0_d[1], c1c1_d[1])

    return distances, p_values


################################################
# main
################################################


def make_parser():
    parser = configargparse.ArgParser(
        description="Compute MMD between groups in an AnnData experiment.",
    )
    parser.add_argument(
        '--config',
        is_config_file=True,
        required=False,
        help='path to a configuration file.',
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='path to an AnnData `.h5ad` object.',
    )
    parser.add_argument(
        '--out_path',
        type=str,
        required=True,
        help='output path for additional outputs.',
    )
    parser.add_argument(
        '--representation',
        type=str,
        default='X_pca',
        help='representation to use for MMD computation.',
    )
    gb_help = 'variable in `.obs`.\n'
    gb_help += 'computes MMD across `contrast` separately in each level of `groupby`.'
    parser.add_argument(
        '--groupby',
        required=True,
        type=str,
        help=gb_help,
    )
    parser.add_argument(
        '--contrast',
        type=str,
        required=True,
        help='binary var in `.obs` for contrast or "distmat" to compute a distance matrix.',
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=100,
        help='size of samples to draw from each population.',
    )
    parser.add_argument(
        '--n_iters',
        type=int,
        default=100,
        help='number of bootstrap iterations to perform.',
    )
    parser.add_argument(
        '--use_cuda',
        action='store_true',
        help='use CUDA if a device is available.',
    )
    parser.add_argument(
        '--n_permutations',
        type=int,
        default=1000,
        help='number of permutations to perform for significance testing.',
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    # load data
    adata = anndata.read_h5ad(args.data)
    print('%d cells and %d genes are present in the input object.' % adata.shape)

    # check that groupby and group are valid
    if args.groupby not in adata.obs.columns:
        msg = f'{args.groupby} is not in `adata.obs`'
        raise ValueError(msg)

    if args.contrast not in adata.obs.columns:
        msg = f'{args.contrast} is not in `adata.obs`.\n'
        raise ValueError(msg)
    else:
        # check that the provided contrast is binary
        n_l = len(np.unique(adata.obs[args.contrast]))
        if n_l != 2:
            msg = f'`adata.obs[{args.contrast}]` has {n_l} values.\n'
            msg += 'must have exactly 2 values.'
            raise ValueError(msg)
        print(
            f'Computing MMD across {args.contrast} in groups {args.groupby}',
        )

    # ensure the desired representation is in `adata`
    if args.representation not in adata.obsm.keys() and args.representation != 'X':
        msg = f'{args.representation} is not in `adata.obsm`.'
        msg += 'must be in `.obsm` or `==X`.'
        raise ValueError(msg)

    # see if cuda is available if desired
    if args.use_cuda and torch.cuda.is_available():
        print('Using CUDA compute device.')
        use_cuda = True
    elif args.use_cuda and not torch.cuda.is_available():
        print('CUDA device not available.')
        print('Using CPU.')
        use_cuda = False
    else:
        use_cuda = False

    # ensure the output directory exists
    os.makedirs(args.out_path, exist_ok=True)

    distances, p_values = compute_mmd_contrast(
        adata=adata,
        representation=args.representation,
        groupby=args.groupby,
        contrast=args.contrast,
        n_iters=args.n_iters,
        n_permutations=args.n_permutations,
        sample_size=args.sample_size,
        use_cuda=use_cuda,
    )
    group_f = args.groupby.replace(' ', '_').replace('/', '-')
    contrast_f = args.contrast.replace(' ', '_').replace('/', '-')
    np.save(
        osp.join(
            args.out_path,
            f'mmd_distances_{group_f}_{contrast_f}.npy',
        ),
        distances,
    )
    np.save(
        osp.join(
            args.out_path,
            f'mmd_p_values_{group_f}_{contrast_f}.npy',
        ),
        p_values,
    )
    np.savetxt(
        osp.join(
            args.out_path,
            f'mmd_group_names_{group_f}_{contrast_f}.csv',
        ),
        np.unique(adata.obs[args.groupby]),
        fmt='%s',
        delimiter=',',
    )

    print('Done.')

    return

#################################
# __main__
#################################


if __name__ == '__main__':
    main()
