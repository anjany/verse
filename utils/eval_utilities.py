"""eval_utilities.py: Everything evaluation-related for VerSe."""

__author__      = "Anjany Sekuboyina"



import numpy as np

# -------
# dice
# -------

def compute_dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum



# -------
# id.rate
# -------

def construct_distance_matrix(act, pred, max_vert_idx):
    # computing the distance matrix. Should be a N*N matrix -- distance
    # from every vertebrae to every other vetabrae
    act_stack = np.transpose(np.repeat(np.expand_dims(act, -1), max_vert_idx, axis=2), [2, 1, 0])
    pred_stack = np.repeat(np.expand_dims(pred, -1), max_vert_idx, axis=2)
    d_mat = np.sqrt(np.sum(np.square(pred_stack - act_stack), axis=1))
    return d_mat


def get_hits(cent_list_gt, cent_list_pred, max_vert_idx):

    """
    Computes vertebrae identification stats
    Parameters
    ----------
    cent_list_gt : array-like, float
        Any array of shape (max_vert_idx, 3), each row containing the 3D coordinates of the vertebrae
        NaN if a vertebrae is absent. 
    cent_list_pred : array-like, float
        Predicted centroids. Same as cent_list_gt
    max_vert_idx : int
        Maximum vertebral index that the code is dealing with.

    Returns
    -------
    hits : int
        Count of the number of successful identifcations
        Successful identification defined as the correct label
        being closest and < 20mm away.
    hit_list: array-like
        A 1-d array of lenght max_vert_idx. Contains nan (if vertebra is absent in GT),
        1 (if vertebra is succesfully identified), and 0 (if failed to identify)

    Notes
    -----
    Return hits = 0 if no successful identification.
    """

    hit_list = np.full(max_vert_idx, np.nan)

    # vertebrae present according to json labels
    verts_in_im = np.argwhere(~np.isnan(cent_list_gt[:, 0])) + 1
    verts_in_pred = np.argwhere(~np.isnan(cent_list_pred[:, 0])) + 1

    hit_list[verts_in_im - 1] = 0

    # intersection of vertebrae in actual and predicted jsons
    intersect_verts = np.intersect1d(verts_in_im, verts_in_pred)

    if intersect_verts.size != 0:
        # construct distance_matrix
        d_mat = construct_distance_matrix(cent_list_gt, cent_list_pred, max_vert_idx)
        d_mat_verts = d_mat[intersect_verts - 1, :][:, intersect_verts - 1]

        # ID rates
        # 1. check if pred landmark is closest to actual landmark
        mask = np.ones_like(d_mat_verts, dtype=bool)
        mask[range(mask.shape[0]), np.argmin(d_mat_verts, axis=1)] = False
        d_mat_verts[mask] = np.nan

        # 2. after check above, check if the predicted landmark is less
        # than 20 mm away.
        d_id_verts = np.copy(np.diagonal(d_mat_verts))
        d_id_verts[d_id_verts > 20.] = np.nan

        hits = np.count_nonzero(~np.isnan(d_id_verts))
        hit_list[intersect_verts[~np.isnan(d_id_verts)] - 1] = 1
        return hits, hit_list
    else:
        return 0., hit_list
