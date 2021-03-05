from pathlib import Path
from numpy.core.numeric import NaN
import numpy as np
import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Circle
import json


v_dict = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
    21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum',
    27: 'Cocc', 28: 'T13'
}

colors_itk = (1/255)*np.array([
    [255,  0,  0], [  0,255,  0], [  0,  0,255], [255,255,  0], [  0,255,255],
    [255,  0,255], [255,239,213],  # Label 1-7 (C1-7)
    [  0,  0,205], [205,133, 63], [210,180,140], [102,205,170], [  0,  0,128],
    [  0,139,139], [ 46,139, 87], [255,228,225], [106, 90,205], [221,160,221],
    [233,150,122], [165, 42, 42],  # Label 8-19 (T1-12)
    [255,250,250], [147,112,219], [218,112,214], [ 75,  0,130], [255,182,193],
    [ 60,179,113], [255,235,205],  # Label 20-26 (L1-6, sacrum)
    [255,235,205], [255,228,196],  # Label 27 cocc, 28 T13,
    [218,165, 32], [  0,128,128], [188,143,143], [255,105,180],  
    [255,  0,  0], [  0,255,  0], [  0,  0,255], [255,255,  0], [  0,255,255],
    [255,  0,255], [255,239,213],  # 29-39 unused
    [  0,  0,205], [205,133, 63], [210,180,140], [102,205,170], [  0,  0,128],
    [  0,139,139], [ 46,139, 87], [255,228,225], [106, 90,205], [221,160,221],
    [233,150,122],   # Label 40-50 (subregions)
    [255,250,250], [147,112,219], [218,112,214], [ 75,  0,130], [255,182,193],
    [ 60,179,113], [255,235,205], [255,105,180], [165, 42, 42], [188,143,143],
    [255,235,205], [255,228,196], [218,165, 32], [  0,128,128] # rest unused     
    ])
cm_itk = ListedColormap(colors_itk)
cm_itk.set_bad(color='w', alpha=0)  # set NaN to full opacity for overlay
#plt.rcParams['axes.facecolor'] = 'black'  # set axis background black
# define HU windows
wdw_sbone = Normalize(vmin=-500, vmax=1300, clip=True)
wdw_hbone = Normalize(vmin=-200, vmax=1000, clip=True)

#########################
# Resample and reorient #


def reorient_to(img, axcodes_to=('P', 'I', 'R'), verb=False):
    # Note: nibabel axes codes describe the direction not origin of axes
    # direction PIR+ = origin ASL
    aff = img.affine
    arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    ornt_fr = nio.io_orientation(aff)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)
    if verb:
        print("[*] Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return newimg


def resample_nib(img, voxel_spacing=(1, 1, 1), order=3):
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
        ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img


def resample_mask_to(msk, to_img):
    to_img.header['bitpix'] = 8
    to_img.header['datatype'] = 2  # uint8
    new_msk = nib.processing.resample_from_to(msk, to_img, order=0)
    print("[*] Mask resampled to image size:", new_msk.header.get_data_shape())
    return new_msk


def get_plane(img_path):
    plane_dict = {
        'S': 'ax', 'I': 'ax', 'L': 'sag', 'R': 'sag', 'A': 'cor', 'P': 'cor'}
    img = nib.load(str(img_path))
    axc = np.array(nio.aff2axcodes(img.affine))
    zms = np.around(img.header.get_zooms(), 1)
    ix_max = np.array(zms == np.amax(zms))
    num_max = np.count_nonzero(ix_max)
    if num_max == 2:
        plane = plane_dict[axc[~ix_max][0]]
    elif num_max == 1:
        plane = plane_dict[axc[ix_max][0]]
    else:
        plane = 'iso'
    return plane


######################
# Handling centroids #

def load_centroids(ctd_path):
    with open(ctd_path) as json_data:
        dict_list = json.load(json_data)
        json_data.close()
    ctd_list = []
    for d in dict_list:
        if 'direction' in d:
            ctd_list.append(tuple(d['direction']))
        elif 'nan' in str(d):            #skipping NaN centroids
            continue
        else:
            ctd_list.append([d['label'], d['X'], d['Y'], d['Z']]) 
    return ctd_list


def centroids_to_dict(ctd_list):
    dict_list = []
    for v in ctd_list:
        if any('nan' in str(v_item) for v_item in v): continue   #skipping invalid NaN values
        v_dict = {}
        if isinstance(v, tuple):
            v_dict['direction'] = v
        else:
            v_dict['label'] = int(v[0])
            v_dict['X'] = v[1]
            v_dict['Y'] = v[2]
            v_dict['Z'] = v[3]
        dict_list.append(v_dict)
    return dict_list


def save_centroids(ctd_list, out_path):
    if len(ctd_list) < 2:
        print("[#] Centroids empty, not saved:", out_path)
        return
    json_object = centroids_to_dict(ctd_list)
    # Problem with python 3 and int64 serialisation.
    def convert(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError
    with open(out_path, 'w') as f:
        json.dump(json_object, f, default=convert)
    print("[*] Centroids saved:", out_path)


def calc_centroids(msk, decimals=1, world=False):
    # Centroids are in voxel coordinates!
    # world=True: centroids are in world coordinates
    msk_data = np.asanyarray(msk.dataobj, dtype=msk.dataobj.dtype)
    axc = nio.aff2axcodes(msk.affine)
    ctd_list = [axc]
    verts = np.unique(msk_data)[1:]
    verts = verts[~np.isnan(verts)]  # remove NaN values
    for i in verts:
        msk_temp = np.zeros(msk_data.shape, dtype=bool)
        msk_temp[msk_data == i] = True
        ctr_mass = center_of_mass(msk_temp)
        if world:
            ctr_mass = msk.affine[:3, :3].dot(ctr_mass) + msk.affine[:3, 3]
            ctr_mass = ctr_mass.tolist()
        ctd_list.append([i] + [round(x, decimals) for x in ctr_mass])
    return ctd_list


def reorient_centroids_to(ctd_list, img, decimals=1, verb=False):
    # reorient centroids to image orientation
    # todo: reorient to given axcodes (careful if img ornt != ctd ornt)
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    if len(ctd_arr) == 0:
        print("[#] No centroids present") 
        return ctd_list
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ornt_fr = nio.axcodes2ornt(ctd_list[0])  # original centroid orientation
    axcodes_to = nio.aff2axcodes(img.affine)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    trans = nio.ornt_transform(ornt_fr, ornt_to).astype(int)
    perm = trans[:, 0].tolist()
    shp = np.asarray(img.dataobj.shape)
    ctd_arr[perm] = ctd_arr.copy()
    for ax in trans:
        if ax[1] == -1:
            size = shp[ax[0]]
            ctd_arr[ax[0]] = np.around(size - ctd_arr[ax[0]], decimals)
    out_list = [axcodes_to]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
    if verb:
        print("[*] Centroids reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return out_list


def rescale_centroids(ctd_list, img, voxel_spacing=(1, 1, 1)):
    # rescale centroid coordinates to new spacing in current x-y-z-orientation
    ornt_img = nio.io_orientation(img.affine)
    ornt_ctd = nio.axcodes2ornt(ctd_list[0])
    if np.array_equal(ornt_img, ornt_ctd):
        zms = img.header.get_zooms()
    else:
        ornt_trans = nio.ornt_transform(ornt_img, ornt_ctd)
        aff_trans = nio.inv_ornt_aff(ornt_trans, img.dataobj.shape)
        new_aff = np.matmul(img.affine, aff_trans)
        zms = nib.affines.voxel_sizes(new_aff)
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ctd_arr[0] = np.around(ctd_arr[0] * zms[0] / voxel_spacing[0], decimals=1)
    ctd_arr[1] = np.around(ctd_arr[1] * zms[1] / voxel_spacing[1], decimals=1)
    ctd_arr[2] = np.around(ctd_arr[2] * zms[2] / voxel_spacing[2], decimals=1)
    out_list = [ctd_list[0]]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
    print("[*] Rescaled centroid coordinates to spacing (x, y, z) =", voxel_spacing, "mm")
    return out_list

def create_figure(dpi, *planes):
    fig_h = round(2 * planes[0].shape[0] / dpi, 2)
    plane_w = [p.shape[1] for p in planes]
    w = sum(plane_w)
    fig_w = round(2 * w / dpi, 2)
    x_pos = [0]
    for x in plane_w[:-1]:
        x_pos.append(x_pos[-1] + x)
    fig, axs = plt.subplots(1, len(planes), figsize=(fig_w, fig_h))
    for a in axs:
        a.axis('off')
        idx = axs.tolist().index(a)
        a.set_position([x_pos[idx]/w, 0, plane_w[idx]/w, 1])
    return fig, axs


def plot_sag_centroids(axs, ctd, zms):
    # requires v_dict = dictionary of mask labels
    for v in ctd[1:]:
        axs.add_patch(Circle((v[2]*zms[1], v[1]*zms[0]), 2, color=colors_itk[v[0]-1]))
        axs.text(4, v[1]*zms[0], v_dict[v[0]], fontdict={'color': cm_itk(v[0]-1), 'weight': 'bold'})


def plot_cor_centroids(axs, ctd, zms):
    # requires v_dict = dictionary of mask labels
    for v in ctd[1:]:
        axs.add_patch(Circle((v[3]*zms[2], v[1]*zms[0]), 2, color=colors_itk[v[0]-1]))
        axs.text(4, v[1]*zms[0], v_dict[v[0]], fontdict={'color': cm_itk(v[0]-1), 'weight': 'bold'})