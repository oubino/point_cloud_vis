# imports 
import polars as pl
import numpy as np
import os
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import time
import smlm_cloud
from dotenv import load_dotenv, find_dotenv

# Load in environment file
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

#csv_path = os.environ.get('WIND_PATH')
csv_path = os.environ.get('LINUX_PATH')

# python method for extracting the features
def load_csv(csv, x_name, y_name, z_name, channel_name):

    # Read in csv
    df = pl.read_csv(csv, columns=[x_name, y_name, z_name, channel_name])

    return df, df[channel_name].unique()

def cov_to_feats_linear(covs):

    np_list = []
    for X in covs:
        mean = np.mean(X, axis =1)
        mean = np.expand_dims(mean, axis=1)
        X = X - mean

        # covariance matrix
        C = np.matmul(X, X.T)/(30)

        # eigenvals eigenvectors of C
        eigs, vecs = np.linalg.eig(C)
        sorted_eig_vecs = sorted(zip(eigs,vecs), key=lambda pair:pair[0])

        # lambdas sorted from biggest to smallest
        lam_1 = sorted_eig_vecs[2][0]
        lam_2 = sorted_eig_vecs[1][0]
        lam_3 = sorted_eig_vecs[0][0]

        # linearity
        linearity = (lam_1**0.5 - lam_2**0.5)/lam_1**0.5

        # planarity
        planarity = (lam_2**0.5 - lam_3**0.5)/lam_1**0.5

        # scattering
        scattering = lam_3**0.5/lam_1**0.5

        if linearity > planarity:
            np_list.append(1)
        else:
            np_list.append(0)

    return np.array(np_list, dtype='int32')

def df_to_feats(pcd, csv_path, x_col_name, y_col_name, z_col_name, nn):

    output = smlm_cloud.extract_features(csv_path, 'X (nm)', 'Y (nm)', 'Z (nm)', 1000)
    output = np.array(output)
    labels = np.array([1 if b[0] > 0.3 else 0 for b in output], dtype='int32')

    pcd.estimate_covariances(k=30)
    covs = np.asarray(pcd.covariances)
    print(covs)

    print('linears', np.count_nonzero(labels==1))
    print('non linears', np.count_nonzero(labels==0))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcd

def visualise():
    df, unique_chans = load_csv(csv_path, 'X (nm)', 'Y (nm)', 'Z (nm)', 'Channel')

    pcds = []

    def add_pcd(df, chan, x_name, y_name, z_name, chan_name, unique_chans, cmap, pcds):
        if chan in unique_chans:
            pcd = o3d.geometry.PointCloud()
            coords = df.filter(pl.col(chan_name) == chan).select([x_name, y_name, z_name]).to_numpy()
            pcd.points = o3d.utility.Vector3dVector(coords)
            pcd.paint_uniform_color(cl.to_rgb(cmap[chan]))
            pcds.append(pcd)

    cmap = ['r', 'darkorange', 'b', 'y']

    add_pcd(df, 0, 'X (nm)', 'Y (nm)', 'Z (nm)', 'Channel', unique_chans, cmap, pcds)
    add_pcd(df, 1, 'X (nm)', 'Y (nm)', 'Z (nm)', 'Channel', unique_chans, cmap, pcds)
    add_pcd(df, 2, 'X (nm)', 'Y (nm)', 'Z (nm)','Channel', unique_chans, cmap, pcds)
    add_pcd(df, 3, 'X (nm)', 'Y (nm)', 'Z (nm)','Channel', unique_chans, cmap, pcds)

    assert(len(pcds) == len(unique_chans))

    #pcd = df_to_feats(pcd, csv_path, 'X (nm)', 'Y (nm)', 'Z (nm)', 1000)

    class present:
        def __init__(self):
            self.chan_zero_present = True
            self.chan_one_present = True
            self.chan_two_present = True
            self.chan_three_present = True

    present = present()

    def visualise_chan_0(vis):
        if present.chan_zero_present:
            vis.remove_geometry(pcds[3], False)
            present.chan_zero_present = False
        else:
            vis.add_geometry(pcds[3], False)
            present.chan_zero_present = True

    def visualise_chan_1(vis):
        if present.chan_one_present:
            vis.remove_geometry(pcds[2], False)
            present.chan_one_present = False
        else:
            vis.add_geometry(pcds[2], False)
            present.chan_one_present = True

    def visualise_chan_2(vis):
        if present.chan_two_present:
            vis.remove_geometry(pcds[1], False)
            present.chan_two_present = False
        else:
            vis.add_geometry(pcds[1], False)
            present.chan_two_present = True

    def visualise_chan_3(vis):
        if present.chan_three_present:
            vis.remove_geometry(pcds[0], False)
            present.chan_three_present = False
        else:
            vis.add_geometry(pcds[0], False)
            present.chan_three_present = True

    # reverse pcds for visualisation
    pcds.reverse()

    key_to_callback = {}
    key_to_callback[ord("K")] = visualise_chan_0
    key_to_callback[ord("R")] = visualise_chan_1
    key_to_callback[ord("T")] = visualise_chan_2
    key_to_callback[ord("Y")] = visualise_chan_3
    o3d.visualization.draw_geometries_with_key_callbacks(pcds, key_to_callback)

if __name__ == "__main__":
    visualise()