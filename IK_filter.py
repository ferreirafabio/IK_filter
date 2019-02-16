import numpy as np
from vispy import app, visuals, scene, color

PATH_PCN = "./866_rho0.303173_azi345.000000_ele89.226024_theta-0.009812_xcam0.000000_ycam0.000000_zcam0.303173_scale0.188282_xdim0.111353_ydim0.146529_zdim0.03974700000_pcn.npz"
INDEX_FILE_PATH = "./866_rho0.303173_azi345.000000_ele89.226024_theta-0.009812_xcam0.000000_ycam0.000000_zcam0.303173_scale0.188282_xdim0.111353_ydim0.146529_zdim0.03974700000_par_robotiq3f_full.npy"


def get_all_p_n_pairs():
    pcn_file = np.load(PATH_PCN)['pcn']
    index_file = np.load(INDEX_FILE_PATH)

    p_n_pairs = []

    index_file = index_file[:1000000]
    for i in range(index_file.shape[0]):
        idx1, idx2, idx3 = index_file[i, :3].astype(int)
        p1, n1 = pcn_file[idx1][:3], pcn_file[idx1][3:]
        p2, n2 = pcn_file[idx2][:3], pcn_file[idx2][3:]
        p3, n3 = pcn_file[idx3][:3], pcn_file[idx3][3:]
        p_n_pairs.append((idx1, idx2, idx3, p1, p2, p3, n1, n2, n3))

    return p_n_pairs

def highlight_candidate_in_pc(pcn_file, candidate_idx_coords):
    idx1 = candidate_idx_coords[0]
    idx2 = candidate_idx_coords[1]
    idx3 = candidate_idx_coords[2]
    p1 = candidate_idx_coords[3]
    p2 = candidate_idx_coords[4]
    p3 = candidate_idx_coords[5]

    pcn_file = np.delete(pcn_file, idx1, axis=0)
    pcn_file = np.delete(pcn_file, idx2, axis=0)
    pcn_file = np.delete(pcn_file, idx3, axis=0)


    Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
    canvas = scene.SceneCanvas(keys='interactive', show=True)

    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 10
    view.camera.distance = 1

    scatter = Scatter3D(parent=view.scene)

    #scatter.set_gl_state('translucent', blend=True, depth_test=True)
    clr = color.Color(color="grey").rgba
    clr_marker = color.Color(color="yellow").rgba
    colors = np.tile(clr, (pcn_file.shape[0], 1))
    colors_markers = np.tile(clr_marker, (3, 1))
    colors = np.append(colors, colors_markers, axis=0)

    data = np.append(pcn_file[:, :3], np.expand_dims(p1, axis=0), axis=0)
    data = np.append(data, np.expand_dims(p2, axis=0), axis=0)
    data = np.append(data, np.expand_dims(p3, axis=0), axis=0)
    sizes = np.append(np.tile(10, 2045), np.tile(20, 3))
    scatter.set_data(data, size=sizes, face_color=colors)

    #scatter.set_data(np.expand_dims(p1, axis=0), size=5, face_color=clr)
    scatter.symbol = visuals.marker_types[10]
    app.run()


def IK_filter(p_n_pair):
    raise NotImplementedError


if __name__ == '__main__':
    max_angle = 90
    min_angle = 0
    total_angle = 360

    pcn_file = np.load(PATH_PCN)['pcn']
    index_file = np.load(INDEX_FILE_PATH)

    # point coordinates and normal coordinates
    p_n_pairs = get_all_p_n_pairs()
    p_n_pair = p_n_pairs[5]
    highlight_candidate_in_pc(pcn_file, p_n_pair)

    # generate some thetas
    thetas = np.random.uniform(min_angle, max_angle, 3)
    angle_sum = np.sum(thetas)
    thetas = (thetas / angle_sum) * total_angle
    print(thetas)

    # generate some r's
    # todo

    IK_filter()
