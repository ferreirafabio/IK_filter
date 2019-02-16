import numpy as np
import os
import itertools
from vispy import app, visuals, scene, color

PATH_PCN = "./866_rho0.303173_azi345.000000_ele89.226024_theta-0.009812_xcam0.000000_ycam0.000000_zcam0.303173_scale0.188282_xdim0.111353_ydim0.146529_zdim0.03974700000_pcn.npz"
INDEX_FILE_PATH = "./866_rho0.303173_azi345.000000_ele89.226024_theta-0.009812_xcam0.000000_ycam0.000000_zcam0.303173_scale0.188282_xdim0.111353_ydim0.146529_zdim0.03974700000_par_robotiq3f_full.npy"


def get_all_p_n_pairs(pcn_file, index_file):
    p_n_pairs = []
    for i in range(index_file.shape[0]):
        idx1, idx2, idx3 = index_file[i, :3].astype(int)
        p1, n1 = pcn_file[idx1][:3], pcn_file[idx1][3:]
        p2, n2 = pcn_file[idx2][:3], pcn_file[idx2][3:]
        p3, n3 = pcn_file[idx3][:3], pcn_file[idx3][3:]
        p_n_pairs.append([idx1, idx2, idx3, p1, p2, p3, n1, n2, n3])

    return p_n_pairs

def highlight_candidate_in_pc(pcn_file, candidate_idx_coords):
    idx1 = candidate_idx_coords[0]
    idx2 = candidate_idx_coords[1]
    idx3 = candidate_idx_coords[2]
    p1 = candidate_idx_coords[3]
    p2 = candidate_idx_coords[4]
    p3 = candidate_idx_coords[5]

    scale_normals = 0.1

    n1 = candidate_idx_coords[6] * scale_normals
    n2 = candidate_idx_coords[7] * scale_normals
    n3 = candidate_idx_coords[8] * scale_normals

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
    candidate_marker = color.Color(color="yellow").rgba
    normals_marker = color.Color(color="red").rgba
    colors = np.tile(clr, (pcn_file.shape[0], 1))
    candidate_markers = np.tile(candidate_marker, (3, 1))
    normals_markers = np.tile(normals_marker, (3, 1))
    colors = np.concatenate([colors, candidate_markers, normals_markers], axis=0)

    # grasp coordinates
    data = np.append(pcn_file[:, :3], np.expand_dims(p1, axis=0), axis=0)
    data = np.append(data, np.expand_dims(p2, axis=0), axis=0)
    data = np.append(data, np.expand_dims(p3, axis=0), axis=0)
    # normal coordinates
    data = np.append(data, np.expand_dims(n1, axis=0), axis=0)
    data = np.append(data, np.expand_dims(n2, axis=0), axis=0)
    data = np.append(data, np.expand_dims(n3, axis=0), axis=0)

    sizes = np.append(np.tile(10, 2045), np.tile(20, 6))
    scatter.set_data(data, size=sizes, face_color=colors)

    #scatter.set_data(np.expand_dims(p1, axis=0), size=5, face_color=clr)
    scatter.symbol = visuals.marker_types[10]
    app.run()


def _dismiss_parallel_normals(n1, n2, n3, atol=1e-03):
    """ returns true if normals all point in the same direction"""
    dots = []
    for pair in itertools.combinations([n1, n2, n3], 2):
        dots.append(np.dot(pair[0], pair[1]))

    return np.isclose(dots, 1, atol=atol).all() or np.isclose(dots, -1, atol=atol).all()

def _gripper_fingertip_order(p1,p2,p3,n1,n2,n3):
    normal = np.cross((p1-p2),(p1-p3))
    n = normal/(np.linalg.norm(normal) + 1e-16)
    c = (p1 + p2 + p3) / 3.0
    tip_order = [0,1,2]
    if np.inner(n1,n2) > 0.8:
        tip_order[0] = 3
        if np.cross((n1 - c),(n3 - c))[2] > 0:
            tip_order[2] = 1
            tip_order[1] = 2
        else:
            tip_order[2] = 2
            tip_order[1] = 1 
    if np.inner(n1,n3) > 0.8:
        tip_order[0] = 2
        if np.cross((n1 - c),(n2 - c))[2] > 0:
            tip_order[2] = 1
            tip_order[1] = 3
        else:
            tip_order[2] = 3
            tip_order[1] = 1 
 
    if np.inner(n2,n3) > 0.8:
        tip_order[0] = 1
        if np.cross((n2 - c),(n - c))[2] > 0:
            tip_order[2] = 2
            tip_order[1] = 3
        else:
            tip_order[2] = 3
            tip_order[1] = 2 
    return tip_order

def _dismiss_contacts_normals(n1,n2,n3,theta1,theta2,eps=10):
    a12 = np.arccos(np.sum(np.array(n1),np.array(n2))) / np.pi * 180.0
    a13 = np.arccos(np.sum(np.array(n1),np.array(n3))) / np.pi * 180.0
    if a12 < theta1 + eps and a12 > theta1 - eps and a13 < theta2 + eps and a13 > theta2 - eps:
      return False
    else:
      return True


def _dismiss_unreachable(p1, p2, p3, r1, r2, r3, l):
    l1 = r1 + l
    l2 = r2 + l
    l3 = r3 + l
    list1 = [p1, p2, p3]
    list2 = [l1, l2, l3]
    # todo
    raise NotImplementedError


def _dismiss_close_contacts(p1, p2, p3, atol=5e-02):
    """ returns true if all three contact points are within a radius of "atol" meters """
    return np.all([np.allclose(pair[0], pair[1], atol=atol) for pair in itertools.combinations([p1, p2, p3], 2)])

def IK_filter(p_n_pairs,theta1=120.0,theta2=120.0):
    len_before = len(p_n_pairs)
    print("total contact point tuples before filtering: ", len_before)
    for p_n_pair in p_n_pairs:
        p1, p2, p3, n1, n2, n3 = p_n_pair[3], p_n_pair[4], p_n_pair[5], p_n_pair[6], p_n_pair[7], p_n_pair[8]

        # dismiss samples for which all three normals point in the same direction
        #dismiss = _dismiss_parallel_normals(n1, n2, n3)
        #if dismiss:
        #    p_n_pairs.remove(p_n_pair)
        #    continue

        # dismiss samples for which the contact points are too close to each other for grasping
        #dismiss = _dismiss_close_contacts(p1, p2, p3)
        #if dismiss:
        #    p_n_pairs.remove(p_n_pair)
        #    continue

        fingertip_order = _gripper_fingertip_order(p1,p2,p3,n1,n2,n3)
        print("fingertip_order",fingertip_order)
        dismiss = _dismiss_contacts_normals(p_n_pair[fingertip_order[0]+5],p_n_pair[fingertip_order[1]+5],p_n_pair[fingertip_order[2]+5],theta1,theta2)
        if dismiss:
            p_n_pairs.remove(p_n_pair)
 
        #dismiss = _dismiss_unreachable(p1, p2, p3, r1, r2, r3, l)

    len_after = len(p_n_pairs)
    print("total contact point tuples after filtering: ", len_after)
    print("removed: ", len_before-len_after)



if __name__ == '__main__':
    max_angle = 90
    min_angle = 0
    total_angle = 360

    l = 0.1  # assume finger is 10cm long
    r1 = 0.09
    r2 = 0.12
    r3 = 0.11

    pcn_file = np.load(PATH_PCN)['pcn']
    index_file = np.load(INDEX_FILE_PATH)

    index_file = index_file[:50000]
    # point coordinates and normal coordinates
    p_n_pairs = get_all_p_n_pairs(pcn_file, index_file)
    #highlight_candidate_in_pc(pcn_file, p_n_pairs[5])
    print("len",len(p_n_pairs[0]))

    #for pair in p_n_pairs:
        # generate thetas
        #thetas = np.random.uniform(min_angle, max_angle, 3)
        #angle_sum = np.sum(thetas)
        #thetas = list((thetas / angle_sum) * total_angle)
        #pair.extend(thetas)
        #pair.extend([r1, r2, r3, l])

    IK_filter(p_n_pairs=p_n_pairs)
