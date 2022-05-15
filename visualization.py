import time
import cv2
import open3d as o3d
import numpy as np
from tqdm import tqdm
import os


def load_background_img(img_path=None, h=432, w=768):
    if img_path is not None:
        out_img = cv2.imread(img_path)
    else:
        out_img = 40 * np.ones((w, h, 3))
    return out_img


def extract_video(scene, start_t, end_t, scale=10, h=432, w=768, background_img_path=None,
                  out=".\\out3\\", show_va=False, color=((128,0,0), (0,128,0), (0,0,128))):
    bgi = load_background_img(img_path=background_img_path, h=h, w=w)
    # video_size = (bgi.shape[1], bgi.shape[0])
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter("out.mp4", fourcc, 30, video_size, True)

    for t in tqdm(range(start_t, end_t)):
        img = bgi.copy()
        line_set = scene.x_i[t][scene.M]
        for triangle in line_set:
            pt1, pt2, pt3 = np.array(triangle * scale, dtype=int)
            cv2.line(img, pt1, pt2, color=color[2], thickness=2)
            cv2.line(img, pt2, pt3, color=color[2], thickness=2)
            cv2.line(img, pt3, pt1, color=color[2], thickness=2)
        if show_va:
            xv_i = scene.x_i[t] + scene.v_i[t] * 30
            xa_i = scene.x_i[t] + scene.a_i[t] * 30
            for x_i, v_i, a_i in zip(scene.x_i[t], xv_i, xa_i):
                pt1 = np.array(x_i * scale, dtype=int)
                pt2 = np.array(v_i * scale, dtype=int)
                pt3 = np.array(a_i * scale, dtype=int)
                cv2.line(img, pt1, pt2, color=color[0], thickness=2)
                cv2.line(img, pt1, pt3, color=color[1], thickness=2)
        cv2.imwrite(out + str(t).zfill(4) + ".png", img)
        # video.write(img)
    # video.release()

def generate_video3d(scene, start_t, end_t, step, boundary_ids, out=".\\out3\\", given_xi = None):
    # idx = scene.M
    if given_xi is not None:
        x_i = np.load(given_xi)
    else:
        x_i = scene.x_i
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    triangles = o3d.utility.Vector3iVector(boundary_ids)
    for t in range(start_t, end_t, step):
        vertices = o3d.utility.Vector3dVector(x_i[t])
        mesh = o3d.geometry.TriangleMesh(vertices, triangles)
        mesh.compute_vertex_normals()
        mesh.transform([ [np.sqrt(0.5), 0, -np.sqrt(0.5), 0], [0, 1, 0, 0], [np.sqrt(0.5), 0, np.sqrt(0.5), 0], [0, 0, 0, 1]])

        mesh.paint_uniform_color([1, 0.706, 0])
        if given_xi is None:
            o3d.io.write_triangle_mesh(out+"test{}.ply".format(t), mesh)
        vis.add_geometry(mesh)
        if t == start_t:
            ctr = vis.get_view_control()
            param = ctr.convert_to_pinhole_camera_parameters()
        else:
            ctr_now = vis.get_view_control()
            ctr_now.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.clear_geometries()


        # time.sleep(0.5)
        # o3d.visualization.draw_geometries([mesh])
        # input()


# def generate_video_from_ply(start, end, path):
#     assert os.path.exists(path)
#     for f in range(start, end)