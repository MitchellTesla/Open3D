#!/usr/bin/env python
import numpy as np
import open3d as o3d
import sys

COLORS = [[1.0, 0.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0],
          [0.5, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.5, 1.0],
          [0.0, 1.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0],
          [0.5, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.33, 1.0],
          [1.0, 0.33, 0.33, 1.0], [1.0, 0.75, 0.33, 1.0], [1.0, 1.0, 0.33, 1.0],
          [0.75, 1.0, 0.33, 1.0], [0.33, 1.0, 0.33, 1.0],
          [0.33, 1.0, 0.75, 1.0], [0.33, 1.0, 1.0, 1.0], [0.33, 0.75, 1.0, 1.0],
          [0.33, 0.33, 1.0, 1.0], [0.75, 0.33, 1.0, 1.0], [1.0, 0.33, 1.0, 1.0],
          [1.0, 0.33, 0.75, 1.0]]


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("    ", sys.argv[0], " path/to/kitti/dataset/sequences/NN")
        exit(0)
    path = sys.argv[1]

    o3d.visualization.webrtc_server.enable_webrtc()

    pts = np.load(path + "/velodyne/000000.npy")
    tcloud = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
    tcloud.point["points"] = o3d.core.Tensor.from_numpy(pts)

    labels = np.load(path + "/labels/000000.npy")
    labels = np.reshape(labels, (labels.shape[0],))
    max_label = max(labels)
    labels = np.array(labels, dtype="float32")
    tcloud.point["__visualization_scalar"] = o3d.core.Tensor.from_numpy(labels)

    gradient_pts = []
    for i in range(0, max_label):
        gradient_pts.append(
            o3d.visualization.rendering.Gradient.Point(
                float(i) / (max_label - 1.0), COLORS[i]))

    gradient = o3d.visualization.rendering.Gradient(gradient_pts)
    mat = o3d.visualization.rendering.Material()
    mat.shader = "unlitGradient"
    mat.gradient = gradient
    mat.scalar_min = 0.0
    mat.scalar_max = max_label

    o3d.visualization.draw([{
        "name": "000000",
        "geometry": tcloud,
        "material": mat
    }],
                           bg_color=[0, 0, 0, 1])


if __name__ == "__main__":
    main()
