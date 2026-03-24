import taichi as ti
import numpy as np
import math

# 初始化 Taichi
ti.init(arch=ti.cpu)

# --- 1. 定义数据结构 (将 6 个面拆分为 12 个三角形) ---
num_vertices = 8
num_triangles = 12
WIDTH, HEIGHT = 800, 600

# 全局 Field 声明 (必须在 Kernel 外部)
vertices = ti.Vector.field(4, dtype=ti.f32, shape=num_vertices)
tri_indices = ti.field(ti.i32, shape=(num_triangles, 3)) 
tri_depths = ti.field(ti.f32, shape=num_triangles)
proj_tris = ti.Vector.field(2, dtype=ti.f32, shape=(num_triangles, 3))

# 立方体 8 个顶点
init_v = np.array([
    [-1, -1, -1, 1], [ 1, -1, -1, 1], [ 1,  1, -1, 1], [-1,  1, -1, 1],
    [-1, -1,  1, 1], [ 1, -1,  1, 1], [ 1,  1,  1, 1], [-1,  1,  1, 1]
], dtype=np.float32)
vertices.from_numpy(init_v)

# 12 个三角形的顶点索引 (每个面切两半)
tri_v_np = np.array([
    [0, 3, 2], [0, 2, 1], # 后面 (红)
    [4, 5, 6], [4, 6, 7], # 前面 (绿)
    [0, 1, 5], [0, 5, 4], # 下面 (蓝)
    [2, 3, 7], [2, 7, 6], # 上面 (黄)
    [0, 4, 7], [0, 7, 3], # 左面 (紫)
    [1, 2, 6], [1, 6, 5]  # 右面 (青)
], dtype=np.int32) # <--- 把这里的 np.i32 改成 np.int32
tri_indices.from_numpy(tri_v_np)

# 12 个三角形的颜色 (每两个三角形组成一个同色的面)
tri_colors_np = np.array([
    0xFF0000, 0xFF0000, # 红
    0x00FF00, 0x00FF00, # 绿
    0x0000FF, 0x0000FF, # 蓝
    0xFFFF00, 0xFFFF00, # 黄
    0xFF00FF, 0xFF00FF, # 紫
    0x00FFFF, 0x00FFFF  # 青
], dtype=np.uint32)

# --- 2. 定义计算 Kernel ---
@ti.kernel
def compute_triangles(angle_x: ti.f32, angle_y: ti.f32):
    cx, sx = ti.math.cos(angle_x), ti.math.sin(angle_x)
    cy, sy = ti.math.cos(angle_y), ti.math.sin(angle_y)

    model = ti.Matrix([
        [ cy, sx*sy,  cx*sy, 0.0],
        [0.0,    cx,    -sx, 0.0],
        [-sy, sx*cy,  cx*cy, 0.0],
        [0.0,   0.0,    0.0, 1.0]
    ])

    view = ti.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -5.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    fov, aspect, near, far = 60.0 * 3.14159 / 180.0, WIDTH/HEIGHT, 0.1, 50.0
    f = 1.0 / ti.math.tan(fov / 2.0)
    proj = ti.Matrix([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far+near)/(near-far), (2.0*far*near)/(near-far)],
        [0.0, 0.0, -1.0, 0.0]
    ])

    # 组合 MVP 矩阵
    mvp = proj @ view @ model

    # 遍历 12 个三角形
    for i in range(num_triangles):
        avg_depth = 0.0
        # 遍历三角形的 3 个顶点
        for j in ti.static(range(3)): 
            v_idx = tri_indices[i, j]
            v_clip = mvp @ vertices[v_idx]
            
            avg_depth += v_clip[2] # 累加 Z 值用于深度排序

            w = v_clip[3]
            if w == 0.0: w = 1e-5
            proj_tris[i, j] = ti.Vector([(v_clip[0]/w + 1.0)*0.5, (v_clip[1]/w + 1.0)*0.5])

        # 记录三角形平均深度
        tri_depths[i] = avg_depth / 3.0

# --- 3. 初始化 GUI ---
gui = ti.GUI("Taichi Solid Triangles", res=(WIDTH, HEIGHT), background_color=0x111111)

angle_x, angle_y = 0.0, 0.0
rot_speed = 0.15

while gui.running:
    # --- 事件监听 ---
    while gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'a': angle_y -= rot_speed
        elif gui.event.key == 'd': angle_y += rot_speed
        elif gui.event.key == 'w': angle_x += rot_speed
        elif gui.event.key == 's': angle_x -= rot_speed
        elif gui.event.key == ti.GUI.ESCAPE: gui.running = False
            
    # --- 1. 计算坐标与深度 ---
    compute_triangles(angle_x, angle_y)

    # --- 2. 深度排序 (从远到近) ---
    depths = tri_depths.to_numpy()
    sorted_indices = np.argsort(depths)[::-1] 
    
    # 提取所有顶点数据，形状为 (12, 3, 2)
    coords = proj_tris.to_numpy()

    # 根据排序结果，重新排列坐标和颜色
    sorted_coords = coords[sorted_indices]
    sorted_colors = tri_colors_np[sorted_indices]

    # --- 3. 提取三角形的三个顶点 a, b, c 用于批量绘制 ---
    a = sorted_coords[:, 0, :]
    b = sorted_coords[:, 1, :]
    c = sorted_coords[:, 2, :]

    # 批量绘制三角形
    gui.triangles(a, b, c, color=sorted_colors)

    gui.show()