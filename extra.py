import taichi as ti
import numpy as np
import math

# 初始化 Taichi，这里使用 CPU 即可，对于 8 个顶点的计算量绰绰有余
ti.init(arch=ti.cpu)

# ----------------- 1. 定义数据结构 -----------------
num_vertices = 8
num_edges = 12

# 存放 3D 齐次坐标 (x, y, z, w) 的 Taichi 字段
vertices = ti.Vector.field(4, dtype=ti.f32, shape=num_vertices)
# 存放投影后 2D 屏幕坐标 (x, y) 的 Taichi 字段
projected_vertices = ti.Vector.field(2, dtype=ti.f32, shape=num_vertices)

# 正方体的 8 个顶点，中心在 (0,0,0)，范围 [-1, 1]
init_v = np.array([
    [-1, -1, -1, 1], # 0
    [ 1, -1, -1, 1], # 1
    [ 1,  1, -1, 1], # 2
    [-1,  1, -1, 1], # 3
    [-1, -1,  1, 1], # 4
    [ 1, -1,  1, 1], # 5
    [ 1,  1,  1, 1], # 6
    [-1,  1,  1, 1]  # 7
], dtype=np.float32)
vertices.from_numpy(init_v) # 将 Numpy 数据导入 Taichi 字段

# 12 条边的顶点索引连接
edges_np = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0], # 后面
    [4, 5], [5, 6], [6, 7], [7, 4], # 前面
    [0, 4], [1, 5], [2, 6], [3, 7]  # 侧面连接线
])

# ----------------- 2. 定义 MVP 变换 Kernel -----------------
@ti.kernel
def compute_projection(angle_x: ti.f32, angle_y: ti.f32):
    # 提前计算三角函数
    cx, sx = ti.math.cos(angle_x), ti.math.sin(angle_x)
    cy, sy = ti.math.cos(angle_y), ti.math.sin(angle_y)

    # 模型矩阵 (Model) - 绕 X 轴和 Y 轴旋转
    rot_x = ti.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,  cx, -sx, 0.0],
        [0.0,  sx,  cx, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    rot_y = ti.Matrix([
        [ cy, 0.0,  sy, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sy, 0.0,  cy, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # 视图矩阵 (View) - 沿 Z 轴向后推远 5 个单位
    view = ti.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -5.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # 透视投影矩阵 (Projection)
    fov = 60.0 * 3.14159265 / 180.0
    f = 1.0 / ti.math.tan(fov / 2.0)
    aspect = 800.0 / 600.0
    near = 0.1
    far = 50.0

    proj = ti.Matrix([
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far)],
        [0.0, 0.0, -1.0, 0.0]
    ])

    # 组合 MVP 矩阵
    mvp = proj @ view @ rot_y @ rot_x

    # 遍历并处理每个顶点（Taichi 的 for 循环在最外层时会自动并行化）
    for i in range(num_vertices):
        v = vertices[i]
        v_clip = mvp @ v # 矩阵乘以列向量

        # 透视除法
        w = v_clip[3]
        if w == 0.0:
            w = 1e-5
            
        x_ndc = v_clip[0] / w
        y_ndc = v_clip[1] / w

        # 视口变换：映射到 ti.GUI 的坐标系 [0, 1] 范围内
        # (ti.GUI 的原点 (0,0) 在左下角，(1,1) 在右上角)
        x_gui = (x_ndc + 1.0) * 0.5
        y_gui = (y_ndc + 1.0) * 0.5

        projected_vertices[i] = ti.Vector([x_gui, y_gui])

# ----------------- 3. 初始化 GUI -----------------
gui = ti.GUI("Taichi 3D Cube", res=(800, 600), background_color=0x222222)

angle_x = 0.0
angle_y = 0.0
rot_speed = 0.05

# 主循环
# 主循环
# 初始化角度和单次按键的旋转步长
angle_x = 0.0
angle_y = 0.0
rot_speed = 0.15 

# 主循环
while gui.running:
    # --- 采用事件队列驱动的按键逻辑 ---
    # 使用 while 可以确保这一帧里产生的所有按键事件都被处理掉，避免延迟
    while gui.get_event(ti.GUI.PRESS): 
        if gui.event.key == 'a':
            angle_y -= rot_speed
        elif gui.event.key == 'd':
            angle_y += rot_speed
        elif gui.event.key == 'w':
            angle_x += rot_speed
        elif gui.event.key == 's':
            angle_x -= rot_speed
        elif gui.event.key == ti.GUI.ESCAPE:
            gui.running = False
            
    # --- 调用 Kernel 进行计算 ---
    compute_projection(angle_x, angle_y)

    # --- 提取计算结果并进行批量绘制 ---
    pos_2d = projected_vertices.to_numpy()
    
    begin_pos = pos_2d[edges_np[:, 0]]
    end_pos   = pos_2d[edges_np[:, 1]]

    gui.lines(begin=begin_pos, end=end_pos, radius=2.0, color=0x00FF66)

    gui.show()