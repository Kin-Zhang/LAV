import numpy as np
from collections import deque

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


import numpy as np
import enum
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arrow

PIXELS_AHEAD_VEHICLE = 120
ARROW_WIDTH = 10
# Tango colors
ORANGE = '#fcaf3e'
RED    = '#cc0000'
BLUE   = '#3465a4'
GREEN  = '#73d216'
BLACK  = '#000000'

SEM_COLORS = {
    4 : (220, 20, 60),
    5 : (153, 153, 153),
    6 : (157, 234, 50),
    7 : (128, 64, 128),
    8 : (244, 35, 232),
    10: (0, 0, 142),
    18: (220, 220, 0),
}

def visualize_semantic_processed(sem, labels=[4,6,7,10,18]):
    canvas = np.zeros(sem.shape+(3,), dtype=np.uint8)
    for i,label in enumerate(labels):
        canvas[sem==i+1] = SEM_COLORS[label]

    return canvas

def lidar_to_bev(lidar, min_x=-8,max_x=24,min_y=-16,max_y=16, pixels_per_meter=4, hist_max_per_pixel=10):
    xbins = np.linspace(
        min_x, max_x+1,
        (max_x - min_x) * pixels_per_meter + 1,
    )
    ybins = np.linspace(
        min_y, max_y+1,
        (max_y - min_y) * pixels_per_meter + 1,
    )
    # Compute histogram of x and y coordinates of points.
    hist = np.histogramdd(lidar[..., :2], bins=(xbins, ybins))[0]
    # Clip histogram
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    # Normalize histogram by the maximum number of points in a bin we care about.
    overhead_splat = hist / hist_max_per_pixel * 255.
    # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground.
    return overhead_splat[::-1,:]


def visualize_obs(rgb, yaw, control, speed, seg=None, bra=None, loc=None, com=None, plan=None, bev=None, dets=None, lidar=None, text_args=(cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)):
    """
    0 road
    1 lane
    2 stop signs
    3 red light
    4 vehicle
    5 pedestrian
    6-11 waypoints
    """
    canvas = np.array(rgb[...,::-1]) # [288,768,3]
    hegiht, width, _ = canvas.shape

    if seg is not None:

        fig, ax2=plt.subplots(1,1)

        ax2 = plt.Axes(fig, [0., 0., 1., 1.])
        ax2.set_axis_off()
        fig.add_axes(ax2)

        ax2.imshow(visualize_semantic_processed(seg), aspect='auto')
        fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close('all')
        canvas = np.concatenate([canvas, cv2.resize(img, (width, hegiht))], axis=0) # [576,768,3]


    if lidar is not None:
        lidar_viz = lidar_to_bev(lidar).astype(np.uint8)
        lidar_viz = cv2.cvtColor(lidar_viz,cv2.COLOR_GRAY2RGB)
        lidar_img = cv2.resize(lidar_viz.astype(np.uint8), (hegiht, hegiht))

    if bev is not None and dets is not None:
        bev = bev[0].mean(axis=0).cpu()
        bev_center, ego_plan_locs,other_cast_cmds,other_cast_locs = plan
        ego_plan_locs = ego_plan_locs*4 + bev_center
        other_cast_locs = other_cast_locs*4 + bev_center
        fig, ax2=plt.subplots(1,1)

        ax2 = plt.Axes(fig, [0., 0., 1., 1.])
        ax2.set_axis_off()
        fig.add_axes(ax2)

        ax2.imshow(bev, cmap='gray', aspect='auto')
        for color, det in zip([ORANGE, RED], dets):
            for x, y, w, h, cos, sin in det:
                ax2.add_patch(Rectangle((x,y)+[w,h]@np.array([[-sin,cos],[-cos,-sin]]), w*2, h*2, angle=np.rad2deg(np.arctan2(sin, cos)-np.pi/2), color=color))
                ax2.add_patch(Arrow(x,y,ARROW_WIDTH*sin,-ARROW_WIDTH*cos,color=BLACK,width=ARROW_WIDTH))
        
        for loc_x, loc_y in ego_plan_locs:
            ax2.add_patch(Circle((loc_x,loc_y),radius=0.5, color=GREEN))

        cmap = matplotlib.cm.get_cmap('jet')
        for scores, trajs in zip(other_cast_cmds, other_cast_locs):
            for i in range(6):
                score = scores[i]
                traj  = trajs[i]
                for loc_x, loc_y in traj:
                    ax2.add_patch(Circle((loc_x,loc_y),radius=0.5, color=cmap(score)))

        fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        # img = cv2.cvtColor(img,cv2.COLOR_RGB2RGB) # [480, 640, 3]

        canvas_right = np.concatenate([lidar_img, cv2.resize(img, (hegiht,hegiht))], axis=0)
        plt.close('all')

        canvas = np.concatenate([canvas, canvas_right], axis=1) # [576,768,3]

    cv2.putText(canvas, f'speed: {abs(speed):.3f}m/s', (4, 10), *text_args)
    cv2.putText(
        canvas, 
        f'steer: {control[0]:.3f} throttle: {control[1]:.3f} brake: {control[2]:.3f}',
        (4, 20), *text_args
    )
    if loc is not None:
        curr_x, curr_y = loc
        cv2.putText(canvas, f'location: ({curr_y:.2f}, {-curr_x:.2f})', (4, 30), *text_args)

    return canvas
