
import time
import cv2 
import numpy as np
import os 



def draw_box(self, roi, img, linewidth, color, name=None):
    """
        roi: rectangle or polygon
        img: numpy array img
        linewith: line width of the bbox
    """
    if len(roi) > 6 and len(roi) % 2 == 0:
        pts = np.array(roi, np.int32).reshape(-1, 1, 2)
        color = tuple(map(int, color))
        img = cv2.polylines(img, [pts], True, color, linewidth)
        pt = (pts[0, 0, 0], pts[0, 0, 1]-5)
        if name:
            img = cv2.putText(img, name, pt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)
    elif len(roi) == 4:
        if not np.isnan(roi[0]):
            roi = list(map(int, roi))
            color = tuple(map(int, color))
            img = cv2.rectangle(img, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]),
                color, linewidth)
            if name:
                img = cv2.putText(img, name, (roi[0], roi[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)
    return img

def show(self, pred_trajs={}, linewidth=2, show_name=False):
    """
        pred_trajs: dict of pred_traj, {'tracker_name': list of traj}
                    pred_traj should contain polygon or rectangle(x, y, width, height)
        linewith: line width of the bbox
    """
    assert self.imgs is not None
    video = []
    cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
    colors = {}
    if len(pred_trajs) == 0 and len(self.pred_trajs) > 0:
        pred_trajs = self.pred_trajs
    for i, (roi, img) in enumerate(zip(self.gt_traj,
            self.imgs[self.start_frame:self.end_frame+1])):
        img = img.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = self.draw_box(roi, img, linewidth, (0, 255, 0),
                'gt' if show_name else None)
        for name, trajs in pred_trajs.items():
            if name not in colors:
                color = tuple(np.random.randint(0, 256, 3))
                colors[name] = color
            else:
                color = colors[name]
            img = self.draw_box(pred_trajs[0][i], img, linewidth, color,
                    name if show_name else None)
        cv2.putText(img, str(i+self.start_frame), (5, 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 2)
        cv2.imshow(self.name, img)
        cv2.waitKey(40)
        video.append(img.copy())
    return video