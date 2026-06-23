import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.config import Config
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
class UFLDv2:
    def __init__(self, engine_path, config_path, ori_size):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        
        cfg = Config.fromfile(config_path)
        
        self.ori_img_w, self.ori_img_h = ori_size
        # self.cut_height = int(cfg.train_height * (1 - cfg.crop_ratio))
        self.cut_height = int(self.ori_img_h * (1 - cfg.crop_ratio))

        self.input_width = cfg.train_width
        self.input_height = cfg.train_height

        self.num_row = cfg.num_row
        self.num_col = cfg.num_col

        self.row_anchor = np.linspace(0, 1, self.num_row)
        # self.row_anchor = np.linspace(0.22222222, 0.98611111, self.num_row) # From cfg.row_anchor in demo.py
        self.col_anchor = np.linspace(0, 1, self.num_col) # From cfg.col_anchor in demo.py

    # แก้ pred2coords ให้คืน dict {lane_id: [(x,y), ...], ...}
    def pred2coords(self, pred):
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

        max_indices_row = pred['loc_row'].argmax(1)
        valid_row = pred['exist_row'].argmax(1)

        max_indices_col = pred['loc_col'].argmax(1)
        valid_col = pred['exist_col'].argmax(1)

        # Prepare empty containers for all 4 lane indices (0..3)
        lanes = {0: [], 1: [], 2: [], 3: []}
        row_lane_idx = [1, 2]  # middle-left, middle-right
        col_lane_idx = [0, 3]  # left, right

        # Row-based lanes (produce (x, y))
        for i in row_lane_idx:
            tmp = []
            # threshold logic same as original
            if valid_row[0, :, i].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        start = max(0, int(max_indices_row[0, k, i].item()) - self.input_width)
                        end = min(num_grid_row - 1, int(max_indices_row[0, k, i].item()) + self.input_width) + 1
                        all_ind = torch.arange(start, end, dtype=torch.long)
                        loc_logits = pred['loc_row'][0, all_ind, k, i]  # shape: (len(all_ind),)
                        weights = loc_logits.softmax(0)
                        out_tmp = (weights * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * self.ori_img_w
                        tmp.append((int(out_tmp.item()), int(self.row_anchor[k] * self.ori_img_h)))
            lanes[i] = tmp

        # Col-based lanes (produce (x, y))
        for i in col_lane_idx:
            tmp = []
            if valid_col[0, :, i].sum() > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        start = max(0, int(max_indices_col[0, k, i].item()) - self.input_width)
                        end = min(num_grid_col - 1, int(max_indices_col[0, k, i].item()) + self.input_width) + 1
                        all_ind = torch.arange(start, end, dtype=torch.long)
                        loc_logits = pred['loc_col'][0, all_ind, k, i]
                        weights = loc_logits.softmax(0)
                        out_tmp = (weights * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_col - 1) * self.ori_img_h
                        tmp.append((int(self.col_anchor[k] * self.ori_img_w), int(out_tmp.item())))
            lanes[i] = tmp

        return lanes


    def forward(self, img):
        im0 = img.copy()
        img = img[self.cut_height:, :, :]
        img = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(np.float32(img[:, :, :, np.newaxis]), (3, 2, 0, 1))
        img = np.ascontiguousarray(img)
        cuda.memcpy_htod(self.inputs[0]['allocation'], img)
        self.context.execute_v2(self.allocations)
        preds = {}
        for out in self.outputs:
            output = np.zeros(out['shape'], out['dtype'])
            cuda.memcpy_dtoh(output, out['allocation'])
            preds[out['name']] = torch.tensor(output)
        # ใน forward หลังได้ preds:
        lanes = self.pred2coords(preds)  # lanes is dict {0: [...], 1: [...], 2: [...], 3: [...]}
# --------------------------------------------------
        # lane_names = {0: "Left", 1: "Middle Left", 2: "Middle Right", 3: "Right"}
        # print("\n=== Lane Coordinate Summary ===")
        # detected_count = sum(1 for pts in lanes.values() if len(pts) > 0)
        # print(f"Total lanes detected: {detected_count}")
        # for lane_id in [0,1,2,3]:
        #     pts = lanes[lane_id]
        #     print(f" Lane {lane_names[lane_id]} ({lane_id}): {len(pts)} points")
        #     # if pts:
        #     #     print(f"  First point: {pts[0]}")
# --------------------------------------------------
        # กำหนดสีให้แต่ละ lane
        lane_colors = {
            0: (0, 0, 255),    # Left = Red
            1: (255, 0, 0),    # Middle Left = Blue
            2: (0, 255, 0),    # Middle Right = Green
            3: (0, 255, 255)   # Right = Yellow
        }

        # วาดจุดตาม lane ด้วยสีเฉพาะ
        for lane_id, pts in lanes.items():
            # color = (0,255,0)
            color = lane_colors.get(lane_id, (255,255,255))  # default = White
            for coord in pts:
                cv2.circle(im0, coord, 2, color, -1)

        # inside your forward(), replace polyfit section:
        lane_fits = {}
        for lane_id, pts in lanes.items():
            color = lane_colors.get(lane_id, (255,255,255))

            if len(pts) < 3:
                lane_fits[lane_id] = {"coeffs": None, "points": pts}
                continue  

            pts = np.array(pts)
            pts = pts[np.argsort(pts[:,1])]   # sort by y
            x = pts[:,0]
            y = pts[:,1]

            # coeffs, smooth_pts = self.robust_polyfit(y, x, max_degree=5) #<=== FOR RANSACC Method
            coeffs, smooth_pts = self.adaptive_polyfit(y, x, max_degree=5, dispersion_thresh=300) # <===== ADJUST dispersion_thresh FOR thresold POLYFIT

            if coeffs is not None and len(smooth_pts) > 1:
                lane_fits[lane_id] = {
                    "coeffs": coeffs.tolist(),
                    "points": smooth_pts
                }
                cv2.polylines(im0, [np.array(smooth_pts, np.int32).reshape((-1,1,2))],
                            isClosed=False, color=color, thickness=2)
            else:
                lane_fits[lane_id] = {"coeffs": None, "points": pts.tolist()}
                if len(pts) > 1:
                    cv2.polylines(im0, [pts.reshape((-1,1,2))], False, color, 2)
                elif len(pts) == 1:
                    cv2.circle(im0, tuple(pts[0]), 3, color, -1)

        # === Debug: print lane data for robot control ===
        print("\n=== Debug: print lane data for robot control ===")
        for lane_id, data in lane_fits.items():
            if data["coeffs"] is not None:
                print(f"Lane {lane_id} poly coeffs: {data['coeffs']}")
                print(f"  First 5 points: {data['points'][:5]}")

        cv2.imshow("result", im0)

    # helper: robust adaptive polynomial fit
    def robust_polyfit(self, y, x, max_degree=5, dispersion_thresh=50): #<=== FOR RANSACC Method
        """
        Fit polynomial x = f(y) adaptively.
        - ถ้า std(x) > dispersion_thresh → ใช้ linear/quadratic เท่านั้น
        - ใช้ RANSAC เพื่อตัด outlier
        """
        x_std = np.std(x)
        if x_std > dispersion_thresh:
            degrees_to_try = [1, 2]   # บังคับใช้ degree ต่ำ
        else:
            degrees_to_try = range(1, max_degree+1)

        best_model = None
        best_error = float('inf')
        best_degree = 1
        coeffs = None

        for deg in degrees_to_try:
            try:
                model = make_pipeline(PolynomialFeatures(deg), RANSACRegressor())
                model.fit(y.reshape(-1,1), x)
                x_pred = model.predict(y.reshape(-1,1))
                error = np.mean((x_pred - x)**2)

                if error < best_error:
                    best_error = error
                    best_model = model
                    best_degree = deg
            except Exception:
                continue

        if best_model is None:
            return None, []

        # generate smooth path
        y_new = np.linspace(min(y), max(y), num=50).reshape(-1,1)
        x_new = best_model.predict(y_new)

        try:
            coeffs = np.polyfit(y_new.ravel(), x_new, deg=best_degree)
        except Exception:
            coeffs = None

        smooth_pts = list(zip(x_new.astype(int), y_new.ravel().astype(int)))
        return coeffs, smooth_pts

    def adaptive_polyfit(self, y, x, max_degree=5, dispersion_thresh=50):
        """
        Adaptive polynomial fitting without RANSAC.
        - x = f(y)
        - Use lower degree if lane points are highly dispersed
        """
        x_std = np.std(x)
        print("x_std: ", x_std)
        if x_std > dispersion_thresh:
            deg = 2  # high dispersion → linear
        else:
            deg = max_degree  # low dispersion → higher degree

        try:
            coeffs = np.polyfit(y, x, deg)
            poly = np.poly1d(coeffs)
            y_new = np.linspace(min(y), max(y), num=50)
            x_new = poly(y_new)
            smooth_pts = list(zip(x_new.astype(int), y_new.astype(int)))
            return coeffs, smooth_pts
        except Exception as e:
            # fallback: return original points
            return None, list(zip(x.astype(int), y.astype(int)))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/culane_res34.py', help='path to config file', type=str)
    parser.add_argument('--engine_path', default='weights/culane_res34.engine',
                        help='path to engine file', type=str)
    parser.add_argument('--video_path', default='example.mp4', help='path to video file', type=str)
    parser.add_argument('--ori_size', default=(1280, 720), help='size of original frame', type=tuple)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cap = cv2.VideoCapture(args.video_path)
    isnet = UFLDv2(args.engine_path, args.config_path, args.ori_size)
    while True:
        success, img = cap.read()
        isnet.forward(img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
