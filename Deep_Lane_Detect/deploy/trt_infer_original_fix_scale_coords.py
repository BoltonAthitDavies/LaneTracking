import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import argparse
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.config import Config


class UFLDv2:
    def __init__(self, engine_path, config_path):
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
        self.dataset = cfg.dataset
        self.num_lanes = cfg.num_lanes
        self.input_width = cfg.train_width
        self.input_height = cfg.train_height
        
        # self.cut_height = int(cfg.train_height * (1 - cfg.crop_ratio)) #original
        # Base cut_height on the model_h (e.g., 320)
        self.cut_height = int(self.input_height * (1 - cfg.crop_ratio))
        

        self.num_row = cfg.num_row
        self.num_col = cfg.num_col
        
        # Anchors are relative to model_h (320)
        if self.dataset == "CULane":
            self.row_anchor = np.linspace(0.42, 1, self.num_row) 
        # ==== Curvelane =====
        elif self.dataset == "CurveLanes":
            self.row_anchor = np.linspace(0.4, 1, self.num_row)
        elif self.dataset == "Tusimple":
            self.row_anchor = np.linspace(0.0, 1, self.num_row)
        
        self.col_anchor = np.linspace(0, 1, self.num_col)

    def pred2coords(self, pred):
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape
        max_indices_row = pred['loc_row'].argmax(1)
        valid_row = pred['exist_row'].argmax(1)
        max_indices_col = pred['loc_col'].argmax(1)
        valid_col = pred['exist_col'].argmax(1)

        pred['loc_row'] = pred['loc_row']
        pred['loc_col'] = pred['loc_col']

        coords = []
        row_lane_idx = None
        col_lane_idx = None
        if self.num_lanes == 2 and self.dataset == "Tusimple":
            row_lane_idx = [0, 1]
            col_lane_idx = [0, 1]
        #  ==== Culane ====
        elif self.dataset == "CULane" or self.dataset == "Tusimple":
            row_lane_idx = [1, 2]
            col_lane_idx = [0, 3]
        # ==== Curvelane =====
        elif self.dataset == "CurveLanes":
            # row_lane_idx = [3, 4, 5, 6]
            # col_lane_idx = [2, 7]
            # col_lane_idx = [0, 1, 2, 7, 8, 9]
            # row_lane_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            col_lane_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # row_lane_idx = [4, 5]
            # col_lane_idx = [3, 6]

        if row_lane_idx != None:
            for i in row_lane_idx:
                tmp = []
                if valid_row[0, :, i].sum() > num_cls_row / 2:
                    for k in range(valid_row.shape[1]):
                        if valid_row[0, k, i]:
                            all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - self.input_width),
                                                            min(num_grid_row - 1,
                                                                max_indices_row[0, k, i] + self.input_width) + 1)))

                            out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                            
                            # 3. CHANGED: Scale to `self.input_width`
                            out_tmp = out_tmp / (num_grid_row - 1) * self.input_width
                            
                            # 4. CHANGED: Scale to `self.input_height`
                            tmp.append((int(out_tmp), int(self.row_anchor[k] * self.input_height)))
                    coords.append(tmp)
                    
        if col_lane_idx != None:
            for i in col_lane_idx:
                tmp = []
                if valid_col[0, :, i].sum() > num_cls_col / 4:
                    for k in range(valid_col.shape[1]):
                        if valid_col[0, k, i]:
                            all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - self.input_width),
                                                            min(num_grid_col - 1,
                                                                max_indices_col[0, k, i] + self.input_width) + 1)))
                            out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                            
                            # 5. CHANGED: Scale to `self.input_height`
                            out_tmp = out_tmp / (num_grid_col - 1) * self.input_height
                            
                            # 6. CHANGED: Scale to `self.input_width`
                            tmp.append((int(self.col_anchor[k] * self.input_width), int(out_tmp)))
                    coords.append(tmp)
        
        if self.num_lanes == 2 and self.dataset == "Tusimple":
            # --- Assign 2 lanes: left/right based on bottom x-position ---
            if len(coords) == 0:
                lanes = {0: [], 1: []}  # left=0, right=1
            elif len(coords) == 1:
                # Only one lane: decide left or right based on bottom x
                bottom_x = coords[0][-1][0]
                if bottom_x < self.input_width // 2:
                    lanes = {0: coords[0], 1: []}  # left lane
                else:
                    lanes = {0: [], 1: coords[0]}  # right lane
            else:
                # Two lanes: compare bottom x of each lane
                bottom_x0 = coords[0][-1][0]
                bottom_x1 = coords[1][-1][0]
                if bottom_x0 < bottom_x1:
                    lanes = {0: coords[0], 1: coords[1]}  # left=0, right=1
                else:
                    lanes = {0: coords[1], 1: coords[0]}  # left=0, right=1
            coords = list(lanes.values())

        # This now returns coords in model_size space (1600x320)
        return coords

    # 7. CHANGED: `forward` now handles all scaling
    def forward(self, img_ori):
        # Keep original image for drawing and final scaling
        im0 = img_ori.copy()
        ori_h, ori_w = im0.shape[:2]
        
        # a. Resize original image (e.g., 1280x720) to model size (1600x320)
        img_model = cv2.resize(img_ori, (self.input_width, self.input_height), interpolation=cv2.INTER_CUBIC)

        # b. Preprocess the model-sized image for the network
        img = img_model[self.cut_height:, :, :]
        img = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(np.float32(img[:, :, :, np.newaxis]), (3, 2, 0, 1))
        img = np.ascontiguousarray(img)
        
        # c. Run inference
        cuda.memcpy_htod(self.inputs[0]['allocation'], img)
        self.context.execute_v2(self.allocations)
        preds = {}
        for out in self.outputs:
            output = np.zeros(out['shape'], out['dtype'])
            cuda.memcpy_dtoh(output, out['allocation'])
            preds[out['name']] = torch.tensor(output)
            
        # d. Get coords in model space (e.g., 1600x320)
        coords_model_space = self.pred2coords(preds)
        
        # e. Scale coords from model space (1600x320) back to original space (1280x720)
        scale_x = ori_w / self.input_width
        scale_y = ori_h / self.input_height
        
        coords_ori_space = []
        for lane in coords_model_space:
            lane_ori_space = []
            for (x_model, y_model) in lane:
                x_ori = int(x_model * scale_x)
                y_ori = int(y_model * scale_y)
                if x_ori > 0 and y_ori > 0:
                    lane_ori_space.append((x_ori, y_ori))
            coords_ori_space.append(lane_ori_space)
        
        # f. Return original-space coords and the original image for drawing
        return coords_ori_space, im0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/culane_res34.py', help='path to config file', type=str)
    parser.add_argument('--engine_path', default='weights/culane_res34.engine',
                        help='path to engine file', type=str)
    parser.add_argument('--video_path', default='example.mp4', help='path to video file', type=str)
    
    # 8. REMOVED: `ori_size` is no longer needed here.
    # parser.add_argument('--ori_size', default=(1600, 320), help='size of original frame', type=tuple)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cap = cv2.VideoCapture(args.video_path)
    
    # <-- 3. ADDED VIDEO WRITER SETUP -->
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        exit()

    # Get video properties for VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define output filename and codec
    output_filename = "/home/nvidia/LaneTracking/Deep_Lane_Detect/rama6_culane_ufldv2.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 'mp4v' for .mp4
    
    # Initialize VideoWriter
    video_writer = cv2.VideoWriter(output_filename, fourcc, video_fps, (frame_width, frame_height))
    
    print(f"Recording video to {output_filename}...")
    # <-- END OF VIDEO WRITER SETUP -->

    isnet = UFLDv2(args.engine_path, args.config_path)
    
    while True:
        start_time = time.time()

        success, img = cap.read()
        if not success:
            break
        
        # 10. CHANGED: The main loop is now clean.
        # Just pass the original image (e.g., 1280x720) directly.
        coords, im_to_draw = isnet.forward(img)
        
        # Draw the returned (correctly scaled) coordinates on the original image
        for lane in coords:
            for coord in lane:
                cv2.circle(im_to_draw, coord, 2, (0, 255, 0), -1)
        
        # <-- 3. ADDED FPS CALCULATION AND DISPLAY -->
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        fps_text = f"FPS: {fps:.2f}"
        
        # Put text on the image (top-left corner)
        cv2.putText(im_to_draw, 
                    fps_text, 
                    (10, 30),  # Position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1,  # Font scale
                    (0, 255, 0),  # Color (B, G, R)
                    2) # Thickness
        
        # <-- 3. WRITE THE FRAME TO THE VIDEO FILE -->
        # video_writer.write(im_to_draw)

        cv2.imshow("result", im_to_draw)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # <-- 3. RELEASE ALL RESOURCES -->
    # print(f"Video saved to {output_filename}")
    cap.release()
    # video_writer.release()
    cv2.destroyAllWindows()