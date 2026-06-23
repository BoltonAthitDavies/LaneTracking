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
from config import Config

class UFLDv2:
    def __init__(self, engine_path, config_path, ori_size):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.fp16_mode = "_fp16" in engine_path.lower()
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

            # Override dtype if using fp16 mode
            if self.fp16_mode and dtype == np.float32:
                dtype = np.float16

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

        print(f"[INFO] TensorRT UFLDv2 initialized (FP16 mode: {self.fp16_mode})")

    # แก้ pred2coords ให้คืน dict {lane_id: [(x,y), ...], ...}
    def pred2coords(self, pred):
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

        max_indices_row = pred['loc_row'].argmax(1)
        valid_row = pred['exist_row'].argmax(1)

        max_indices_col = pred['loc_col'].argmax(1)
        valid_col = pred['exist_col'].argmax(1)
        # Temporary list for all detected lanes
        all_lanes = []
        # num_lanes_row = pred['loc_row'].shape[-1]  # last dimension = number of lanes in row
        # num_lanes_col = pred['loc_col'].shape[-1]  # last dimension = number of lanes in col

        # # assign left/right indices dynamically
        # row_lane_idx = list(range(num_lanes_row))
        # col_lane_idx = list(range(num_lanes_col))
        # Row-based lanes (x, y)
        for i in range(num_lane_row):
            tmp = []
            if valid_row[0, :, i].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        start = max(0, int(max_indices_row[0, k, i].item()) - self.input_width)
                        end = min(num_grid_row - 1, int(max_indices_row[0, k, i].item()) + self.input_width) + 1
                        all_ind = torch.arange(start, end, dtype=torch.long)
                        loc_logits = pred['loc_row'][0, all_ind, k, i]
                        weights = loc_logits.softmax(0)
                        out_tmp = (weights * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * self.ori_img_w
                        tmp.append((int(out_tmp.item()), int(self.row_anchor[k] * self.ori_img_h)))
            if tmp:
                all_lanes.append(tmp)

        # Col-based lanes (produce (x, y))
        for i in range(num_lane_col):
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
            if tmp:
                all_lanes.append(tmp)

        # --- Assign 2 lanes: left/right based on bottom x-position ---
        if len(all_lanes) == 0:
            lanes = {0: [], 1: []}  # left=0, right=1
        elif len(all_lanes) == 1:
            # Only one lane: decide left or right based on bottom x
            bottom_x = all_lanes[0][-1][0]
            if bottom_x < self.ori_img_w // 2:
                lanes = {0: all_lanes[0], 1: []}  # left lane
            else:
                lanes = {0: [], 1: all_lanes[0]}  # right lane
        else:
            # Two lanes: compare bottom x of each lane
            bottom_x0 = all_lanes[0][-1][0]
            bottom_x1 = all_lanes[1][-1][0]
            if bottom_x0 < bottom_x1:
                lanes = {0: all_lanes[0], 1: all_lanes[1]}  # left=0, right=1
            else:
                lanes = {0: all_lanes[1], 1: all_lanes[0]}  # left=0, right=1

        return lanes

    # def forward(self, img):
    #     im0 = img.copy()
    #     # Crop and resize
    #     img_crop = img[self.cut_height:, :, :]
    #     img_resize = cv2.resize(img_crop, (self.input_width, self.input_height), cv2.INTER_CUBIC)
    #     img_input = img_resize.astype(np.float32) / 255.0
    #     if self.fp16_mode:
    #         img_input = img_input.astype(np.float16)

    #     img_input = np.transpose(np.expand_dims(img_input, axis=-1), (3, 2, 0, 1))
    #     img_input = np.ascontiguousarray(img_input)

    #     # Send to GPU
    #     cuda.memcpy_htod(self.inputs[0]['allocation'], img_input)
    #     self.context.execute_v2(self.allocations)

    #     # Get predictions
    #     preds = {}
    #     for out in self.outputs:
    #         output = np.zeros(out['shape'], out['dtype'])
    #         cuda.memcpy_dtoh(output, out['allocation'])
    #         # preds[out['name']] = torch.tensor(output)
    #         preds[out['name']] = torch.tensor(output.astype(np.float32))  # cast back to fp32 for CPU ops

    #     # Convert predictions to 2-lane coordinates
    #     lanes = self.pred2coords(preds)  # returns {0: left_lane, 1: right_lane}
    #     # --------------------------------------------------
    #     # lane_names = {0: "Left", 1: "Middle Left", 2: "Middle Right", 3: "Right"}
    #     # print("\n=== Lane Coordinate Summary ===")
    #     # detected_count = sum(1 for pts in lanes.values() if len(pts) > 0)
    #     # print(f"Total lanes detected: {detected_count}")
    #     # for lane_id in [0,1,2,3]:
    #     #     pts = lanes[lane_id]
    #     #     print(f" Lane {lane_names[lane_id]} ({lane_id}): {len(pts)} points")
    #     #     # if pts:
    #     #     #     print(f"  First point: {pts[0]}")
    #     # --------------------------------------------------
    #     # --- Visualization ---
    #     lane_fits, im0 = self.draw_lanes(im0, lanes)

    #     return lane_fits, im0

    def forward(self, img):
        im0 = img.copy()

        # --- crop & resize as before ---
        img_crop = img[self.cut_height:, :, :]
        img_resize = cv2.resize(img_crop, (self.input_width, self.input_height), cv2.INTER_CUBIC)

        # Normalize to float32 (or float16 if engine uses fp16)
        img_input = img_resize.astype(np.float32) / 255.0
        if self.fp16_mode:
            target_dtype = np.float16
        else:
            target_dtype = np.float32

        # Convert HWC -> NCHW with batch dim 1
        # Start from HWC (H,W,3), produce (1,3,H,W)
        img_input = np.transpose(img_input, (2, 0, 1))  # -> (C, H, W)
        img_input = np.expand_dims(img_input, axis=0)  # -> (1, C, H, W)

        # Cast dtype
        if img_input.dtype != target_dtype:
            img_input = img_input.astype(target_dtype)

        # Ensure contiguous
        img_input = np.ascontiguousarray(img_input)

        # --- DEBUG: check what the engine expects ---
        try:
            binding_shape = tuple(self.inputs[0]['shape'])
            binding_dtype = np.dtype(self.inputs[0]['dtype'])
        except Exception:
            # Fallback if inputs metadata formatted differently
            binding_shape = None
            binding_dtype = None

        # Log the produced input info
        print(f"[UFLD] prepared img_input shape={img_input.shape} dtype={img_input.dtype} nbytes={img_input.nbytes}")
        if binding_shape is not None:
            print(f"[UFLD] engine binding expects shape={binding_shape} dtype={binding_dtype}")

        # If binding info exists, try to automatically adapt if possible
        if binding_shape is not None:
            # many TRT bindings include batch dim. Common: (1,3,H,W)
            expected_elems = int(np.prod(binding_shape))
            expected_bytes = expected_elems * binding_dtype.itemsize
            actual_bytes = img_input.nbytes

            # If dtype mismatch, cast
            if binding_dtype != img_input.dtype:
                try:
                    img_input = img_input.astype(binding_dtype)
                    actual_bytes = img_input.nbytes
                    print(f"[UFLD] casted img_input to {binding_dtype}")
                except Exception as e:
                    raise RuntimeError(f"[UFLD] cannot cast img_input to engine dtype {binding_dtype}: {e}")

            # If shape mismatch but compatible sizes (e.g., flattened), try reshape
            if tuple(img_input.shape) != binding_shape:
                # If total bytes equal, attempt reshape
                if actual_bytes == expected_bytes:
                    try:
                        img_input = img_input.reshape(binding_shape)
                        print(f"[UFLD] reshaped input to binding shape {binding_shape}")
                    except Exception as e:
                        raise RuntimeError(f"[UFLD] reshape to {binding_shape} failed: {e}")
                else:
                    # If sizes differ, give clear diagnostics and fail fast
                    raise RuntimeError(
                        f"[UFLD] input size mismatch: prepared nbytes={actual_bytes} != expected nbytes={expected_bytes}. "
                        f"prepared.shape={img_input.shape}, expected.shape={binding_shape}"
                    )

        # Final safety: contiguous & correct dtype
        img_input = np.ascontiguousarray(img_input)
        # Debug print before memcpy
        print(f"[UFLD] final input: shape={img_input.shape} dtype={img_input.dtype} nbytes={img_input.nbytes}")

        # --- Send to GPU (wrapped to give good error context) ---
        try:
            cuda.memcpy_htod(self.inputs[0]['allocation'], img_input)
        except Exception as e:
            # provide exact diagnostics for easier debugging
            raise RuntimeError(
                f"[UFLD] cuda.memcpy_htod failed. img_input.shape={img_input.shape} img_input.dtype={img_input.dtype} "
                f"img_input.nbytes={img_input.nbytes} | binding_shape={binding_shape} binding_dtype={binding_dtype} | error: {e}"
            )

        # Execute
        self.context.execute_v2(self.allocations)

        # Get predictions
        preds = {}
        for out in self.outputs:
            output = np.zeros(out['shape'], out['dtype'])
            cuda.memcpy_dtoh(output, out['allocation'])
            preds[out['name']] = torch.tensor(output.astype(np.float32))

        # Convert predictions to coords & draw
        lanes = self.pred2coords(preds)
        lane_fits, im0 = self.draw_lanes(im0, lanes)

        return lane_fits, im0

    
    def draw_lanes(self, im0, lanes):
        lane_colors = {0: (0, 0, 255), 1: (0, 255, 0)}
        lane_fits = {}

        for lane_id in [0, 1]:
            pts = lanes.get(lane_id, [])
            color = lane_colors[lane_id]

            for coord in pts:
                cv2.circle(im0, coord, 2, color, -1)

            if len(pts) >= 3:
                pts_arr = np.array(pts)
                pts_arr = pts_arr[np.argsort(pts_arr[:, 1])]
                x, y = pts_arr[:, 0], pts_arr[:, 1]
                try:
                    coeffs = np.polyfit(y, x, deg=2)
                    poly = np.poly1d(coeffs)
                    y_new = np.linspace(min(y), max(y), num=50)
                    x_new = poly(y_new)
                    smooth_pts = list(zip(x_new.astype(int), y_new.astype(int)))
                    lane_fits[lane_id] = {"coeffs": coeffs.tolist(), "points": smooth_pts}
                    cv2.polylines(im0, [np.array(smooth_pts, np.int32).reshape((-1, 1, 2))],
                                  isClosed=False, color=color, thickness=2)
                except Exception as e:
                    print(f"Polyfit fail on lane {lane_id}: {e}")
                    lane_fits[lane_id] = {"coeffs": None, "points": pts}
            else:
                lane_fits[lane_id] = {"coeffs": None, "points": pts}

        return lane_fits, im0