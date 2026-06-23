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
        
        self.cut_height = int(self.input_height * (1 - cfg.crop_ratio))
        
        self.num_row = cfg.num_row
        self.num_col = cfg.num_col
        
        # Anchors are relative to model_h (320)
        if self.dataset == "CULane":
            self.row_anchor = np.linspace(0.42, 1, self.num_row) 
        elif self.dataset == "CurveLanes":
            self.row_anchor = np.linspace(0.4, 1, self.num_row)
        elif self.dataset == "Tusimple":
            self.row_anchor = np.linspace(0.0, 1, self.num_row)
        
        self.col_anchor = np.linspace(0, 1, self.num_col)

        # <-- 1. Define and print lane indices based on dataset -->
        self.row_lane_idx = None
        self.col_lane_idx = None
        print(f"--- Initializing model for {self.dataset} dataset ---")

        if self.num_lanes == 2 and self.dataset == "Tusimple":
            self.row_lane_idx = [0, 1]
            self.col_lane_idx = [0, 1]
            print(f"Detecting Tusimple (2 lanes):")
            print(f"  Row anchors (lanes): {self.row_lane_idx}")
            print(f"  Col anchors (lanes): {self.col_lane_idx}")
        
        elif self.dataset == "CULane" or self.dataset == "Tusimple":
            self.row_lane_idx = [1, 2]
            self.col_lane_idx = [0, 3]
            print(f"Detecting {self.dataset} (4 lanes):")
            print(f"  Row anchors (lanes): {self.row_lane_idx}")
            print(f"  Col anchors (lanes): {self.col_lane_idx}")
        
        elif self.dataset == "CurveLanes":
            self.col_lane_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            self.row_lane_idx = None # Explicitly None
            print(f"Detecting CurveLanes:")
            print(f"  Row anchors (lanes): Not Used")
            print(f"  Col anchors (lanes): {self.col_lane_idx}")
        else:
            print(f"WARNING: Unknown dataset '{self.dataset}'. Lane indices not set.")
        print("--------------------------------------------------")

        # --- Define colors and names ---
        self.lane_colors = [
            (255, 0, 0),   # Blue (for lane 0)
            (0, 255, 0),   # Green (for lane 1)
            (0, 0, 255),   # Red (for lane 2)
            (255, 255, 0), # Cyan (for lane 3)
            (0, 255, 255), # Yellow (for lane 4)
            (255, 0, 255), # Magenta (for lane 5)
            (255, 128, 0), # Orange (for lane 6)
            (128, 0, 255), # Purple (for lane 7)
            (0, 128, 0),   # Dark Green (for lane 8)
            (128, 128, 128) # Gray (for lane 9)
        ]
        self.lane_color_names = [
            "Blue", "Green", "Red", "Cyan", "Yellow", 
            "Magenta", "Orange", "Purple", "Dark Green", "Gray"
        ]
        
        # --- Define polynomial degree ---
        self.polyfit_degree = 2

        # --- Initialize adaptive lane width (in pixels) ---
        self.lane_width_px_est = None  
        self.lane_width_alpha = 0.9  # smoothing factor for running average

        # --- Add state for last known good lanes ---
        self.last_known_left_lane = None
        self.last_known_right_lane = None

    def pred2coords(self, pred):
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape
        max_indices_row = pred['loc_row'].argmax(1)
        valid_row = pred['exist_row'].argmax(1)
        max_indices_col = pred['loc_col'].argmax(1)
        valid_col = pred['exist_col'].argmax(1)

        pred['loc_row'] = pred['loc_row']
        pred['loc_col'] = pred['loc_col']

        coords_with_indices = []
        
        row_lane_idx = self.row_lane_idx
        col_lane_idx = self.col_lane_idx

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
                            out_tmp = out_tmp / (num_grid_row - 1) * self.input_width
                            tmp.append((int(out_tmp), int(self.row_anchor[k] * self.input_height)))
                    if tmp:
                        coords_with_indices.append((i, tmp))
                    
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
                            out_tmp = out_tmp / (num_grid_col - 1) * self.input_height
                            tmp.append((int(self.col_anchor[k] * self.input_width), int(out_tmp)))
                    if tmp:
                        coords_with_indices.append((i, tmp))
        
        if self.num_lanes == 2 and self.dataset == "Tusimple":
            lanes_for_sorting = [pts for idx, pts in coords_with_indices]
            
            if len(lanes_for_sorting) == 0:
                lanes_dict = {0: [], 1: []}
            elif len(lanes_for_sorting) == 1:
                bottom_x = lanes_for_sorting[0][-1][0]
                if bottom_x < self.input_width // 2:
                    lanes_dict = {0: lanes_for_sorting[0], 1: []}
                else:
                    lanes_dict = {0: [], 1: lanes_for_sorting[0]}
            else:
                lanes_for_sorting = lanes_for_sorting[:2] 
                bottom_x0 = lanes_for_sorting[0][-1][0]
                bottom_x1 = lanes_for_sorting[1][-1][0]
                if bottom_x0 < bottom_x1:
                    lanes_dict = {0: lanes_for_sorting[0], 1: lanes_for_sorting[1]}
                else:
                    lanes_dict = {0: lanes_for_sorting[1], 1: lanes_for_sorting[0]}
            
            final_coords = []
            for lane_idx, lane_pts in lanes_dict.items():
                if lane_pts:
                    final_coords.append((lane_idx, lane_pts))
            return final_coords

        return coords_with_indices

    def fit_and_draw_lanes(self, coords, im_to_draw, fps):
        """
        Fits polynomials to lanes, draws them, and returns fit parameters AND fitted points.
        """
        # This will store tuples of (lane_idx, fit_parameters)
        lane_fits_params = []
        # <-- NEW: This will store tuples of (lane_idx, fitted_points_list) -->
        fitted_lanes_points = []
        
        print(f"--- Detected {len(coords)} lanes ---")

        for lane_idx, lane_pts in coords:
            
            color_index = lane_idx % len(self.lane_colors)
            color = self.lane_colors[color_index] 
            color_name = self.lane_color_names[color_index]
            
            print(f"  > Lane {lane_idx}: {color_name}")
            
            if len(lane_pts) > self.polyfit_degree:
                try:
                    y_values = np.array([p[1] for p in lane_pts])
                    x_values = np.array([p[0] for p in lane_pts])

                    fit_params = np.polyfit(y_values, x_values, self.polyfit_degree)
                    lane_fits_params.append((lane_idx, fit_params))
                    
                    lane_fit_function = np.poly1d(fit_params)

                    y_draw_min = int(np.min(y_values))
                    y_draw_max = int(np.max(y_values))
                    plot_y = np.linspace(y_draw_min, y_draw_max, num=20)
                    plot_x = lane_fit_function(plot_y)

                    # <-- NEW: Store the list of (x, y) fitted points -->
                    points_to_draw_list = np.asarray([plot_x, plot_y]).T.astype(np.int32)
                    fitted_lanes_points.append((lane_idx, points_to_draw_list))

                    # Draw the fitted line
                    points_for_polylines = points_to_draw_list.reshape((-1, 1, 2))
                    cv2.polylines(im_to_draw, [points_for_polylines], isClosed=False, color=color, thickness=2)
                    # Also draw raw points
                    for coord in lane_pts:
                        cv2.circle(im_to_draw, coord, 3, color, -1)

                except np.linalg.LinAlgError:
                    print(f"  > Lane {lane_idx}: Failed to fit polynomial. Drawing points.")
                    # <-- NEW: Add original points as fallback -->
                    fitted_lanes_points.append((lane_idx, lane_pts))
                    for coord in lane_pts:
                        cv2.circle(im_to_draw, coord, 3, color, -1)
            else:
                print(f"  > Lane {lane_idx}: Not enough points to fit. Drawing points.")
                # <-- NEW: Add original points as fallback -->
                fitted_lanes_points.append((lane_idx, lane_pts))
                for coord in lane_pts:
                    cv2.circle(im_to_draw, coord, 3, color, -1)
        
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(im_to_draw, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
        
        # <-- NEW: Return fitted_lanes_points as well -->
        return im_to_draw, lane_fits_params, fitted_lanes_points
    
    def forward(self, img_ori):
        start_time = time.time()

        im0 = img_ori.copy()
        ori_h, ori_w = im0.shape[:2]
        
        img_model = cv2.resize(img_ori, (self.input_width, self.input_height), interpolation=cv2.INTER_CUBIC)

        img = img_model[self.cut_height:, :, :]
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
            
        coords_model_space = self.pred2coords(preds)
        
        scale_x = ori_w / self.input_width
        scale_y = ori_h / self.input_height
        
        coords_ori_space = [] # This is the raw detected points
        
        for lane_idx, lane_pts in coords_model_space:
            lane_ori_space = []
            for (x_model, y_model) in lane_pts:
                x_ori = int(x_model * scale_x)
                y_ori = int(y_model * scale_y)
                if x_ori > 0 and y_ori > 0:
                    lane_ori_space.append((x_ori, y_ori))
            
            if lane_ori_space:
                coords_ori_space.append((lane_idx, lane_ori_space))
        end_time = time.time()

        fps = 1 / (end_time - start_time)

        # <-- MODIFIED: Get all 3 return values -->
        im0_draw, lane_fits_params, fitted_lane_coords = self.fit_and_draw_lanes(coords_ori_space, im0, fps)
        
        # <-- MODIFIED: Return all 4 values -->
        return coords_ori_space, im0_draw, lane_fits_params, fitted_lane_coords, fps

    def visualize_lane_offset(self, im0, left_lane, right_lane, lane_width, pixel2meter=True):
        """
        Compute and visualize lateral distance from lane center in meters.
        
        --- MODIFIED to handle Case 4 reconstruction ---
        """
        h, w, _ = im0.shape
        ref_y = h // 2 + h // 3
        x_car = w // 2
        fallback = None # Will be 'left', 'right', or 'both'

        # --- NEW: Handle Case 4 (No Lanes) at the START ---
        if (left_lane is None or len(left_lane) == 0) and \
           (right_lane is None or len(right_lane) == 0):
            print(f"No lane detected Case 4. Attempting reconstruction...")
            if self.last_known_left_lane is not None and self.last_known_right_lane is not None:
                print(f"  > Reconstructing from last known good lanes.")
                left_lane = self.last_known_left_lane   # Use stale data
                right_lane = self.last_known_right_lane # Use stale data
                fallback = "both" # Set new fallback flag
            else:
                # No lanes this frame AND no history. Give up.
                print(f"  > No lanes detected and no history. Cannot reconstruct.")
                return im0, None
        # --- END NEW ---

        # Ensure numpy arrays
        left_lane = np.array(left_lane) if left_lane is not None and len(left_lane) > 0 else None
        right_lane = np.array(right_lane) if right_lane is not None and len(right_lane) > 0 else None

        x_left, x_right = None, None

        # --- Case 1: both lanes available (either new or from fallback) ---
        if left_lane is not None and right_lane is not None:
            try:
                x_left = np.interp(ref_y, left_lane[:, 1], left_lane[:, 0])
                x_right = np.interp(ref_y, right_lane[:, 1], right_lane[:, 0])
            except Exception as e:
                self.get_logger().warn(f"Interpolation failed: {e}")
                return im0, None
            
            # --- NEW: Create and draw the lane overlay ---
            try:
                # Create an overlay image, copy of im0 will be modified
                overlay = np.zeros_like(im0)
                
                # Get the points for the polygon
                # We need to reverse the right lane points to connect them correctly
                polygon_points = np.concatenate((left_lane, np.flipud(right_lane)), axis=0)
                
                # Draw the polygon on the overlay
                # Using a semi-transparent green (0, 255, 0)
                cv2.fillPoly(overlay, [polygon_points.astype(np.int32)], (0, 255, 0))
                
                # Blend the overlay with the original image
                alpha = 0.3 # Transparency factor
                im0 = cv2.addWeighted(im0, 1, overlay, alpha, 0)
                
            except Exception as e:
                self.get_logger().warn(f"Failed to draw lane overlay: {e}")
                # Don't fail the whole function, just log the warning
            # --- END NEW ---

            lane_width_px = x_right - x_left

            # Sanity check: (Only apply if NOT using 'both' fallback)
            if (lane_width_px < 100 or lane_width_px > w * 0.9) and fallback != "both":
                if abs(x_left - x_car) > abs(x_right - x_car):
                    fallback = "left"
                    x_right = x_left + (self.lane_width_px_est or w * 0.45)
                    cv2.circle(im0, (int(x_right), ref_y), 12, (0, 255, 0), -1)
                    print(f"reconstruct right Case 1")
                else:
                    fallback = "right"
                    x_left = x_right - (self.lane_width_px_est or w * 0.45)
                    cv2.circle(im0, (int(x_left), ref_y), 12, (0, 0, 255), -1)
                    print(f"reconstruct left Case 1")
                lane_width_px = x_right - x_left

            # Update adaptive lane width *only if it's a good, new, dual detection*
            if lane_width_px > 100 and fallback is None:
                if self.lane_width_px_est is None:
                    self.lane_width_px_est = lane_width_px
                else:
                    self.lane_width_px_est = (
                        self.lane_width_alpha * self.lane_width_px_est
                        + (1 - self.lane_width_alpha) * lane_width_px
                    )
        # --- Case 2: only left lane detected ---
        elif left_lane is not None:
            fallback = "right"
            try:
                x_left = np.interp(ref_y, left_lane[:, 1], left_lane[:, 0])
            except Exception as e:
                self.get_logger().warn(f"Interpolation failed (left): {e}")
                return im0, None
            x_right = x_left + (self.lane_width_px_est or w * 0.45)
            cv2.circle(im0, (int(x_right), ref_y), 6, (0, 255, 0), -1)
            print(f"reconstruct right Case 2")

        # --- Case 3: only right lane detected ---
        elif right_lane is not None:
            fallback = "left"
            try:
                x_right = np.interp(ref_y, right_lane[:, 1], right_lane[:, 0])
            except Exception as e:
                self.get_logger().warn(f"Interpolation failed (right): {e}")
                return im0, None
            x_left = x_right - (self.lane_width_px_est or w * 0.45)
            cv2.circle(im0, (int(x_left), ref_y), 6, (0, 0, 255), -1)
            print(f"reconstruct left Case 3")

        # --- Final check ---
        if x_left is None or x_right is None:
             print(f"Fatal error: x_left or x_right is still None. Cannot calculate offset.")
             return im0, None

        # Lane center
        x_center = int((x_left + x_right) / 2.0)
        if pixel2meter:
            lane_width_px = x_right - x_left
            if lane_width_px <= 0:
                return im0, None
            meters_per_pixel = lane_width / lane_width_px
            offset_m = (x_car - x_center) * meters_per_pixel
            cv2.putText(im0, f"Offset: {offset_m:.2f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            offset_m = x_car - x_center
            cv2.putText(im0, f"Offset: {offset_m:.2f} pixel", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # === Visualization ===
        cv2.line(im0, (int(x_left), ref_y), (int(x_right), ref_y), (0, 255, 255), 2)
        cv2.line(im0, (x_car, (h//2+h//3)-10), (x_car, (h//2+h//3)+10), (0, 255, 0), 2)
        cv2.line(im0, (x_center, (h//2+h//3)-10), (x_center, (h//2+h//3)+10), (255, 0, 0), 2)
        
        # === Debug overlay if fallback ===
        if fallback == "left":
            cv2.putText(im0, "RECONSTRUCTED LEFT", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif fallback == "right":
            cv2.putText(im0, "RECONSTRUCTED RIGHT", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # --- NEW: Add visualization for "both" ---
        elif fallback == "both":
            cv2.putText(im0,
                        "RECONSTRUCTED BOTH (USING LAST)",
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 165, 255), # Orange color
                        2,
                        cv2.LINE_AA)
        # --- END NEW ---

        return im0, offset_m
    
def get_args():
    """Helper function to get command line arguments."""
    parser = argparse.ArgumentParser()
    # You might need to adjust default paths
    parser.add_argument('--config_path', default='configs/test_byd/culane_res34_480x320.py', help='path to config file', type=str)
    parser.add_argument('--engine_path', default='weights/culane_res34_480x320.engine',
                        help='path to engine file', type=str)
    parser.add_argument('--video_path', default='example.mp4', help='path to video file', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cap = cv2.VideoCapture(args.video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        sys.exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Optional: Setup video writer
    # output_filename = "lane_fit_output.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # video_writer = cv2.VideoWriter(output_filename, fourcc, video_fps, (frame_width, frame_height))
    # print(f"Saving output to {output_filename}...")

    # Initialize the UFLDv2 model
    isnet = UFLDv2(args.engine_path, args.config_path)
    
    frame_count = 0

    fps_all = []

    while True:
        success, img = cap.read()
        if not success:
            print("Finished processing video.")
            break
        
        frame_count += 1
        
        # --- NEW: Updated forward call to get lane_fits ---
        coords_ori_space, im0_draw, lane_fits_params, fitted_lane_coords, fps = isnet.forward(img)

        fps_all.append(fps)
        
        # You can now use lane_fits here for other logic if needed
        # Example: print(f"Frame {frame_count} Fits: {lane_fits}")
        
        # Optional: Write frame to output video
        # video_writer.write(im0_draw)

        # --- MODIFIED: Extract left/right points from coords_ori_space ---
        # This replaces the old loop over `lane_fits.items()`
        # `visualize_lane_offset` just needs the raw points.
        left_lane, right_lane = None, None

        if isnet.num_lanes == 2 and isnet.dataset == "Tusimple":
            ego = [0,1]
        elif isnet.dataset == "CULane" or isnet.dataset == "Tusimple":
            ego = [1,2]
        elif isnet.dataset == "CurveLanes":
            ego = [4,5]
        else:
            print(f"WARNING: Unknown dataset '{args.dataset}'. Lane indices not set.")

        for lane_idx, lane_pts in fitted_lane_coords:
            if lane_idx == ego[0]: # 1 is ego left lane for CuLane
                left_lane = lane_pts
                isnet.last_known_left_lane = lane_pts
            elif lane_idx == ego[1]: # 2 is ego right lane for CuLane
                right_lane = lane_pts
                isnet.last_known_right_lane = lane_pts

        im0_with_offset, offset_m = isnet.visualize_lane_offset(im0_draw, left_lane, right_lane, lane_width=0.42, pixel2meter=False)


        # Display the result
        # cv2.imshow("result", im0_draw)
        cv2.imshow("result", im0_with_offset)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_average = np.mean(fps_all)
    print(f"FPS average: {fps_average}")

    # Release all resources
    cap.release()
    # video_writer.release()
    cv2.destroyAllWindows()
