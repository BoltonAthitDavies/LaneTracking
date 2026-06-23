export PYTHONPATH=$PYTHONPATH:$(pwd)

V1_Step to train UFLD-V2
1. python preprocessing/extract_frames_from_video_resize.py --input "raw_dataset" --output "dataset" --frequency 5
2. python preprocessing/split_train_test_img_only.py --input-directory "dataset/high_bright"
3. python -m  preprocessing.find_hsv --input /home/alex1/LaneTracking/Deep_Lane_Detect/dataset/high_bright/training/images/img/0222.jpg
4. - python preprocessing/generate_tusimple_from_original.py --image-dir dataset/high_bright/test/images/img --output-json test_label.json
   - python preprocessing/generate_tusimple_from_original.py --image-dir dataset/high_bright/training/images/img --output-json label_data_training.json
5. python preprocessing/test_task_to_tusimple.py
6. python preprocessing/convert_tusimple.py --root /home/alex1/LaneTracking/Deep_Lane_Detect
7. python train.py configs/tusimple_res34.py

V2_Step to train UFLD-V2
1. python preprocessing/extract_frames_from_video_resize.py --input "raw_dataset" --output "dataset" --frequency 5
2. python -m  preprocessing.find_hsv --input /home/alex1/LaneTracking/Deep_Lane_Detect/dataset/high_bright/training/images/img/0222.jpg
3. python preprocessing/generate_lane_masks_contours.py --input "dataset/high_bright"
4. python preprocessing/edit_binary_mask.py
5. python preprocessing/split_train_test_img_two_folder.py -i dataset/high_bright
6. python preprocessing/gen_mask_to_tusimple.py
   -  mask_dir='dataset/high_bright/test/bin_masks/img',
      original_image_dir='dataset/high_bright/test/images/img',
      output_json_filename='test_label.json',
   -  mask_dir='dataset/high_bright/training/bin_masks/img',
      original_image_dir='dataset/high_bright/training/images/img',
      output_json_filename='label_data_training.json',
6.1 For check labels from tusimple on raw image.
   -  python preprocessing/gen_tusimple_to_mask.py
7. python preprocessing/test_task_to_tusimple.py
8. python preprocessing/convert_tusimple.py --root /home/alex1/LaneTracking/Deep_Lane_Detect
9. python train.py configs/tusimple_res34.py

---

Step to visualize
1. python preprocessing/sort_test_txt_by_filename.py --input test.txt --output sorted_test.txt
2. python demo.py configs/tusimple_res34.py

---

Tensorrt Deploy
1. python deploy/pt2onnx.py --config_path configs/tusimple_res34.py --model_path weights/tusimple_res34.pth
2. trtexec --onnx=weights/tusimple_res34.onnx --saveEngine=weights/tusimple_res34.engine
3. python deploy/trt_infer.py --config_path  configs/tusimple_res34.py --engine_path weights/tusimple_res34.engine --video_path example.mp4