import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Sample JSON lines (can be read from a file)
annotations = [
    {"lanes": [[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 362, 345, 329, 310, 291, 273, 257, 240, 215, 199, 181, 162, 149, 132, 116, 97, 77, 58, 40, 37, 26, 12, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2], [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 557, 549, 542, 534, 528, 522, 515, 509, 504, 498, 493, 488, 483, 478, 474, 468, 463, 458, 453, 448, 443, 439, 434, 430, 425, 419, 414, 409, 404, 398, 393, 388, 383, 379, 373, 368], [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 733, 737, 741, 747, 752, 759, 765, 770, 776, 782, 789, 797, 805, 813, 820, 828, 836, 844, 851, 860, 868, 875, 884, 891, 900, 908, 915, 923, 929, 935, 941, 946, 952, 957, 963, 968]], "h_samples": [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710], "raw_file": "dataset/training/images/img/0973.jpg"},
    {"lanes": [[-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 573, 551, 517, 490, 463, 442, 422, 390, 362, 340, 315, 291, 268, 241, 220, 195, 176, 148, 128, 106, 83, 57, 41, 36, 20, 9, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2], [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 803, 783, 756, 740, 718, 700, 683, 672, 656, 647, 633, 623, 614, 605, 595, 584, 577, 569, 559, 548, 538, 531, 524, 514, 506, 499, 491, 485, 478, 471, 464, 457, 450, 443, 435, 429]], "h_samples": [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710], "raw_file": "dataset/training/images/img/0786.jpg"}

    # {
    #     "lanes": [[-2, 168, 163, 157, 150, 143, 137, 131, 124, 118, 111, 104, 98],
    #               [-2, 242, 244, 249, 256, 263, 270, 277, 284, 291, 299, 305, 312]],
    #     "h_samples": [110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230],
    #     "raw_file": "dataset/training/images/img/183.jpg"
    # },
    # {
    #     "lanes": [[-2, 233, 222, 208, 194, 181, 167, 155, 143, 132, 120, 111, 102],
    #               [-2, 314, 309, 302, 299, 299, 301, 303, 305, 308, 310, 314, 318]],
    #     "h_samples": [110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230],
    #     "raw_file": "dataset/training/images/img/20.jpg"
    # },
    # {"lanes": [[-2, 304, 287, 262, 240, 221, 202, 188, 174, 159, 145, 131, 118], 
    #            [-2, 404, 398, 390, 383, 375, 369, 362, 357, 354, 354, 355, 355]], 
    #            "h_samples": [110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230], 
    #            "raw_file": "dataset/training/images/img/154.jpg"}

]

def plot_lane_segmentation(anno, i):
    img_path = anno['raw_file']
    lanes = anno['lanes']
    h_samples = anno['h_samples']

    # Load the original image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load image: {img_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a blank label image
    label = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw each lane with a unique label (1–4)
    for lane_idx, lane in enumerate(lanes):
        points = []
        for x, y in zip(lane, h_samples):
            if x >= 0:
                points.append((int(x), int(y)))
        if len(points) >= 2:
            cv2.polylines(label, [np.array(points)], isClosed=False, color=lane_idx + 1, thickness=5)

    # Optional: overlay label on the image
    overlay = image.copy()
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
    for lane_idx, lane in enumerate(lanes):
        points = []
        for x, y in zip(lane, h_samples):
            if x >= 0:
                points.append((int(x), int(y)))
        if len(points) >= 2:
            cv2.polylines(overlay, [np.array(points)], isClosed=False, color=colors[lane_idx % 4], thickness=2)

    # Show both label map and overlay
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.title('Binary Segmentation Labels')
    plt.imshow(label, cmap='gray')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title('Overlay on Original Image')
    plt.imshow(overlay)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"tusimple_lane_{i}.png")  # Save with correct index
    plt.close()

# Run on all annotations
for i, anno in enumerate(annotations):
    plot_lane_segmentation(anno, i)
