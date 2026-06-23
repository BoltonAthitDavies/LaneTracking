import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
import os

def plot_log(filename=None, image_center_val=0.0):
    # If no filename is provided, find the latest pid_log_*.csv
    if filename is None:
        list_of_files = glob.glob('pid_log_*.csv') 
        if not list_of_files:
            print("Error: No 'pid_log_*.csv' files found in the current directory.")
            return
        filename = max(list_of_files, key=os.path.getctime)
        print(f"Auto-detected latest log file: {filename}")
    else:
        print(f"Loading log file: {filename}")

    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return

    # Verify new columns exist
    # required_columns = ['time_sec', 'offset', 'target_speed', 'feedback_velocity', 'speed_error', 'target_steer', 'feedback_steering', 'steer_error']
    required_columns = ['image_center', 'lane_center', 'time_sec', 'offset', 'target_speed', 'feedback_velocity', 'speed_error', 'target_steer', 'feedback_steering', 'steer_error']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV is missing columns. Expected: {required_columns}")
        print(f"Found: {list(df.columns)}")
        return

    # --- Plotting ---
    # 5 Rows: Offset | Speed (T vs A) | Speed Error | Steer (T vs A) | Steer Error
    # fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
    # fig.suptitle(f'PID Controller Full Analysis\n{filename}', fontsize=14)

    # # 1. Lane Offset
    # axs[0].plot(df['time_sec'], df['offset'], label='Offset', color='black', linewidth=1.5)
    # axs[0].set_ylabel('Offset (m)', fontsize=10)
    # axs[0].set_title('Lane Offset', fontsize=10, fontweight='bold')
    # axs[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    # axs[0].grid(True, linestyle=':', alpha=0.6)
    # axs[0].legend(loc='upper right')

    # # 2. Speed Comparison (Target vs Feedback)
    # axs[1].plot(df['time_sec'], df['target_speed'], label='Target Speed', color='blue', linestyle='--', linewidth=1.5)
    # axs[1].plot(df['time_sec'], df['feedback_velocity'], label='Feedback Speed', color='green', linewidth=1.5)
    # axs[1].set_ylabel('Speed (m/s)', fontsize=10)
    # axs[1].set_title('Speed Tracking', fontsize=10, fontweight='bold')
    # axs[1].grid(True, linestyle=':', alpha=0.6)
    # axs[1].legend(loc='upper right')

    # # 3. Speed Error
    # axs[2].plot(df['time_sec'], df['speed_error'], label='Speed Error', color='red', linewidth=1.2)
    # axs[2].set_ylabel('Error (m/s)', fontsize=10)
    # axs[2].set_title('Speed Error (Target - Feedback)', fontsize=10, fontweight='bold')
    # axs[2].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    # axs[2].grid(True, linestyle=':', alpha=0.6)

    # # 4. Steering Comparison (Target vs Feedback)
    # axs[3].plot(df['time_sec'], df['target_steer'], label='Target Steer', color='purple', linestyle='--', linewidth=1.5)
    # axs[3].plot(df['time_sec'], df['feedback_steering'], label='Feedback Steer', color='orange', linewidth=1.5)
    # axs[3].set_ylabel('Steer (rad/deg)', fontsize=10)
    # axs[3].set_title('Steering Tracking', fontsize=10, fontweight='bold')
    # axs[3].grid(True, linestyle=':', alpha=0.6)
    # axs[3].legend(loc='upper right')

    # # 5. Steering Error
    # axs[4].plot(df['time_sec'], df['steer_error'], label='Steer Error', color='brown', linewidth=1.2)
    # axs[4].set_ylabel('Error', fontsize=10)
    # axs[4].set_xlabel('Time (seconds)', fontsize=12)
    # axs[4].set_title('Steering Error (Target - Feedback)', fontsize=10, fontweight='bold')
    # axs[4].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    # axs[4].grid(True, linestyle=':', alpha=0.6)

    # 6 Rows: Offset | Center Refs | Speed (T vs A) | Speed Error | Steer (T vs A) | Steer Error
    fig, axs = plt.subplots(6, 1, figsize=(10, 18), sharex=True)
    fig.suptitle(f'PID Controller Full Analysis\n{filename}', fontsize=14)
    # 1. Lane Offset (Alone)
    axs[0].plot(df['time_sec'], df['offset'], label='Offset', color='black', linewidth=1.5)
    axs[0].set_ylabel('Offset (m)', fontsize=10)
    axs[0].set_title('Lane Offset (Error)', fontsize=10, fontweight='bold')
    axs[0].axhline(0, color='gray', linestyle='--', linewidth=0.8) 
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend(loc='upper right')

    # 2. Lane Center vs Image Center (NEW)
    # axs[1].axhline(image_center_val, label=f'Image Center ({image_center_val})', color='red', linestyle='--', linewidth=1.5)
    axs[1].plot(df['time_sec'], df['image_center'], label='Target Pose', color='blue', linestyle='--', linewidth=1.5)
    axs[1].plot(df['time_sec'], df['lane_center'], label='Lane Center (Param)', color='purple', linewidth=1.5)
    axs[1].set_ylabel('Position (px/m)', fontsize=10)
    axs[1].set_title('Lane Center Reference vs Image Center', fontsize=10, fontweight='bold')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend(loc='upper right')

    # 3. Speed Comparison (Target vs Feedback)
    axs[2].plot(df['time_sec'], df['target_speed'], label='Target Speed', color='blue', linestyle='--', linewidth=1.5)
    axs[2].plot(df['time_sec'], df['feedback_velocity'], label='Feedback Speed', color='green', linewidth=1.5)
    axs[2].set_ylabel('Speed (m/s)', fontsize=10)
    axs[2].set_title('Speed Tracking', fontsize=10, fontweight='bold')
    axs[2].grid(True, linestyle=':', alpha=0.6)
    axs[2].legend(loc='upper right')

    # 4. Speed Error
    axs[3].plot(df['time_sec'], df['speed_error'], label='Speed Error', color='red', linewidth=1.2)
    axs[3].set_ylabel('Error (m/s)', fontsize=10)
    axs[3].set_title('Speed Error (Target - Feedback)', fontsize=10, fontweight='bold')
    axs[3].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axs[3].grid(True, linestyle=':', alpha=0.6)

    # 5. Steering Comparison (Target vs Feedback)
    axs[4].plot(df['time_sec'], df['target_steer'], label='Target Steer', color='purple', linestyle='--', linewidth=1.5)
    axs[4].plot(df['time_sec'], df['feedback_steering'], label='Feedback Steer', color='orange', linewidth=1.5)
    axs[4].set_ylabel('Steer (rad/deg)', fontsize=10)
    axs[4].set_title('Steering Tracking', fontsize=10, fontweight='bold')
    axs[4].grid(True, linestyle=':', alpha=0.6)
    axs[4].legend(loc='upper right')

    # 6. Steering Error
    axs[5].plot(df['time_sec'], df['steer_error'], label='Steer Error', color='brown', linewidth=1.2)
    axs[5].set_ylabel('Error', fontsize=10)
    axs[5].set_xlabel('Time (seconds)', fontsize=12)
    axs[5].set_title('Steering Error (Target - Feedback)', fontsize=10, fontweight='bold')
    axs[5].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axs[5].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot PID log data.')
    parser.add_argument('filename', nargs='?', help='Path to the CSV file (optional, defaults to latest)')
    
    args = parser.parse_args()
    plot_log(args.filename)

# python3 plot_pid_log.py your/path/filename.csv