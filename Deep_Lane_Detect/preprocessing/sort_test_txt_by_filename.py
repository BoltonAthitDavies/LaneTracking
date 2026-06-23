import os

def extract_number_from_path(path: str) -> int:
    """
    Extracts numeric part from filename like '0131.jpg' -> 131
    """
    filename = os.path.basename(path)
    name_without_ext = os.path.splitext(filename)[0]
    try:
        return int(name_without_ext)
    except ValueError:
        return float('inf')  # Push non-numeric names to the end

def sort_paths(input_txt: str, output_txt: str = None):
    with open(input_txt, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    sorted_lines = sorted(lines, key=extract_number_from_path)

    output_txt = output_txt or input_txt  # overwrite if not specified
    with open(output_txt, 'w') as f:
        for line in sorted_lines:
            f.write(line + '\n')

    print(f'[INFO] Sorted paths written to {output_txt}.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Sort test.txt paths by filename number.')
    parser.add_argument('-i', '--input', type=str, default='test.txt',
                        help='Input test.txt file.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file (default: overwrite input).')
    args = parser.parse_args()

    sort_paths(args.input, args.output)

# python preprocessing/sort_test_txt_by_filename.py --input test.txt --output sorted_test.txt
