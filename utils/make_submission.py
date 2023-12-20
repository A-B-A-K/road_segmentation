import argparse
from submission import masks_to_submission

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate submission from predictions.')
    parser.add_argument('pred_dir', type=str, help='Directory containing prediction images.')
    parser.add_argument('output_csv', type=str, help='Output CSV file name.')
    return parser.parse_args()

def main():
    args = parse_arguments()

    submission_filename = f'{args.pred_dir}/{args.output_csv}'
    image_filenames = []
    for i in range(1, 51):
        image_filename = f'{args.pred_dir}/pred_test_{i}.png'
        print(image_filename)
        image_filenames.append(image_filename)
    
    masks_to_submission(submission_filename, *image_filenames)

if __name__ == '__main__':
    main()
