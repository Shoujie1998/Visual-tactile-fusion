import argparse
import logging
import time
import torch.utils.data
from torch.utils.data import RandomSampler
from inference.post_process import post_process_output
from utils.dataset_processing import evaluation
from utils.data.data_processing import TransDataset

logging.basicConfig(level=logging.INFO)


def get_device(force_cpu):
    # Check if CUDA can be used
    if torch.cuda.is_available() and not force_cpu:
        logging.info("CUDA detected. Running with GPU acceleration.")
        device = torch.device("cuda")
    elif force_cpu:
        logging.info("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
        device = torch.device("cpu")
    else:
        logging.info("CUDA is *NOT* detected. Running with only CPU.")
        device = torch.device("cpu")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate networks')

    # Network
    parser.add_argument('--network', metavar='N', type=str, nargs='+',
                        help='Path to saved networks to evaluate')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')

    # Dataset
    parser.add_argument('--dataset', type=str, default='cornell',
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--augment', action='store_true',
                        help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Evaluation
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                        help='Threshold for IOU matching')
    parser.add_argument('--iou-eval', action='store_true',
                        help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true',
                        help='Jacquard-dataset style output')

    # Misc.
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    device = get_device(args.force_cpu)
    args.network = 'path/to/checkpoint'
    test_dataset = TransDataset(args.dataset_path,
                                mode='test',
                                output_size=args.input_size,
                                ds_rotate=args.ds_rotate,
                                random_rotate=True,
                                random_zoom=True)

    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=RandomSampler(test_dataset)
    )
    logging.info('Done')

    network = args.network
    logging.info('\nEvaluating model {}'.format(network))

    net = torch.load(network)

    results = {'correct': 0, 'failed': 0}

    start_time = time.time()
    ld = len(test_data)

    with torch.no_grad():
        for x, y, didx, rot, zoom_factor in test_dataset:
            x = x.expand(1, 3, 224, 224)
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            q_out, r_out = post_process_output(lossd['pred']['pos'], lossd['pred']['radius'])

            s = evaluation.calculate_iou_match(q_out,
                                               test_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                               no_grasps=1,
                                               grasp_width=r_out,
                                               threshold=args.iou_threshold
                                               )
            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1



    avg_time = (time.time() - start_time) / len(test_data)
    logging.info('Average evaluation time per image: {}ms'.format(avg_time * 1000))
