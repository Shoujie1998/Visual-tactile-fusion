import argparse
import logging
import os
import time
import glob
import torch.utils.data
from torch.utils.data import RandomSampler
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation

logging.basicConfig(level=logging.INFO)


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
    parser.add_argument('--use-depth', type=int, default=0,
                        help='Use Depth image for evaluation (1/0)')
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
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the network output')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    args = parse_args()

    device = get_device(args.force_cpu)
    args.network = glob.glob(
        os.path.join(r'/media/disk2/yhx21_data/robotic-grasping/logs/220522_0857_training_cornell_unet_iou0.45/',
                     r'epoch*'))
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    test_dataset = Dataset(args.dataset_path,
                           mode='test_cup6_new',
                           output_size=args.input_size,
                           ds_rotate=args.ds_rotate,
                           random_rotate=True,
                           random_zoom=True,
                           include_depth=args.use_depth,
                           include_rgb=args.use_rgb)

    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=RandomSampler(test_dataset)
    )
    logging.info('Done')

    for network in args.network:
        logging.info('\nEvaluating model {}'.format(network))

        net = torch.load(network)

        results = {'correct': 0, 'failed': 0}

        if args.jacquard_output:
            jo_fn = network + '_jacquard_output.txt'
            with open(jo_fn, 'w') as f:
                pass

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

        print(results['correct'] / 500)


        avg_time = (time.time() - start_time) / len(test_data)
        logging.info('Average evaluation time per image: {}ms'.format(avg_time * 1000))

        if args.iou_eval:
            logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                                                      results['correct'] + results['failed'],
                                                      results['correct'] / (results['correct'] + results['failed'])))

        if args.jacquard_output:
            logging.info('Jacquard output saved to {}'.format(jo_fn))

        del net
        torch.cuda.empty_cache()
