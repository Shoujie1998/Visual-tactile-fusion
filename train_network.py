import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import datetime
import json
import logging
import sys
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data.data_processing import TransDataset
from utils.dataset_processing import evaluation
from torch.utils.data import RandomSampler


def get_device(force_cpu):
    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda")
    elif force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--network', type=str, default='tgcnn',
                        help='Network name in inference/models')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for training (1/0)')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='Use dropout for training (1/0)')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='Dropout prob for training (0-1)')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='Internal channel size for the network')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                        help='Threshold for IOU matching')

    # Datasets
    parser.add_argument('--dataset', type=str, default='cornell',
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to dataset')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=True,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')

    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')

    args = parser.parse_args()
    return args


def validate(net, device, val_data, iou_threshold):
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        for x, y, didx, rot, zoom_factor in val_data:
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']
            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld
            q_out, r_out = post_process_output(lossd['pred']['pos'], lossd['pred']['radius'])

            s = evaluation.calculate_iou_match(q_out,
                                               val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                               no_grasps=1,
                                               grasp_width=r_out,
                                               threshold=iou_threshold
                                               )

            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch):
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    while batch_idx <= batches_per_epoch:
        for x, y, _, _, _ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(xc, yc)
            loss = lossd['loss']

            if batch_idx % 10 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx
    print('epoch is:', epoch)
    return results


def run():
    args = parse_args()

    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}'.format(dt)
    save_folder = os.path.join(args.logdir, net_desc + '_' + args.network + '_' + 'iou{}'.format(
        args.iou_threshold))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)

    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    device = get_device(args.force_cpu)

    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    train_dataset = TransDataset(args.dataset_path,
                                 mode='train',
                                 output_size=args.input_size,
                                 ds_rotate=args.ds_rotate,
                                 random_rotate=True,
                                 random_zoom=True)
    test_dataset = TransDataset(args.dataset_path,
                                mode='test',
                                output_size=args.input_size,
                                ds_rotate=args.ds_rotate,
                                random_rotate=True,
                                random_zoom=True)

    logging.info('Train Dataset size is {}'.format(train_dataset.length))
    logging.info('Test Dataset size is {}'.format(test_dataset.length))

    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=RandomSampler(train_dataset)
    )
    val_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=RandomSampler(test_dataset)
    )
    logging.info('Done')

    logging.info('Loading Network...')
    input_channels = 3
    network = get_network(args.network)
    net = network(
        input_channels=input_channels,
        dropout=args.use_dropout,
        prob=args.dropout_prob,
        channel_size=args.channel_size
    )

    net = net.to(device)
    logging.info('Done')
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(net.parameters())
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))

    summary(net, (input_channels, args.input_size, args.input_size))
    f = open(os.path.join(save_folder, 'arch.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, args.input_size, args.input_size))
    sys.stdout = sys.__stdout__
    f.close()

    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch)
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        logging.info('Validating...')

        test_results = validate(net, device, val_data, args.iou_threshold)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))

        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 1) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.4f' % (epoch, iou)))
            best_iou = iou


if __name__ == '__main__':
    run()
