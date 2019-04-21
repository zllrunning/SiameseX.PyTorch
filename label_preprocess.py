import argparse
import json

parser = argparse.ArgumentParser(description='label preprocessing')

parser.add_argument('--file', default='./data/vot2018.txt', type=str,
                    help='json file.')

parser.add_argument('--output_file', default='./data/vot2018_new.txt', type=str,
                    help='output json file.')

parser.add_argument('--path', default='/home/zll/data/tracking/vot2018/', type=str,
                    help='path to change.')

args = parser.parse_args()


if args.file == './data/vot2018.txt':
    org_path = '/home/leeyh/Downloads/vot2017/'  # '/home/leeyh/Downloads/vot2017/soldier/00000001.jpg'
elif args.file == './data/ilsvrc_vid.txt':
    org_path = '/home/leeyh/Downloads/data/'  # '/home/leeyh/Downloads/data/ILSVRC/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00044000/000000.JPEG'


with open(args.file, 'r') as outfile:
    root = json.load(outfile)
    
for sequence in range(0, len(root)):
    print(sequence)
    for items in range(0, len(root[sequence])):
        for item in range(0, len(root[sequence][items])):
            root[sequence][items][item][0] = root[sequence][items][item][0].replace(org_path, args.path)

with open(args.output_file, 'w') as outfile:
    json.dump(root, outfile)
