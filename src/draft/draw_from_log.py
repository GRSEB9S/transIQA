import matplotlib.pyplot as plt
import argparse
import os.path as osp

# train: loss, lcc, srocc
# test: loss, lcc, srocc

parser = argparse.ArgumentParser('draw picture from log file')
parser.add_argument('--log_path', type=str, default='./log/data_log_0174',
                    help='['']path for log file')
parser.add_argument('--start', type=int, default=0,
                    help='read start from line')

args = parser.parse_args()

log_path = args.log_path
start = args.start


def read_file(file_path):

    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            lines.append(line.rstrip('\n'))

    return lines


def get_train_score(lines):
    """
    return train loss, lcc, srocc and x:step
    :param lines:
    :return:
    """

    x = []
    y_loss = []
    y_lcc = []
    y_srocc = []

    for line in lines:
        split_line = line.split()
        if split_line[0] == 'train':
            epoch = int(split_line[1][6:])  # epoch:x
            percent = float(split_line[2][8:])  # percent:x
            loss = float(split_line[3][5:])  # loss:x
            lcc = float(split_line[4][4:])  # lcc:x
            srocc = float(split_line[5][6:])  # srocc:x

            x.append(epoch + percent)
            y_loss.append(loss)
            y_lcc.append(lcc)
            y_srocc.append(srocc)

    return x, y_loss, y_lcc, y_srocc


def get_test_score(lines):

    x = []
    y_loss = []
    y_lcc = []
    y_srocc = []
    for i in range(len(lines)):
        split_line = lines[i].split()
        if split_line[0] == 'test':
            epoch = int(lines[i-1].split()[1][6:]) \
                if lines[i-1].split()[0] == 'train' \
                else int(lines[i+1].split()[0][6:])
            percent = float(lines[i-1].split()[2][8:]) \
                if lines[i-1].split()[0] == 'train' \
                else int(lines[i+1].split()[2][8:])
            loss = float(split_line[1][5:])
            lcc =  float(split_line[2][4:])
            srocc = float(split_line[3][6:])

            x.append(epoch + percent)
            y_loss.append(loss)
            y_lcc.append(lcc)
            y_srocc.append(srocc)

    return x, y_loss, y_lcc, y_srocc


if osp.isfile(log_path):
    print('Reading file: {}'.format(log_path))

    lines = read_file(log_path)[start:]

    #debug
    debug=0
    if debug:
        print('{}\n{}'.format(lines[0], lines[-1]))
        print(lines[43].split()[-2])

    train, train_loss, train_lcc, train_srocc = get_train_score(lines)
    test, test_loss, test_lcc, test_srocc = get_test_score(lines)

    plt.figure(1)

    plt.subplot(221)
    plt.plot(train, train_lcc, 'y', test, test_lcc, 'b')
    plt.grid(True)
    plt.ylim((0.8, 1.))
    plt.title('lcc')

    plt.subplot(222)
    plt.plot(train, train_srocc, 'y', test, test_srocc, 'b')
    plt.grid(True)
    plt.ylim((0.8, 1.))
    plt.title('srocc')

    plt.subplot(212)
    plt.plot(train, train_loss, 'y', test, test_loss, 'b')
    plt.grid(True)
    plt.ylim((0, 200))
    plt.title('loss')

    plt.show()

