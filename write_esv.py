from read_bible import *


def write_tokens(source, destination):
    with open(destination, 'w') as f:
        for book in source:
            for chapter in source[book]:
                for verse in source[book][chapter]:
                    # print(" ".join(source[book][chapter][verse][1:-1]))
                    f.write(" ".join(source[book][chapter][verse][1:-1]) + '\n')


if __name__ == '__main__':
    esv = read_esv('data/esv.txt', category='full')
    write_tokens(esv, 'data/esv_tokens.txt')
