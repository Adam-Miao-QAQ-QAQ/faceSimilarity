from descutil import *
import os


def infile():
    fstr = input("Filename [example.jpg, e to stop]:")
    if fstr.strip() == '':
        return 'input/example.jpg'
    if fstr.strip() == 'e':
        return False
    return os.path.join('input', fstr)


if __name__ == '__main__':
    fstrs = []
    while True:
        tmp = infile()
        if tmp:
            fstrs.append(tmp)
        else:
            break

    if len(fstrs) == 0:
        raise SystemExit('No files')

    def_desc = gen_descriptor(cv2.imread(fstrs[0]), fn=os.path.split(fstrs[0])[-1])

    with open('result.txt', 'w') as f:
        for i, fstr in enumerate(fstrs[1:]):
            desc = gen_descriptor(cv2.imread(fstr), fn=os.path.split(fstr)[-1])
            distance = euclidean(desc, def_desc)
            print(i + 1, distance)
            f.write(f'The distance between {os.path.split(fstrs[0])[-1]} and {os.path.split(fstr)[-1]} is {distance}.\n')
