import numpy as np
import os
import itertools

pwd = os.path.join(os.path.abspath('../'), 'Datasets')

def samechoice(lists, path, allsame, f, num):
    length = len(lists)
    count = 0
    for i in range(100000):
        dir = lists[i%length]
        dir_path = os.path.join(path, dir)
        files = [os.path.join(dir_path, dp) for dp in os.listdir(dir_path)]
        if len(files)==1:
            continue
        choice = tuple(np.random.choice(files, 2, replace=False))
        cho = (choice[1], choice[0])
        if (choice in allsame) or (cho in allsame):
            continue
        else:
            allsame.add(choice)
            f.write(choice[0]+' '+choice[1]+' 1\n')
            count += 1
        if count==num:
            return

def notsamechoice(lists, path, allnotsame, f, num):
    count = 0
    for i in range(100000):
        for i in itertools.combinations(lists, 2):
            dir_pathone = os.path.join(path, i[0])
            dir_pathtwo = os.path.join(path, i[1])
            filesone = [os.path.join(dir_pathone, dp) for dp in os.listdir(dir_pathone)]
            filestwo = [os.path.join(dir_pathtwo, dp) for dp in os.listdir(dir_pathtwo)]
            choiceone = np.random.choice(filesone, 1)[0]
            choicetwo = np.random.choice(filestwo, 1)[0]
            choice = (choiceone, choicetwo)
            cho = (choicetwo, choiceone)
            if (choice in allnotsame) or (cho in allnotsame):
                continue
            else:
                allnotsame.add(choice)
                f.write(choice[0] + ' ' + choice[1] + ' 0\n')
                count += 1
            if count == num:
                return

path = os.path.join(pwd, 'lfw_funneled', '')
lists = [os.path.join(path, i) for i in os.listdir(path)]
allsame = set()
allnotsame = set()
f = open(os.path.join(pwd, 'testpairs.txt'), 'w')
Kfold = 10
num = 300
for i in range(Kfold):
    samechoice(lists, path, allsame, f, num)
    notsamechoice(lists, path, allnotsame, f, num)
f.close()



