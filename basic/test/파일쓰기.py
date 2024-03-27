root ='/Users/sungheui/PycharmProjects/basic/test'

import os

file = os.path.join(root,'check.txt' )

f = open(file, 'w')

for i in range(10):
    f.write(str(i)+'\n')
    f.flush()
    print()

f.close()



