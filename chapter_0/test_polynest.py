import numerics0_manalu as num
import numpy as np

coeff = [1, 2, 1, 3]
bspts = [2, 2, 2]

coeff2 = [4,1,3,5]

out = num.polynest(1,coeff)
out2 = num.polynest(3, coeff2)

outbs = num.polynest(1,coeff,bspts)
outbs2 = num.polynest(3,coeff2,bspts)
print(out)
print('The value should be 7')

print(out2)
print('The value should be 169')

print(outbs)
print('The value should be -3')

print(outbs2)
print('The value should be 13')
