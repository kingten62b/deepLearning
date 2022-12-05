input = 32
kernel_size = 5
stride = 1
padding = 2
output = (input - kernel_size + 2 * padding)/stride +1
print (output)
# print (5*5*3*16 + 5*5*16*32 + 5*5*32*32)
print (5*5*32*32)