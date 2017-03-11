import matplotlib.pyplot as plt

#file_name = '1_100'
#file_name = '2_75'
#file_name = '3_50'
#file_name = '4_25'

# file_name = 'd_100.txt'

line_width = 1.5
# gap = 5

# epoch = [];train = [];tune = [];test = []

# with open(file_name) as f:
#     for line in f:
#         line = line.strip('\n')

#         if len(line) == 0 or line.startswith('#') or ',' not in line: continue

#         lst = line.split(',')
        
#         try:
#             a1 = int(lst[0])
#             a2 = 100 * float(lst[1])
#             a3 = 100 * float(lst[2])
#             a4 = 100 * float(lst[3])

#             epoch.append(a1)
#             train.append(a2)
#             tune.append(a3)
#             test.append(a4)
#         except:
            # pass
        
# plt.plot(epoch, train, epoch, tune, epoch, test, label='aa')
line1, = plt.plot([0.25, 0.5 , 0.75, 1.0], [0.73, 0.74, 0.71, 0.702], linewidth=line_width)
line2, = plt.plot([0.25, 0.5 , 0.75, 1.0],  [0.65, 0.60, 0.59, 0.58],linewidth=line_width)
# line3, = plt.plot(epoch[::gap], test[::gap], linewidth=line_width)
plt.legend((line1, line2), 
           ('Without dropout', 'With dropout'), loc=4)
plt.ylim([0.4, 1])
plt.xlabel('Precentage of training examples')
plt.ylabel('Accuracy (%)')
plt.show()

