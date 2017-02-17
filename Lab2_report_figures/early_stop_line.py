import matplotlib.pyplot as plt

file_name = '2_rate'
#file_name = '1_nothing'
#file_name = '3_dropout'
#file_name = '4_dropout_mem'
line_width = 1.5
gap = 1

epoch = [];train = [];tune = [];test = []

with open(file_name) as f:
    for line in f:
        line = line.strip('\n')

        if len(line) == 0 or line.startswith('#'): continue

        lst = line.split(',')
        
        try:
            a1 = int(lst[0])
            a2 = 100 * float(lst[1])
            a3 = 100 * float(lst[2])
            a4 = 100 * float(lst[3])

            epoch.append(a1)
            train.append(a2)
            tune.append(a3)
            test.append(a4)
        except:
            pass
        
# plt.plot(epoch, train, epoch, tune, epoch, test, label='aa')
line1, = plt.plot(epoch[::gap], train[::gap], linewidth=line_width)
line2, = plt.plot(epoch[::gap], tune[::gap], linewidth=line_width)
line3, = plt.plot(epoch[::gap], test[::gap], linewidth=line_width)
plt.legend((line1, line2, line3), 
           ('Training set accuracy', 'Tunning set accuracy', 'Testing set accuracy'), loc=4)
plt.ylim([0, 100])
plt.xlabel('Epoch (times)')
plt.ylabel('Accuracy (%)')
plt.show()

