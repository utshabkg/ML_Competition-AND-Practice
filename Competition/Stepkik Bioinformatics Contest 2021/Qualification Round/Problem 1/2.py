with open('2.txt', 'r') as file:
    input_lines = [line.strip() for line in file]
# print(input_lines)

a = []
a_file = open("output2.txt", "w")
for i in range(1, len(input_lines)):
    if ' ' in input_lines[i]:
        n = int(input_lines[i].split(' ')[0])
        l = int(input_lines[i].split(' ')[1])
        # print(n, l)
        a = []
    else:
        a.append(input_lines[i])
        # print(a)
        if len(a) == n:
            comb = ['0'*n]*l
            # print(comb)
            for j in range(l):
                temp = list(comb[j])
                
                for k in range(n):
                    temp[k] = a[k][j]
                comb[j] = "".join(temp)
            # print(comb)
            print(len(set(comb)), file=a_file)

            c, counter = 1, []
            for j in range(l):
                if comb[j] not in comb[: j]:
                    print(c, end=' ', file=a_file)
                    counter.append(c)
                    c += 1
                else:
                    for k in range(len(comb[: j])):
                        if comb[j]==comb[k]:
                            # print('jk', j, k, comb[j], comb[k], comb[: j])
                            print(counter[k], end=' ', file=a_file)
                            counter.append(counter[k])
                            break
                # print(counter)
            print(file=a_file)
            print(len(counter))
        
a_file.close()

# a_file = open("output1.txt", "w")
    
#     print(file=a_file)
    
# a_file.close()     