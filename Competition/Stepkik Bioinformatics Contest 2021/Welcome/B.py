with open('inputB.txt', 'r') as file:
    input_lines = [line.strip() for line in file]

a_file = open("outputB.txt", "w")
for j in range(1, 2*int(input_lines[0]) + 1, 2):
    s = input_lines[j]
    t = input_lines[j+1]
    ls, lt = len(s), len(t)
    # print(s, t, ls, lt)

    for i in range(ls):
        if t==s[i:(lt+i)]:
            print(i+1, end=' ', file=a_file)
        # print(t, len(s[i:(lt+i)]))
    
    print(file=a_file)
    
a_file.close()