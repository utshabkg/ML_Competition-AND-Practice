import math

with open('1.txt', 'r') as file:
    input_lines = [line.strip() for line in file]
# print(input_lines)

a_file = open("output2.1.txt", "w")
for t in range(1, len(input_lines), 4):
    M, K, N = input_lines[t].split(' ')
    M, K, N = int(M), int(K), int(N)
    m, k, s = [], [], []
    for i in range(M):
        m.append(float(input_lines[t+1].split(' ')[i]))
    for i in range(K):
        k.append(float(input_lines[t+2].split(' ')[i]))
    for i in range(N):
        s.append(float(input_lines[t+3].split(' ')[i]))
    # print(M, K, N, m, k, s)

    for x in range(N):
        d = 10**18; m1, m2 = 0, 0
        
        for i in range(M):
            for j in range(K):
                # sum.append(m[i] + k[j])
                if d > math.fabs(s[x] - (m[i]+k[j])):
                    d = math.fabs(s[x] - (m[i]+k[j]))
                    m1, m2 = i+1, j+1
        print(m1, m2, file=a_file)

a_file.close()