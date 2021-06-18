import math
import sys

# ar1[0..m-1] and ar2[0..n-1] are two
# given sorted arrays and x is given
# number. This function prints the pair
# from both arrays such that the sum
# of the pair is closest to x.
def printClosest(ar1, ar2, m, n, x):

	# Initialize the diff between
	# pair sum and x.
	diff=sys.maxsize

	# res_l and res_r are result
	# indexes from ar1[] and ar2[]
	# respectively. Start from left
	# side of ar1[] and right side of ar2[]
	l = 0
	r = n-1
	while(l < m and r >= 0):
	
        # If this pair is closer to x than
        # the previously found closest,
        # then update res_l, res_r and diff
		if math.fabs(ar1[l] + ar2[r] - x) < diff:
			res_l = l
			res_r = r
			diff = math.fabs(ar1[l] + ar2[r] - x)
	
	# If sum of this pair is more than x,
	# move to smaller side
		if ar1[l] + ar2[r] > x:
			r=r-1
		else: # move to the greater side
			l=l+1

	# Print the result
	print(res_l+1, res_r+1)#, file=a_file)


with open('sample input.txt', 'r') as file:
    input_lines = [line.strip() for line in file]
# print(input_lines)

a_file = open("output2.3.txt", "w")
for t in range(1, len(input_lines), 4):
    M, K, N = input_lines[t].split(' ')
    M, K, N = int(M), int(K), int(N)
    m, k = [], []
    for i in range(M):
        m.append(float(input_lines[t+1].split(' ')[i]))
    
    for i in range(K):
        k.append(float(input_lines[t+2].split(' ')[i]))
    # print(M, K, N, m, k, s)
    m, k = sorted(m), sorted(k)

    for x in range(N):
        s = float(input_lines[t+3].split(' ')[x])
        # d = 10**18; m1, m2 = 0, 0
        # for i in range(len(sum)):
        #     if d > math.fabs(s - float(sum[i][:-2])):
        #         d = math.fabs(s - float(sum[i][:-2]))
        #         m1, m2 = int(sum[i][-2]), int(sum[i][-1])
        # print(m1+1, m2+1, file=a_file)
        # print(m1+1, m2+1)
        # print(s)
        printClosest(sorted(m), sorted(k), M, K, s)

a_file.close()