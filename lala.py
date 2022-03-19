def LatinPrinter(sol):
    for index, item in enumerate(sol, start=1):
        print(item, end=' ' if index % 9 else '\n')

sol = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,1,2,3,4,5,6,7,8,9]
# LatinPrinter(sol)

# counter = 0
# while counter < 20:
#     print("currently on" + str(counter) + " ", end='\r')
#     counter +=1
print('test')

a = [1,2,3]
a.append([5,2,3])
print(a)