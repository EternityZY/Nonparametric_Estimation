import sys
print (sys.argv)
if __name__=='__main__':
    print("Program name", sys.argv[0])
    for i in range(1, len(sys.argv)):
        print ("arg%d"%i, sys.argv[i])