from inspect import getframeinfo, stack
import os

def DEBUG_PRINT(*arg):
    caller = getframeinfo(stack()[1][0])
    filename = os.path.basename(caller.filename)
    print("[%s][%s]" % (filename, caller.function), end =" ")
    for message in arg:
        print("%s" % (message), end ="") # python3 syntax print
    print("")
