import os
import inspect
import neat

print("Run this script from the doc/ directory of the repository")
funcs = inspect.getmembers(neat, inspect.isfunction)

for n, f in funcs:
    # print(n + ": "+str(f))
    cmd = "find . -name *\." + n + ".rst -print"
    # print(cmd)
    result = os.popen(cmd).read()
    # print(result)

    if len(result) == 0:
        print("Missing file from docs:  ", n)

print("Done finding functions that are missing from the docs")
