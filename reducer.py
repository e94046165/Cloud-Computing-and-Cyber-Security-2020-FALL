import sys

current = nextline = None
count = 0
total_count = line_count = 0
for line in sys.stdin:
    line_count += 1
    line = line.strip()
    timestamp, _ = line.split('\t')
    if current != timestamp:
        if current != None:
            print("%s\t%s" %(current, count))
        current = timestamp
        total_count += count
        count = 1
    else:
        count += 1
print("%s\t%s" %(current, count))
total_count += count
# print(total_count)

