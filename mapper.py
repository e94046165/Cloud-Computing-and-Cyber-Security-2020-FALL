import sys
import re
line_count = time_count = 0
for line in sys.stdin:
    #print(line)
    line = line.strip()
    line_count += 1
    # print(line)
    # parse month in the log
    time_pattern = re.compile("\[.*\]")
    # month_pattern = re.compile("[A-Za-z]+")
    # day_pattern = re.compile("\d+\/[A-Za-z]+\/\d+")
    hour_pattern = re.compile("\d+\/[A-Za-z]+\/\d+:\d+")
    time = time_pattern.findall(line)
    # print(time)
    hour = hour_pattern.findall(str(time))
    for i in hour:
        print("%s\t%s" % (i, 1))
        # time_count+=1
# print(line_count, time_count)


