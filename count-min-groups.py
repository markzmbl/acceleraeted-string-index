import mmap
import re

count_mins = 0
count_total = 0
average_forward_step = 0
average_backward_step = 0
max_n_avg = 0
max_n_count = 0
max_n = 0
max_m_avg = 0
max_m = 0
line_count = 200000000
with open("output.txt", "r") as f:
    for line in f:
        line_count += 1
        if "[GROUP]" in line:
            count_total += 1
        if "\tm:\t3584\n" in line:
            count_mins += 1
        if "\tfsteps:\t" in line:
            average_forward_step += int(line[line.find(":") + 2 : line.find("\n")])
        if "\tbsteps:\t" in line:
            average_backward_step += int(line[line.find(":") + 2 : line.find("\n")])
        if "\tn:\t" in line:
            n = int(line[line.find(":") + 2 : line.find("\n")])
            max_n = max(max_n, n)
            max_n_avg += n
        if "\tm:\t" in line:
            m = int(line[line.find(":") + 2 : line.find("\n")])
            max_m = max(max_m, m)
            max_m_avg += m
with open("output.txt", "r") as f:
    for line in f:
        if "\tn:\t" in line:
            if int(line[line.find(":") + 2 : line.find("\n")]) == max_n:
                max_n_count += 1


print(count_mins, " / ", count_total, " -> ", round(count_mins / count_total, 2))
print(count_mins * 3584, " / ", line_count, " -> ", round(count_mins * 3584 / line_count, 2))
print(max_m, " -> ", round(max_m_avg / count_total, 2))
print(max_n, " : ", max_n_count, " -> ", round(max_n_avg / count_total, 2))
print(round(average_forward_step / count_total, 2))
print(round(average_backward_step / count_total, 2))