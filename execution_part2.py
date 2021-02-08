import os
"""
to_run = "python test_change.py"
os.system(to_run)
"""

"""
for _ in hid_num_list:
    file_name = "part_2_q1_output_" + str(_) + ".txt"
    print(f"Executing {file_name}")
    to_run = "python spiral_main.py --net polar --hid " + str(_) + " >> " + file_name
    os.system(to_run)

for _ in range(0,10):
    file_name = "part_2_q1_output_" + str(7) + "_run_" + str(_) + ".txt"
    print(f"Executing {file_name}")
    to_run = "python spiral_main.py --net polar --hid " + str(7) + " >> " + file_name
    os.system(to_run)

for _ in range(0,10):
    file_name = "part_2_q1_output_" + str(8) + "_run_" + str(_) + ".txt"
    print(f"Executing {file_name}")
    to_run = "python spiral_main.py --net polar --hid " + str(6) + " >> " + file_name
    os.system(to_run)

#run ntest for number of nodes = 8 with 0.2 weight - the ones with epoch < 200000
for _ in range(0,10):
    #Now we execute:
    file_name = "part_2_q2_output_" + str(8) + "_" + str(0.2) + ".txt"
    print(f"Executing {file_name}")
    to_run = "python spiral_main.py --net raw --hid " + str(8) + " --init " + str(0.2) + " >> " + file_name
    os.system(to_run)
"""
#run ntest for number of nodes = 8 with 0.2 weight - the ones with epoch < 200000
for _ in range(0,10):
    #Now we execut:
    file_name = "part_2_q2_output_" + str(15) + "_" + str(0.1) + "_" + str(_) +".txt"
    print(f"Executing {file_name}")
    to_run = "python spiral_main.py --net raw --hid " + str(15) + " --init " + str(0.1) + " >> " + file_name
    os.system(to_run)
for _ in range(0,10):
    #Now we execut:
    file_name = "part_2_q2_output_" + str(15) + "_" + str(0.2) + "_" + str(_) +".txt"
    print(f"Executing {file_name}")
    to_run = "python spiral_main.py --net raw --hid " + str(15) + " --init " + str(0.2) + " >> " + file_name
    os.system(to_run)
for _ in range(0,10):
    #Now we execut:
    file_name = "part_2_q2_output_" + str(15) + "_" + str(0.3) + "_" + str(_) +".txt"
    print(f"Executing {file_name}")
    to_run = "python spiral_main.py --net raw --hid " + str(15) + " --init " + str(0.3) + " >> " + file_name
    os.system(to_run)
for _ in range(0,10):
    #Now we execut:
    file_name = "part_2_q2_output_" + str(15) + "_" + str(0.4) + "_" + str(_) +".txt"
    print(f"Executing {file_name}")
    to_run = "python spiral_main.py --net raw --hid " + str(15) + " --init " + str(0.4) + " >> " + file_name
    os.system(to_run)
for _ in range(0,10):
    #Now we execut:
    file_name = "part_2_q2_output_" + str(10) + "_" + str(0.4) + "_" + str(_) +".txt"
    print(f"Executing {file_name}")
    to_run = "python spiral_main.py --net raw --hid " + str(10) + " --init " + str(0.4) + " >> " + file_name
    os.system(to_run)
"""
#Since there is no promising result we try and increase hidden num
hid_num_list = [5,6,7,8,9,10,15, 20, 25, 30]
initial_weights = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]

for hid_num in hid_num_list:
    for weight in initial_weights:
        #Now we execute:
        file_name = "part_2_q2_output_" + str(hid_num) + "_" + str(weight) + "_new.txt"
        print(f"Executing {file_name}")
        to_run = "python spiral_main.py --net raw --hid " + str(hid_num) + " --init " + str(weight) + " >> " + file_name
        os.system(to_run)
"""