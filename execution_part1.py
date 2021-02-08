import os
"""
to_run = "python test_change.py"
os.system(to_run)
"""
#Re Run All these
last_layer_feature = [4101, 2050, 1025, 512]
prob_feature = [0.7, 0.5]
for prob_num in prob_feature:
    for last_layer in last_layer_feature:
        first_num = 128
        second_num = 32
        multiplication = 2*first_num*second_num
        line_1 = "            nn.Conv2d(in_channels=1, out_channels=" + str(first_num) + ", kernel_size=5), \n" #For line 51
        line_2 = "            nn.Conv2d(in_channels="+ str(first_num) +", out_channels=" + str(second_num) + ", kernel_size=3), \n" #line 54
        line_3 = "            nn.Linear("+ str(multiplication) +", " + str(last_layer) + "), \n" #For line 61
        line_4 = "            nn.Linear("+ str(last_layer) + ", 10), \n" #For line 63
        line_5 = "            nn.Dropout(p=" + str(prob_num) + "), \n" #line 57
        with open('kuzu.py') as myFile:
            content = myFile.readlines()
        #print(content[51],content[54],content[61],content[63])

        with open('kuzu.py', 'w') as myFile:
            for index, line in enumerate(content):
                if index == 51:
                    newLine = line_1
                    myFile.write(newLine)
                elif index == 54:
                    newLine = line_2
                    myFile.write(newLine)
                elif index == 57:
                    newLine = line_5
                    myFile.write(newLine)
                elif index == 62:
                    newLine = line_3
                    myFile.write(newLine)
                elif index == 64:
                    newLine = line_4
                    myFile.write(newLine)
                else:
                    myFile.write(line)

        #Now we execute:

        file_name = "q3_output_" + str(first_num) + "_5_1_" + str(second_num) + "_3_1_" + str(last_layer) + "_" + str(prob_num)  +".txt"
        print(f"Executing {file_name}")
        to_run = "python kuzu_main.py --net conv >> " + file_name
        os.system(to_run)