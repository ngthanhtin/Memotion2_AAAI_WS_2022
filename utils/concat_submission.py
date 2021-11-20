root_path = './results/' + 'onlyimage/'
f1 = open(root_path + "/sentiment_result.txt", "r")
lines1 = f1.readlines()
f2 = open(root_path + "/emotion_result.txt", "r")
lines2 = f2.readlines()
f3 = open(root_path + "/intensity_result.txt", "r")
lines3 = f3.readlines()

f = open('./results/answer.txt', 'w')
for line1, line2, line3 in zip(lines1, lines2, lines3):
    f.write("{}_{}_{}\n".format(line1[:-1], line2[:-1], line3[:-1]))






