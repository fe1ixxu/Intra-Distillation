import matplotlib.pyplot as plt
import collections
import matplotlib.pyplot
import torch

def get_input(i):
    return "dynamic/outputs-temp-" + str(i) + "/panx/xlm-roberta-large-LR2e-5-epoch10-MaxLen128/test_results.txt"
    # return "dynamic/outputs-temp-" + str(i) + "/udpos/xlm-roberta-large-LR2e-5-epoch10-MaxLen128/test_results.txt"
    # return "dynamic/outputs-temp-" + str(i) + "/squad/xlm-roberta-large_LR3e-5_EPOCH3.0_maxlen384_batchsize4_gradacc8/predictions/xquad/test_results.txt" 



data = collections.defaultdict(list)
for i in range(2, 25, 2):
    input = get_input(i)
    lang = None
    with open(input) as f:
        line  = f.readline()
        while line:
            line = line.strip().split()
            if "language=" in line[0]:
                lang = line[0][-2:]
            if len(line) == 3:
                types, place_holder, score = line
                if types == "f1" and lang != None:
                    data[lang].append(float(score))
                    lang = None
                elif types == "f1" and lang == None:
                    data[place_holder].append(float(score))

            line = f.readline()


src = "en"
data_en = torch.tensor(data[src])
res_dis = []
res_sim = []
for lang, nums in data.items():
    if lang != src:
        data_tgt = torch.tensor(nums)
        vx = data_en - torch.mean(data_en)
        vy = data_tgt - torch.mean(data_tgt)
        pearson = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        if pearson < 0.3:
            res_dis.append((lang, pearson))
        else:
            res_sim.append((lang, pearson))
print(res_sim)
print(res_dis)


# plt.figure()
# fig = matplotlib.pyplot.gcf()
# fig.set_size_inches(18.5, 30.5)
# i = 1
# NUM=len(data)
# xaxis = list(range(2, 25,2))
# for lang, nums in data.items():
#     plt.subplot(NUM,1, i)
#     plt.plot(xaxis, nums, label=lang)
#     plt.subplot_tool()
#     plt.legend(loc="upper right")
#     i+=1
#     if i > NUM:
#         break

# # plt.show()
# fig.savefig('panx.png', dpi=100)
# # plt.savefig("temp.png", dpi=100)

