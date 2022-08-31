# basic settings
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12.5,
         }
MARKERS = (['o', 's', 'v', "^", "h", "v", ">", "x", "d", "<", "|", "", "|", "_"])
# you may want to change the color map for different figures
COLOR_MAP = ('#B03A2E', '#2874A6', '#239B56', '#7D3C98', '#F1C40F', '#F5CBA7', '#82E0AA', '#AEB6BF', '#AA4499')
# you may want to change the patterns for different figures
PATTERNS = (["\\", "///", "o", "||", "\\\\", "\\\\", "//////", "//////", ".", "\\\\\\", "\\\\\\"])
LABEL_WEIGHT = 'bold'
LINE_COLORS = COLOR_MAP
LINE_WIDTH = 5.0
MARKER_SIZE = 1.0
MARKER_FREQUENCY = 1000


# Different proportions of positive tuples
lexicon_acc = [0.397,0.4793,0.544,0.659,0.7585,0.826,0.88]
lexicon_f1 = [0.568,0.545,0.557,0.656,0.747,0.833,0.936]
lda_acc = [0.496,0.595,0.6445,0.72242,0.729,0.7739,0.811]
lda_f1 = [0.639,0.661,0.6667,0.731,0.734,0.793,0.876]
clustering_acc = [0.511,0.498,0.510,0.5338,0.5424,0.512,0.482]
clustering_f1 = [0.674,0.581,0.541,0.613,0.5782,0.591,0.648]
senti_acc = [0.878,0.862,0.831,0.796,0.8232,0.877,0.981]
senti_nt_acc = [0.856,0.828,0.7862,0.793,0.7964,0.82,0.93]
senti_f1 = [0.927,0.865,0.827,0.832,0.845,0.850,0.99]
plt.figure(figsize=(8,3))
plt.plot(['0%','12.5%','25%','50%','75%','87.5%','100%'],lexicon_acc,marker = 'd',markersize =10,color = COLOR_MAP[0],linewidth = 3)
plt.plot(['0%','12.5%','25%','50%','75%','87.5%','100%'],lda_acc,marker = 'v',markersize =10,color = COLOR_MAP[1],linewidth = 3)
plt.plot(['0%','12.5%','25%','50%','75%','87.5%','100%'],clustering_acc,marker = '*',markersize =10,color = COLOR_MAP[2],linewidth = 3)
plt.plot(['0%','12.5%','25%','50%','75%','87.5%','100%'],senti_acc,marker = '>',markersize =10,color = COLOR_MAP[3],linewidth = 3)
plt.plot(['0%','12.5%','25%','50%','75%','87.5%','100%'],senti_nt_acc,marker = 'v',markersize =10,color = COLOR_MAP[4],linewidth = 3)
plt.xlabel('Different proportions of positive tuples',fontproperties = 'Times New Roman',size =22)
plt.ylabel('Accuracy',fontproperties = 'Times New Roman',size = 22)
# plt.ylim(0.4,1)
plt.grid()
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.legend(['Lexicon-based','LDA-based','Clustering','SentiStream','SentiStream w/o TTD'],prop=font1,bbox_to_anchor=(1,0.7))
#plt.legend(['Lexicon-based','LDA-based','Clustering','SentiStream'],prop=font1,bbox_to_anchor=(1.05,-0.27),ncol=4)
plt.savefig('./v_acc_distributions.pdf',bbox_inches = 'tight')
plt.show()


# Number of vocabulary per input tuple
t_lexicon_acc = [0.584,0.636,0.633,0.630,0.601]
t_lda_acc = [0.551,0.675,0.689,0.701,0.712]
t_clustering_acc = [0.524,0.516,0.510,0.538,0.529]
t_senti_acc = [0.641,0.778,0.775,0.781,0.782]
plt.figure(figsize=(8,2.5))
plt.plot(['<30','30-100','100-200','200-300','>300'],t_lexicon_acc,marker = 'd',markersize =10,color = COLOR_MAP[0],linewidth = 3)
plt.plot(['<30','30-100','100-200','200-300','>300'],t_lda_acc,marker = 'v',markersize =10,color = COLOR_MAP[1],linewidth = 3)
plt.plot(['<30','30-100','100-200','200-300','>300'],t_clustering_acc,marker = '*',markersize =10,color = COLOR_MAP[2],linewidth = 3)
plt.plot(['<30','30-100','100-200','200-300','>300'],t_senti_acc,marker = '>',markersize =10,color = COLOR_MAP[3],linewidth = 3)
plt.xlabel('Number of vocabulary per input tuple',fontproperties = 'Times New Roman',size =22)
plt.ylabel('Accuracy',fontproperties = 'Times New Roman',size = 22)
plt.ylim(0.4,0.9)
plt.grid()
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
plt.legend(['Lexicon-based','LDA-based','Clustering','SentiStream'],prop=font1,bbox_to_anchor=(1,0.7))
#plt.legend(['Lexicon-based','LDA-based','Clustering','SentiStream'],prop=font1,bbox_to_anchor=(1.02,-0.27),ncol=4)
plt.savefig('./v_acc_length.pdf',bbox_inches = 'tight')
plt.show()


# violin plotting: please first learn the format of violin.csv
# import seaborn as sns
# import pandas as pd
# data = pd.read_csv('violin.csv')
# plt.figure(figsize=(11,6))
# my_pal = {"Lexicon-based": COLOR_MAP[0], "LDA-based": COLOR_MAP[1], "Clustering": COLOR_MAP[2],  "SentiStream": COLOR_MAP[5]}
# #sns.set_style(style='white',rc={'patch.edgecolor':'yellow'})
# sns.violinplot(x = 'increm', y = 'acc1',hue='Legend',palette = my_pal,hue_order=['Lexicon-based','LDA-based','Clustering','SentiStream'],data =data, cut=2,saturation=3)
# #sns.violinplot(x = 'increm', y = 'acc2',data =data,)
# plt.ylim(0.4,0.9)
# plt.title(f"OSC Accuracy on Sentiment140 Dataset",size =20)
# plt.xlabel('Number of processed input tuples',fontproperties = 'Times New Roman',size =25)
# plt.ylabel('Accuracy',fontproperties = 'Times New Roman',size = 25)
# plt.xticks(fontproperties = 'Times New Roman', size = 20)
# plt.yticks(fontproperties = 'Times New Roman', size = 20)
# plt.legend(prop=font2,loc=2)
# plt.grid()
# plt.savefig('./v_acc_twitter.pdf')
# plt.show()


# bar plot (throughput for example)
# new throughputs and error bar should be tested
plt.figure(figsize =(8,3))
x = np.arange(2)
total_width, n = 0.9, 5     
width = total_width / n
x = x - (total_width - width) / 2

dataset = (u'Yelp Review',u'Sentiment140')
plt.xticks(x+0.24,dataset,fontproperties = 'Times New Roman', size = 20)
#plt.title("Throughput on two Datasets",size =20)
#plt.xlabel('Regenerated Sentiment140 Dataset',size =20)
plt.ylabel('Tpt. (tuples/second)',fontproperties = 'Times New Roman', size = 22)
plt.yticks(fontproperties = 'Times New Roman', size = 20)

plt.bar(x, throughput1,  width=width, label='label1',hatch =PATTERNS[0],color=COLOR_MAP[0])
plt.bar(x + width, throughput2, width=width, label='label2',hatch =PATTERNS[1],color=COLOR_MAP[1])
plt.bar(x + 2 * width, throughput3, width=width, label='label3',hatch =PATTERNS[2], color=COLOR_MAP[2])
#plt.bar(x + 3 * width, throughput4, width=width, label='label2',hatch =PATTERNS[3],color=COLOR_MAP[4])
plt.bar(x + 3 * width, throughput5, width=width, label='label2',hatch =PATTERNS[4],color=COLOR_MAP[5])

plt.legend(['Lexicon-based','LDA-based','clustering','SentiStream'],prop=font1,bbox_to_anchor=(1, 0.6))
plt.errorbar(x, throughput1,  yerr=264, fmt='.k',capsize=21)
plt.errorbar(x + width, throughput2,   yerr=312, fmt='.k',capsize=21)
plt.errorbar(x + 2 * width, throughput3,  yerr=355, fmt='.k',capsize=21)
#plt.errorbar(x + 3 * width, throughput4,  yerr=323, fmt='.k',capsize=16)
plt.errorbar(x + 3 * width, throughput5,  yerr=372, fmt='.k',capsize=21)
plt.savefig(f'./overall_throughput.pdf',bbox_inches = 'tight')




# LRU performance
dataset =['30,000','60,000','80,000','100,000']
y = time_to_test
y1 = time_to_test
y2 = acc_to_test
y3 = acc_to_test
fig,ax1= plt.subplots(figsize = (8,3.5))#figure()
x = np.arange(4)
total_width, n = 0.6, 2  
width = total_width / n
x = x - (total_width - width) / 2


ax1.grid(axis = 'y')
ax1.bar(x, y, width=width, label='label2',hatch =PATTERNS[1],color=COLOR_MAP[1])
ax1.bar(x + width, y1, width=width, label='label2',hatch =PATTERNS[3],color=COLOR_MAP[5])
ax1.set_ylim(100,700)
plt.xlabel('Number of vocabulary seen by the system',fontproperties = 'Times New Roman',size =20)
plt.ylabel('Model update cost (ms)',fontproperties = 'Times New Roman',size =20)
plt.xticks(x+0.15,dataset,fontproperties = 'Times New Roman', size = 20)
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)

ax2 = ax1.twinx()
ax2.plot(x,y2,c=COLOR_MAP[1],linewidth = 3,marker = 'd',markersize =5)
ax2.plot(x,y3,c=COLOR_MAP[5],linewidth = 3,marker = 'D',markersize =5)
ax2.set_ylim(0.4,1)
ax2.tick_params(labelsize = 15)
ax2.set_ylabel('Accuracy',font=font1,size =20)
ax2.legend(['SentiStream without LRU-pruning','SentiStream'],prop=font1,bbox_to_anchor=(0.8,1),ncol =2)
ax1.legend(['SentiStream without LRU-pruning','SentiStream'],prop=font1,bbox_to_anchor=(0.8,0.9),ncol =2)
ax1.errorbar(x, y,  yerr=39, fmt='.k',capsize=16)
ax1.errorbar(x + width, y1,   yerr=36, fmt='.k',capsize=16)

plt.savefig('./LRU_performance.pdf',bbox_inches = 'tight')

