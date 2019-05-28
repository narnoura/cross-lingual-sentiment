#Author: Mohammad

import os,sys,codecs
from collections import defaultdict

g_lab = [line.split('\t')[2].strip() for line in codecs.open(os.path.abspath(sys.argv[1]),'r')]
p_lab = [line.split('\t')[1].strip() for line in codecs.open(os.path.abspath(sys.argv[2]),'r')]
verbose = True if len(sys.argv)>3 and sys.argv[3]=='v' else False
if verbose:
    g_lines = [line.split('\t')[0].strip() for line in codecs.open(os.path.abspath(sys.argv[1]),'r')]


correct = len([1 for i in range(len(g_lab)) if g_lab[i]==p_lab[i]])
print (round(100*float(correct)/len(g_lab),1))


tp = defaultdict(float)
fp = defaultdict(float)
fn = defaultdict(float)
labels = set()
for i in range(len(g_lab)):
    labels.add(g_lab[i])
    labels.add(p_lab[i])
    if p_lab[i]==g_lab[i]:
        tp[g_lab[i]]+=1
    else:
        fn[g_lab[i]]+=1
        fp[p_lab[i]]+=1
        if verbose: print (g_lines[i],g_lab[i],p_lab[i])

avg_recall=0
f_pn =0
f_macro=0

for lab in labels:
    p,r = 100.0*tp[lab]/(tp[lab]+fp[lab]) if (tp[lab]+fp[lab])>0 else 0 , 100.0*tp[lab]/(tp[lab]+fn[lab]) if (tp[lab]+fn[lab])>0 else 0
    f = 2*p*r /(p+r) if (p+r)>0 else 0
    avg_recall += r
    f_macro += f
    if lab == "positive" or lab == "negative":
        f_pn += f
    print (lab,round(p,1),round(r,1),round(f,1))


avg_recall = avg_recall/len(labels)
f_pn = f_pn/2
f_macro = f_macro/len(labels)


print ("MacroAverage recall:", avg_recall)
print ("F-PN: ", f_pn)
print ("F-Macro: ", f_macro)
