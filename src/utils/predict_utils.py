from progressbar import ProgressBar
import folia.main as folia
from pathlib import Path
import glob
import subprocess
import time
from trainer_utils import makeFormatGold

def uctoTokenizer(input_text: str, output_text: str):
    from ucto import Tokenizer

    """
    A function to create a folia file from a text file

    :param input_text: path to text file
    :param output_text: name for the folia file

    """
    configurationfile = "../ucto_config/tokconfig_eng_ch"

    tokenizer = Tokenizer(configurationfile, foliaoutput=True)

    folia_file_P = Path(input_text)
    isFolder = folia_file_P.is_dir()
    if isFolder:
        # files = [f for f in glob.glob(folia_file + "**/*.xml", recursive=True)]
        files = [f for f in glob.glob(input_text + "**/*.*", recursive=True)]
        pbar = ProgressBar()

        path = Path(output_text)
        if not path.exists():
            path.mkdir()

        for f in pbar(files):
            path_out = f.split("/")
            name_out = path / path_out[len(path_out) - 1]
            out = str(name_out).replace(".txt", ".folia.xml")
            tokenizer.tokenize(f, str(out))
    else:
        tokenizer.tokenize(input_text, output_text)


def formatToAnnotation(folia_file: str):
    doc = folia.Document(file=folia_file)
    doc.declare(
        folia.EntitiesLayer,
        "https://raw.githubusercontent.com/BaptisteBlouin/ttl/master/namedentities.ontonotesset.ttl",
    )
    doc.save(folia_file)


def textToConll(input:str,output:str,segmentor:str,segmentor_path:str):

    proc = subprocess.Popen([segmentor_path, segmentor],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    time.sleep(15)
    out = open(output,"w",encoding="utf8")

    with open(input,"r",encoding="utf-8") as fp :
        for line in fp:
            if line != "\n" or line != "":
                line_space = line.replace(" ", "").replace("\n","")+"\n "

                proc.stdin.write(line_space.encode())
                proc.stdin.flush()
                segmentation = proc.stdout.readline().decode()

                segmentation = makeFormatGold(segmentation.replace("\n", ""))

                for token,idx in zip(segmentation,range(len(segmentation))):
                    if token[-1] != "S" and token[-1] != "E":
                        word = str(idx+1) +"\t" +token+"\t_\t_\t_\t_\t_\t_\t_\t_\t"+"   SpaceAfter=No"+"\n"
                    else:
                        word = str(idx+1) +"\t"+  token + "\t_\t_\t_\t_\t_\t_\t_\t_\t" + "SpaceAfter=Yes" + "\n"

                    out.write(word)
                out.write("\n")
    out.close()

    proc.stdin.close()
    proc.terminate()
    proc.wait(timeout=0.2)


def evaluate2bratFile(prediction,test):
    tp = {}
    fp = {}
    tn = {}
    fn = {}

    pred_tab = {}
    test_tab = {}

    with open(prediction,"r",encoding="utf-8") as file :
        for line in file:
            if line[0] != "#":
                splt = line.split()
                types = splt[1]
                start = splt[2]
                stop = splt[3]
                text = splt[4]

                try :
                    pred_tab[types].append((start,stop,text))
                except:
                    pred_tab.update({types:[(start,stop,text)]})

    with open(test,"r",encoding="utf-8") as file :
        for line in file:
            if line[0] != '#':
                splt = line.split()
                types = splt[1]
                start = splt[2]
                stop = splt[3]
                text = splt[4]

                try :
                    test_tab[types].append((start,stop,text))
                except:
                    test_tab.update({types:[(start,stop,text)]})


    for types in pred_tab:
        tp.update({types:0})
        fp.update({types: 0})
        tn.update({types: 0})
        fn.update({types: 0})

        for pred in pred_tab[types]:
            if pred in test_tab[types]:
                tp[types]+=1
            else:
                fp[types]+=1

    for types in test_tab:
        for gold in test_tab[types]:
            if gold not in pred_tab[types]:
                fn[types]+=1
            else:
                tn[types]+=1

    prec = {}
    recall = {}
    fscore = {}
    micro ={}
    for types in fp:
        prec.update({types:0})
        if tp[types] + fp[types] >0:
            prec[types] = round(tp[types]/(tp[types]+fp[types]),4)

        recall.update({types:0})
        if tp[types] + fn[types] > 0:
            recall[types] = round(tp[types]/(tp[types]+fn[types]),4)

        fscore.update({types:0})
        if prec[types] + recall[types]> 0:
            fscore[types] = round(2*(prec[types]*recall[types])/(prec[types]+recall[types]),4)

        try:
            micro['tp']+= tp[types]
            micro['fp'] += fp[types]
            micro['tn'] += tn[types]
            micro['fn'] += fp[types]
        except:
            micro.update({'tp':tp[types]})
            micro.update({'fp': fp[types]})
            micro.update({'tn': tn[types]})
            micro.update({'fn': fn[types]})


    prec_micro = round(micro["tp"]/(micro["tp"]+micro["fp"]),4)
    recall_micro = round(micro["tp"]/(micro["tp"]+micro["fn"]),4)
    if (prec_micro+recall_micro) > 0 :
        f_score_micro = round(2*(prec_micro*recall_micro)/(prec_micro+recall_micro),4)
    else:
        f_score_micro = 0

    macro_f_score = [fscore[types] for types in fscore]
    macro_f_score = sum(macro_f_score) / len(macro_f_score)

    for types in fscore:
        print(types,"\t tp: ",tp[types]," tn: ",tn[types]," fp: ",fp[types]," fn: ",fn[types]," precision: ",prec[types],
              " recall: ",recall[types]," fscore: ",fscore[types])

    print("micro : precision: ",prec_micro, " recall: ",recall_micro," fscore : ",f_score_micro)
    print("macro fscore: ",macro_f_score)

if __name__ == "__main__":
    import sys

    evaluate2bratFile(sys.argv[1],sys.argv[2])