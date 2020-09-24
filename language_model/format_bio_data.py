import typing

def addSegToBioUnsup(input:str,model:str,output:str,char_ind:int=0,tag_ind:int=1):
    from eleve import Segmenter
    from eleve.memory import  CSVStorage


    lm = CSVStorage(model)

    s = Segmenter(lm)

    all = []
    tags = []


    sub = []
    tag = []
    with open(input, "r", encoding="utf-8") as fp:
        for line in fp :
            if line=="\n" or line =="":
                all.append(sub)
                tags.append(tag)
                sub=[]
                tag = []
            else:
                line_split=line.replace("\n","").split()
                c = line_split[char_ind]
                t = line_split[tag_ind]
                sub.append(c)
                tag.append(t)

    out = open(output,"w",encoding='utf-8')
    for i in range(len(all)):
        if len(all[i]) > 0:
            sentence = " ".join(all[i])
            segmentation = s.segmentSentenceTIWBIES(sentence).split(" ")

            for j in range(len(segmentation)):
                line_out = segmentation[j] +" "+tags[i][j]+"\n"
                out.write(line_out)
            out.write('\n')
    out.close()


def makeFormatGold(sentence: str):
    split_str = sentence.split()
    list_tok = []
    for token in split_str:
        if len(token) == 1:
            tok = token+"-_-S"
            list_tok.append(tok)
        elif len(token) > 1:
            tok = token[0] + "-B"
            list_tok.append(tok)
            for i in range(1,len(token)-1):
                tok = token[i] +"-I"
                list_tok.append(tok)
            tok = token[-1] + "-E"
            list_tok.append(tok)
    return list_tok

def addSegToBio(input:str,output:str,modelSup:str,zpar_path:str,char_ind:int=0,tag_ind:int=1):
    proc = subprocess.Popen([zpar_path, modelSup],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    time.sleep(10)
    all = []
    tags = []


    sub = []
    tag = []
    with open(input, "r", encoding="utf-8") as fp:
        for line in fp :
            if line=="\n" or line =="":
                all.append(sub)
                tags.append(tag)
                sub=[]
                tag = []

            else:
                line_split=line.replace("\n","").split()
                c = line_split[char_ind]
                t = line_split[tag_ind]
                sub.append(c)
                tag.append(t)

    out = open(output,"w",encoding='utf-8')
    for i in range(len(all)):
        sentence = "".join(all[i])+"\n"
        proc.stdin.write(sentence.encode())
        proc.stdin.flush()
        segmentation = proc.stdout.readline().decode()

        segmentation = makeFormatGold(segmentation.replace("\n", ""))

        for j in range(len(segmentation)):
            line_out = segmentation[j] +" "+tags[i][j]+"\n"
            out.write(line_out)
        out.write('\n')
    out.close()
    proc.stdin.close()
    proc.terminate()
    proc.wait(timeout=0.2)


import sys
import argparse

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-U', '--unsup',
                        help='Do you want to use a unsupervised model or not.',
                        default=False,
                        required=True)
    parser.add_argument('-I', '--input',
                        help='Path to the input file',
                        required=True)
    parser.add_argument('-O', '--output',
                        help='Path to the output file',
                        required=True)
    parser.add_argument('-M', '--model',
                        help='Path to the model',
                        required=True)
    parser.add_argument('-Z', '--zpar',
                        help='Path to the zpar segmenter',
                        required=False)
    parser.add_argument('-C', '--char_ind',
                        help='index of the char in your BIO file',
                        default=0,
                        required=False)
    parser.add_argument('-T', '--tag_ind',
                        help='index of the tag in your BIO file',
                        default=1,
                        required=False)

    arg = parser.parse_args(args)
    return arg

if __name__ == "__main__":
    arg = main()
    if arg.unsup == True :
        addSegToBioUnsup(arg.input, arg.model, arg.output, arg.char_ind, arg.tag_ind)
    #else:
    #    addSegToBio(input:str,output:str,modelSup:str,zpar_path:str,char_ind:int=0,tag_ind:int=1)
