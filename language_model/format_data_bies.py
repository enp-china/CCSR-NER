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

def fileToBIES(inputFile:str,outputFile:str):
    out = open(outputFile, "w", encoding="utf-8")
    with open(inputFile, "r", encoding="utf-8") as fp:
        line = fp.readline()
        while line:
            out.write(" ".join((makeFormatGold(line))+"\n"))
            line = fp.readline()
    out.close()


import sys

if __name__ == "__main__":
    fileToBIES(sys.argv[1],sys.argv[2])

