from typing import List
from pathlib import Path
from multiprocessing import Pool
import itertools as it
import gc


from eleve import CMemoryStorage as Storage
from eleve import CLeveldbStorage as SavedStorage
from eleve.memory import  CSVStorage
from eleve import Segmenter
from eleve.preprocessing import chinese


def preproc(l: str) -> List[str]:
    chunks = chinese.tokenize_by_unicode_category(l)
    return [cjk for cjk in chinese.filter_cjk(chunks)]

def train_batch(corpus: List[str], save = None):
    if save is None:
        storage = Storage() #"/tmp/level")
    else:
        storage = SavedStorage(save)
    pool = Pool()
    buf = []
    voc = set()
    counter = 0
    for line in corpus:
        buf.append(line)
        counter += 1
        if counter % 10000 == 0:
            print("read", counter)
            lines = pool.map(preproc, buf)
            for chunks in lines:
                for chunk in chunks:
                    storage.add_sentence(list(chunk))
                    for i in range(len(chunk)):
                        for j in range(i+1,min(i+6,len(chunk))):
                            voc.add(chunk[i:j])
            buf = []
    if len(buf) > 0:
        print("read", counter)
        lines = pool.map(preproc, buf)
        for chunks in lines:
            for chunk in chunks:
                storage.add_sentence(list(chunk))
                for i in range(len(chunk)):
                    for j in range(i + 1, min(i + 6, len(chunk))):
                        voc.add(chunk[i:j])
    pool.terminate()
    pool.close()
    return storage, voc


def train(file: Path, size: int, save):
    i = 0
    buf = []
    with open(file) as f:
        for line in f:
            if size and size < i:
                break
            buf.append(line)
            i += 1
        return train_batch(buf, save)

def segment_batch(storage: Storage, batch: List[str]) -> List[str]:
    segmenter = Segmenter(storage)
    result = []
    # pool = Pool()
    # for line in pool.map(lambda x: chinese.segment_with_preprocessing(segmenter,x), batch):
    for line in batch:
        line = line[:-1]
        if line.strip() != "":
            result.append(chinese.segment_with_preprocessing(segmenter, line))
    return result

def segment_file(storage, input_file: Path, output_file: Path):
    segmenter = Segmenter(storage)
    with open(input_file) as f:
        with open(output_file, "w") as out:
            for line in f:
                line = line[:-1]
                if line.strip() != "":
                    out.write(chinese.segment_with_preprocessing(segmenter, line) +"\n")

wiki_file = Path("/home/pierre/Corpora/wiki7M.raw")
train_file = Path("/home/pierre/Corpora/PKU/all.raw.u8")

test_file = Path("/home/pierre/Corpora/PKU/small.raw.u8")

def run_with_batch(text_file: Path, target_file: Path, batch_size: int=100000):
    buf = []
    counter = 0
    with open(text_file) as f:
        with open(target_file, "w") as out:
            for line in f:
                buf.append(line)
                counter += 1
                if counter % batch_size == 0:
                    storage = train_batch(buf)
                    result = segment_batch(storage, buf)
                    out.write("\n".join(result))
                    buf = []
                    storage.clear(storage)
                    storage.close(storage)
                    del storage
                    del result
                    gc.collect()
            if len(buf) > 0:
                storage = train_batch(buf)
                result = segment_batch(storage, buf)
                out.write("\n".join(result))



def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('action',
                        help='text file used for training',
                        default=False)
    parser.add_argument('-c', '--corpus',
                        help='source file (text corpus)',
                        default=False)
    parser.add_argument('-m', '--model',
                        help='model file (csv)',
                        default=False)
    parser.add_argument('-t', '--target',
                        help = 'target file (segmented text)',
                               default = False)
    parser.add_argument('-L', '--training_length',
                        help='number of lines read for training',
                        default=100000,
                        type=int,
                        required=False)
    parser.add_argument('-S', '--save',
                        help='save model to file',
                        default=None,
                        required=False)
    args = parser.parse_args()
    if args.action == "train":
        assert(args.corpus and args.model)
        storage, voc = train(Path(args.corpus), args.training_length, None)
        storage.update_stats()
        CSVStorage.writeCSV(storage, voc, args.model)
    elif args.action == "segment":
        assert(args.corpus and args.target)
        if args.model:
            storage = CSVStorage(args.model)
        else:
            storage, _ = train(Path(args.corpus), args.training_length, None)
        segment_file(storage, Path(args.corpus), Path(args.target))




if __name__ == "__main__":
    # import math
    # storage, voc = train(wiki_file, 500000, None)
    # print(storage.items())
    # with open("/tmp/lex.csv", "w") as f:
    #     for w in voc:
    #         wl = list(w)
    #         e =  storage.query_autonomy(wl)
    #         if not math.isnan(e):
    #             f.write("\t".join([w, str(e), str(storage.query_count(wl))]) + "\n")
    # del storage
    # del voc
    # stor2 = CSVStorage("/tmp/lex.csv")
    # segment_file(stor2, test_file, Path("/tmp/seg"))
    main()
    #run_with_batch(wiki_file, Path("/tmp/bubu"), 200000)
