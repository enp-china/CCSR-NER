def generate_dictionary(file_path):
    from flair.data import Dictionary
    char_dictionary: Dictionary = Dictionary()

    # counter object
    import collections
    counter = collections.Counter()

    processed = 0

    import glob
    files = glob.glob(file_path+'*.*')

    print(files)
    for file in files:
        print(file)

        with open(file, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:

                processed += 1            
                chars = list(line)
                tokens += len(chars)

                # Add chars to the dictionary
                counter.update(chars)

                # comment this line in to speed things up (if the corpus is too large)
                # if tokens > 50000000: break

        # break

    total_count = 0
    for letter, count in counter.most_common():
        total_count += count

    print(total_count)
    print(processed)

    sum = 0
    idx = 0
    for letter, count in counter.most_common():
        sum += count
        percentile = (sum / total_count)

        # comment this line in to use only top X percentile of chars, otherwise filter later
        # if percentile < 0.00001: break

        char_dictionary.add_item(letter)
        idx += 1
        print('%d\t%s\t%7d\t%7d\t%f' % (idx, letter, count, sum, percentile))

    print(char_dictionary.item2idx)

    import pickle
    with open(file_path+'mappings', 'wb') as f:
        mappings = {
            'idx2item': char_dictionary.idx2item,
            'item2idx': char_dictionary.item2idx
        }
        pickle.dump(mappings, f)


def train_LM(file_path,model_path,is_forward_lm=True):
    from flair.data import Dictionary
    from flair.models import LanguageModel
    from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

    
    dictionary = Dictionary.load_from_file(file_path+'mappings')


    # get your corpus, process forward and at the character level
    corpus = TextCorpus(file_path,
                        dictionary,
                        is_forward_lm,
                        character_level=True)

    # instantiate your language model, set hidden size and number of layers
    language_model = LanguageModel(dictionary,
                                   is_forward_lm,
                                   hidden_size=128,
                                   nlayers=1)

    # train your language model
    trainer = LanguageModelTrainer(language_model, corpus)

    trainer.train(model_path,
                  sequence_length=100,
                  mini_batch_size=32,
                  max_epochs=10)

if __name__ == "__main__":
    generate_dictionary(sys.argv[1])
    train_LM(sys.argv[1],sys.argv[2],bool(sys.argv[3]))
