# modified for custom training and testing on GPU by Utkarsh Patel

from classifiers import AbstractTokenizedDocumentClassifier
from embeddings import WordEmbeddings
from nnclassifiers import StackedLSTMTokenizedDocumentClassifier, CNNTokenizedDocumentClassifier
from nnclassifiers_experimental import StructuredSelfAttentiveSentenceEmbedding
from readers import JSONPerLineDocumentReader, AHVersusDeltaThreadReader
from tcframework import LabeledTokenizedDocumentReader, AbstractEvaluator, Fold, TokenizedDocumentReader, \
    TokenizedDocument, ClassificationEvaluator
from comment import Comment
from vocabulary import Vocabulary
import argparse, os
import numpy as np
import pickle


class ClassificationExperiment:
    def __init__(self, labeled_document_reader: LabeledTokenizedDocumentReader,
                 classifier: AbstractTokenizedDocumentClassifier, evaluator: AbstractEvaluator):
        self.reader = labeled_document_reader
        self.classifier = classifier
        self.evaluator = evaluator

    def run(self) -> None:
        __folds = self.reader.get_folds()

        for i, fold in enumerate(__folds, start=1):
            assert isinstance(fold, Fold)
            assert fold.train and fold.test

            print("Running fold %d/%d" % (i, len(__folds)))
            self.classifier.train(fold.train)
            predicted_labels = self.classifier.test(fold.test, fold_no=i)

            self.evaluate_fold(fold.test, predicted_labels)

            print("Evaluating after %d folds" % i)
            self.evaluator.evaluate()

        print("Final evaluation; reader.input_path_train was %s" % self.reader.input_path_train)
        self.evaluator.evaluate()

    def evaluate_fold(self, labeled_document_instances: list, predicted_labels: list):
        assert labeled_document_instances
        assert len(predicted_labels)
        assert len(labeled_document_instances) == len(predicted_labels), "Prediction size mismatch"

        assert isinstance(labeled_document_instances[0].label, type(predicted_labels[0]))

        # convert string labels int
        all_gold_labels = [doc.label for doc in labeled_document_instances]

        # collect IDs
        ids = [doc.id for doc in labeled_document_instances]

        self.evaluator.add_single_fold_results(all_gold_labels, predicted_labels, ids)

    def label_external(self, document_reader: TokenizedDocumentReader) -> dict:
        self.classifier.train(self.reader.train, validation=False)
        instances = document_reader.instances

        predictions, probs = self.classifier.test(instances)
        probs = list(probs)
        result = dict()
        for instance, prediction, prob in zip(instances, predictions, probs):
            assert isinstance(instance, TokenizedDocument)
            # assert isinstance(prediction, float)
            # get id and put the label to the resulting dictionary
            cur_text = ' '.join(instance.tokens)
            result[instance.id] = (prediction, prob)

        return result

def cross_validation_ah(model_type):
    # classification without context
    import random
    random.seed(1234567)

    import tensorflow as tf
    if tf.test.is_gpu_available():
        strategy = tf.distribute.MirroredStrategy()
        print('Using GPU')
    else:
        raise ValueError('CPU not recommended.')

    with strategy.scope():
        vocabulary = Vocabulary.deserialize('en-top100k.vocabulary.pkl.gz')
        embeddings = WordEmbeddings.deserialize('en-top100k.embeddings.pkl.gz')
        reader = JSONPerLineDocumentReader('data/experiments/ah-classification1/exported-3621-sampled-positive-negative-ah-no-context.json', True)
        e = None
        if model_type == 'cnn':
            e = ClassificationExperiment(reader, CNNTokenizedDocumentClassifier(vocabulary, embeddings), ClassificationEvaluator())
        else:
            e = ClassificationExperiment(reader, StackedLSTMTokenizedDocumentClassifier(vocabulary, embeddings), ClassificationEvaluator())
        e.run()

def cross_validation_thread_ah_delta_context3():
    # classification with context
    import random
    random.seed(1234567)

    import tensorflow as tf
    if tf.test.is_gpu_available():
        strategy = tf.distribute.MirroredStrategy()
        print('Using GPU')
    else:
        raise ValueError('CPU not recommended.')

    with strategy.scope():
        vocabulary = Vocabulary.deserialize('en-top100k.vocabulary.pkl.gz')
        embeddings = WordEmbeddings.deserialize('en-top100k.embeddings.pkl.gz')
        reader = AHVersusDeltaThreadReader('data/sampled-threads-ah-delta-context3', True)
        e = ClassificationExperiment(reader, StructuredSelfAttentiveSentenceEmbedding(vocabulary, embeddings, '/tmp/visualization-context3'), ClassificationEvaluator())
        e.run()

def train_test_model_with_context(train_dir, indir, outdir):
    '''Custom training and testing SSAE model
    :param train_dir: Path to JSON file containing training examples
    :param indir: Path to LOG file containing examples as Comment() object (which has already been classified by Bert)
    :param outdir: Path to LOG file to be created by adding prediction of this model as well'''

    import random
    random.seed(1234567)

    import tensorflow as tf
    if tf.test.is_gpu_available():
        strategy = tf.distribute.MirroredStrategy()
        print('Using GPU')
    else:
        raise ValueError('CPU not recommended.')

    with strategy.scope():
        vocabulary = Vocabulary.deserialize('en-top100k.vocabulary.pkl.gz')
        embeddings = WordEmbeddings.deserialize('en-top100k.embeddings.pkl.gz')
        reader = JSONPerLineDocumentReader(train_dir, True)
        e = ClassificationExperiment(reader, StructuredSelfAttentiveSentenceEmbedding(vocabulary, embeddings), ClassificationEvaluator())
        test_comments = TokenizedDocumentReader(indir)
        result = e.label_external(test_comments)

    for k in result.keys():
        print(f'{k}: {result[k]}')

    instances = dict()

    e = Comment(-1, 'lol', 'ah')
    f = open(indir, 'rb')

    try:
        while True:
            e = pickle.load(f)
            print(e)
            instances[str(e.id)] = e
    except EOFError:
        f.close()

    f = open(outdir, 'wb')
    
    for key in result.keys():
        model_label, model_score = result[key]
        model_label = model_label.lower()
        score = model_score[1]
        if model_label == 'none':
            score = model_score[0]
        instances[key].add_model(model_type, model_label, score, None)
        e = instances[key]
        print(e)
        print(e.labels)
        print(e.scores)
        print('=' * 20)
        pickle.dump(instances[key], f)
        
    f.close()


def train_test_model_no_context(model_type, train_dir, indir, outdir):
    # Training and testing CNN / BiLSTM model on custom data
    # :param train_dir: Path to JSON file containing training examples
    # :param indir: Path to LOG file containing examples as Comment() object (which has already been classified by Bert)
    # :param outdir: Path to LOG file to be created by adding prediction of this model as well

    import random
    random.seed(1234567)

    import tensorflow as tf
    if tf.test.is_gpu_available():
        strategy = tf.distribute.MirroredStrategy()
        print('Using GPU')
    else:
        raise ValueError('CPU not recommended.')

    with strategy.scope():
        vocabulary = Vocabulary.deserialize('en-top100k.vocabulary.pkl.gz')
        embeddings = WordEmbeddings.deserialize('en-top100k.embeddings.pkl.gz')
        reader = JSONPerLineDocumentReader(train_dir, True)
        e = None
        if model_type == 'cnn':
            e = ClassificationExperiment(reader, CNNTokenizedDocumentClassifier(vocabulary, embeddings), ClassificationEvaluator())
        else:
            e = ClassificationExperiment(reader, StackedLSTMTokenizedDocumentClassifier(vocabulary, embeddings), ClassificationEvaluator())
        # e.run()
        test_comments = TokenizedDocumentReader(indir)
        result = e.label_external(test_comments)
    for k in result.keys():
        print(f'{k}: {result[k]}')

    instances = dict()

    e = Comment(-1, 'lol', 'ah')
    f = open(indir, 'rb')

    try:
        while True:
            e = pickle.load(f)
            print(e)
            instances[str(e.id)] = e
    except EOFError:
        f.close()

    f = open(outdir, 'wb')
    
    for key in result.keys():
        model_label, model_score = result[key]
        model_label = model_label.lower()
        score = model_score[1]
        if model_label == 'none':
            score = model_score[0]
        instances[key].add_model(model_type, model_label, score, None)
        e = instances[key]
        print(e)
        print(e.labels)
        print(e.scores)
        print('=' * 20)
        pickle.dump(instances[key], f)
        
    f.close()

def main3():
    # Custom training and testing for context-model (SSAE)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default=None, type=str, required=True, help="Path to JSON file containing training examples")
    parser.add_argument("--indir", default=None, type=str, required=True, help="Path to LOG file containing examples as Comment() object (which has already been classified by Bert)")
    parser.add_argument("--outdir", default=None, type=str, required=True, help="Path to LOG file to be created by adding prediction of this model as well")
    args = parser.parse_args()
    train_test_model_with_context(args.train_dir, args.indir, args.outdir)
        
def main2():
    # Custom training and testing for no-context models
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True, help="Model used for classification")
    parser.add_argument("--train_dir", default=None, type=str, required=True, help="Path to JSON file containing training examples")
    parser.add_argument("--indir", default=None, type=str, required=True, help="Path to LOG file containing examples as Comment() object (which has already been classified by Bert)")
    parser.add_argument("--outdir", default=None, type=str, required=True, help="Path to LOG file to be created by adding prediction of this model as well")
    args = parser.parse_args()
    train_test_model_no_context(args.model, args.train_dir, args.indir, args.outdir)

def main():
    # For supervised learning task (with or without context) as described in the paper
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True, help="Model used for classification")
    args = parser.parse_args()
    if args.model == 'ssase':
        cross_validation_thread_ah_delta_context3()
    else:
        cross_validation_ah(args.model)


if __name__ == '__main__':
    # main()
    main2()
