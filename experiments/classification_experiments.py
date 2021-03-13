from classifiers import AbstractTokenizedDocumentClassifier
from embeddings import WordEmbeddings
from nnclassifiers import StackedLSTMTokenizedDocumentClassifier
from nnclassifiers_experimental import StructuredSelfAttentiveSentenceEmbedding
from readers import JSONPerLineDocumentReader, AHVersusDeltaThreadReader
from tcframework import LabeledTokenizedDocumentReader, AbstractEvaluator, Fold, TokenizedDocumentReader, \
    TokenizedDocument, ClassificationEvaluator
from vocabulary import Vocabulary


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

        predictions = self.classifier.test(instances)

        result = dict()
        for instance, prediction in zip(instances, predictions):
            assert isinstance(instance, TokenizedDocument)
            assert isinstance(prediction, float)
            # get id and put the label to the resulting dictionary
            result[instance.id] = prediction

        return result

def cross_validation_ah():
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
    e = ClassificationExperiment(reader, StackedLSTMTokenizedDocumentClassifier(vocabulary, embeddings), ClassificationEvaluator())
    e.run()


if __name__ == '__main__':
  cross_validation_ah()
