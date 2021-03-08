import numpy as np

DBG = True

def dbgout(s):
    if DBG:
        print(s)

training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(np.int)
print("Shape of the spam training data set:", training_spam.shape)
#print(training_spam)

testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(np.int)
print("Shape of the spam testing data set:", testing_spam.shape)
#print(testing_spam)

def my_accuracy_estimate():
    return 0.5

# This skeleton code simply classifies every input as ham
#
# Here you can see there is a parameter k that is unused, the
# point is to show you how you could set up your own. You might
# also pass in extra data via a train method (also does nothing
#  here). Modify this code as much as you like so long as the
# accuracy test in the cell below runs.

class SpamClassifier:
    def __init__(self, k):
        self.k = k

    def log_class_priors_unbiased(self, data):
        """
        Given a data set with binary response variable (0s and 1s) in the
        left-most column, calculate the logarithm of the empirical class priors,
        that is, the logarithm of the proportions of 0s and 1s:
            log(p(C=0)) and log(p(C=1))

        :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
                     the first column contains the binary response (coded as 0s and 1s).

        :return log_class_priors: a numpy array of length two
        """

        return np.array([np.log(0.5),
                         np.log(0.5)])

    def estimate_log_class_priors(self, data):
        """
        Given a data set with binary response variable (0s and 1s) in the
        left-most column, calculate the logarithm of the empirical class priors,
        that is, the logarithm of the proportions of 0s and 1s:
            log(p(C=0)) and log(p(C=1))

        :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
                     the first column contains the binary response (coded as 0s and 1s).

        :return log_class_priors: a numpy array of length two
        """

        return np.array([np.log(data[data[:, 0] == 1].shape[0] / data.shape[0]),
                         np.log(data[data[:, 0] == 0].shape[0] / data.shape[0])])

    def estimate_log_class_conditional_likelihoods(self, data, alpha=1.0):
        """
        Given a data set with binary response variable (0s and 1s) in the
        left-most column and binary features (words), calculate the empirical
        class-conditional likelihoods, that is,
        log(P(w_i | c)) for all features w_i and both classes (c in {0, 1}).

        Assume a multinomial feature distribution and use Laplace smoothing
        if alpha > 0.

        :param data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]

        :return theta:
            a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
            logarithm of the probability of feature i appearing in a sample belonging
            to class j.
        """
        spam = data[data[:, 0] == 1]
        ham = data[data[:, 0] != 1]

        spam_total = spam.shape[0]
        ham_total = ham.shape[0]

        Nwspam = [spam[:, x].sum() for x in range(1, spam.shape[1])]
        Nwham = [ham[:, x].sum() for x in range(1, ham.shape[1])]
        Nspam = sum(Nwspam)
        Nham = sum(Nwham)

        dbgout("Total messages: {}".format(spam_total + ham_total))
        dbgout("Total spam:{}".format(spam_total))
        dbgout("Total ham:{}".format(ham_total))
        dbgout("Num times w appears spam:{}".format(Nwspam))
        dbgout("Num times w appears ham:{}".format(Nwham))
        dbgout("Num words spam:{}".format(Nspam))
        dbgout("Num words ham:{}".format(Nham))
        dbgout("Num words total:{}".format(sum([data[:, x].sum() for x in range(1, data.shape[1])])))
        dbgout(self.estimate_log_class_priors(data))

        spam_theta = [np.log((Nwspam[x] + alpha)/(Nspam + self.k*alpha)) for x in range(0, len(Nwspam))]
        ham_theta = [np.log((Nwham[x] + alpha) / (Nham + self.k*alpha)) for x in range(0, len(Nwham))]
        return np.array([spam_theta, ham_theta])

    def train(self):
        # Isolating spam and ham messages first
        train_set = np.loadtxt(open("data/training_spam.csv"), delimiter=",").astype(np.int)
        self.log_priors = self.estimate_log_class_priors(train_set)
        # self.log_priors = self.log_class_priors_unbiased(train_set)
        # print(self.log_priors)
        self.log_class_cond_likelihoods = self.estimate_log_class_conditional_likelihoods(train_set)

    def predict(self, new_data):
        pspam = new_data[:,:] * self.log_class_cond_likelihoods[0]
        pham = new_data[:,:] * self.log_class_cond_likelihoods[1]
        result = [1 if (self.log_priors[0] + sum(s)) > (self.log_priors[1] + sum(h)) else 0 for s, h in zip(pspam,pham)]

        return result


def create_classifier():
    classifier = SpamClassifier(k=1)
    classifier.train()
    return classifier


classifier = create_classifier()

SKIP_TESTS = False

if not SKIP_TESTS:
    testing_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(np.int)
    test_data = testing_spam[:, 1:]
    test_labels = testing_spam[:, 0]

    predictions = classifier.predict(test_data)
    # print(predictions)
    # print(test_labels)
    accuracy = np.count_nonzero(predictions == test_labels)/test_labels.shape[0]
    print(f"Accuracy on test data is: {accuracy}")

    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in range(len(predictions)):
        if predictions[i] != test_labels[i]:
            if predictions[i] == 1:
                fp += 1
            else:
                fn += 1
        else:
            if predictions[i] == 1:
                tp += 1
            else:
                tn += 1

    print(f"FPs on test data is: {fp}")
    print(f"FNs on test data is: {fn}")
    print(f"TPs on test data is: {tp}")
    print(f"TNs on test data is: {tn}")