from hmmlearn import hmm
import numpy as np
from hmmlearn.utils import normalize
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA, Counter


class HMMBiasResolver(object):
    """
    Parameters
    ----------
    tol: float
        tolerance
    n_iter: int
        number of iterations
    num_of_model: int
        number of models for each code
    startprob_prior: tuple
        prior for start probability
    transmat_prior: tuple
        prior for transition matrix
    emisson_prior: tuple
        prior for emission matrix
    """

    class BiasResolvingHMM(hmm.MultinomialHMM):
        def __init__(self, n_components=1,
                     startprob_prior=1.0, transmat_prior=1.0,
                     algorithm="viterbi", random_state=None,
                     n_iter=10, tol=1e-2, verbose=False,
                     params="ste", init_params="ste", emisson_prior=1.0):
            hmm._BaseHMM.__init__(self, n_components,
                                  startprob_prior=startprob_prior,
                                  transmat_prior=transmat_prior,
                                  algorithm=algorithm,
                                  random_state=random_state,
                                  n_iter=n_iter, tol=tol, verbose=verbose,
                                  params=params, init_params=init_params)
            self.emission_prior = emisson_prior

        def _do_mstep(self, stats):
            hmm._BaseHMM._do_mstep(self, stats)

            if 'e' in self.params:
                emissionprob_ = self.emission_prior - 1.0 + stats['obs']
                self.emissionprob_ = np.where(self.emissionprob_ == 0.0,
                                              self.emissionprob_, emissionprob_)
                normalize(self.emissionprob_, axis=1)

    def __init__(self, tol=1e-4, n_iter=10, num_of_model=2,
                 startprob_prior=(1, 1), transmat_prior=(1, 1), emisson_prior=(1, 1)):

        self.pi_a_b = startprob_prior
        self.a_a_b = transmat_prior
        self.b_a_b = emisson_prior

        self.tol = tol
        self.n_iter = n_iter
        self.num_of_model = num_of_model

        self.code_models = []
        self.data_shape = None

    def _fit(self, data):
        np.seterr(divide='ignore')

        patient_num = data.shape[0]
        T_num = data.shape[1]
        code_num = data.shape[2]
        self.data_shape = data.shape

        # maintain for each code, the feature data part for input to hmmlearn
        code_input_tuple = dict()
        for code in range(code_num):
            patient_list = []
            X = []
            for patient in range(patient_num):
                # if the patient has the code
                if np.count_nonzero(data[patient, :, code]) > 0:
                    patient_list.append(patient)
                    sequence = [0] * T_num
                    for time_window in range(T_num):
                        sequence[time_window] = data[patient][time_window][code]
                    X.append(sequence)
            lengths = [T_num] * len(patient_list)

            code_input_tuple[code] = (patient_list, X, lengths)

        # for each code, expectation of hidden states
        # code -> sequences of expectation of hidden state
        code_q = dict()

        pbar = ProgressBar(widgets=[Counter('Calculating %d/{}'.format(code_num)),
                                    Percentage(), Bar(), Timer(), ETA()], maxval=code_num).start()

        # train a HMM for each code
        for code, (patient_list, X, lengths) in code_input_tuple.items():
            pbar.update(code)
            start_prior = np.array([self.pi_a_b[0], self.pi_a_b[1]])

            trans_prior = np.array([[self.a_a_b[0], self.a_a_b[1]],
                                    [self.a_a_b[1], self.a_a_b[0]]])

            emission_prior = np.array([[self.b_a_b[0], self.b_a_b[1], 1],
                                       [1, self.b_a_b[1], self.b_a_b[0]]])

            best_model = None
            best_logprob = None
            best_posteriors = None

            # replace {-1,0,1} into {0,1,2} as use hidden states' idx instead of real value
            X = np.array(X) + 1

            # number of initialization of A, B, PI models per code, choose the model with max prob
            for i in range(self.num_of_model):
                # construct one model with random init params
                model = self.BiasResolvingHMM(n_components=2, init_params='',
                                              startprob_prior=start_prior,
                                              transmat_prior=trans_prior,
                                              emisson_prior=emission_prior,
                                              n_iter=self.n_iter, tol=self.tol)

                # sample from beta function
                pi_init = np.random.beta(self.pi_a_b[0], self.pi_a_b[1])
                a1_init = np.random.beta(self.a_a_b[0], self.a_a_b[1])
                a2_init = np.random.beta(self.a_a_b[0], self.a_a_b[1])
                b1_init = np.random.beta(self.b_a_b[0], self.b_a_b[1])
                b2_init = np.random.beta(self.b_a_b[0], self.b_a_b[1])

                startprob = np.array([pi_init, 1 - pi_init])
                transmat = np.array([[a1_init, 1 - a1_init], [1 - a2_init, a2_init]])
                emissionprob = np.array([[b1_init, 1 - b1_init, 0], [0, 1 - b2_init, b2_init]])

                model.startprob_ = startprob
                model.transmat_ = transmat
                model.emissionprob_ = emissionprob

                # fit according to current randomized init params
                model.fit(np.concatenate(X).reshape(-1, 1), lengths=lengths)

                # get the match degree between random init params and the X
                logprob, posteriors = model.score_samples(np.concatenate(X).reshape(-1, 1), lengths=lengths)

                # update the best model etc.
                if best_logprob is None or best_logprob < logprob:
                    best_model, best_logprob, best_posteriors = model, logprob, posteriors

            q = np.dot(best_posteriors, [-1, 1])
            code_q[code] = q
            self.code_models.append(best_model)

        pbar.finish()
        return code_q, patient_num, T_num, code_num, code_input_tuple

    def fit(self, data):
        """
        Parameters
        ----------
        data: array-like, shape (n_patients, n_timewindows, n_features)
        """
        self._fit(data)

    def transform(self, data):
        """
        Parameters
        ----------
        data: array-like, shape (n_patients, n_timewindows, n_features)

        Returns
        -------
        transformed_data: array-like, shape (n_patients, n_timewindows, n_features)
        """
        try:
            assert self.data_shape is not None
            assert data.shape == self.data_shape
        except AssertionError as e:
            raise ValueError('model has not been fit or unmatched shape')

        patient_num = data.shape[0]
        T_num = data.shape[1]
        code_num = data.shape[2]
        transformed_data = np.zeros((patient_num, T_num, code_num))

        for code in range(code_num):
            for patient in range(patient_num):
                if np.count_nonzero(data[patient, :, code]) > 0:
                    sequence = data[patient, :, code]
                    sequence = sequence.reshape(-1, 1) + 1
                    lengths = [T_num]
                    posteriors = self.code_models[code].predict_proba(sequence, lengths)
                    q = np.dot(posteriors, [-1, 1])
                    transformed_data[patient, :, code] = q

        return transformed_data

    def fit_transform(self, data):
        """
        Parameters
        ----------
        data: array-like, shape (n_patients, n_timewindows, n_features)

        Returns
        -------
        transformed_data: array-like, shape (n_patients, n_timewindows, n_features)
        """
        transformed_data = np.zeros(data.shape)
        code_q, patient_num, T_num, code_num, code_input_tuple = self._fit(data)
        for code, (patient_list, X, lengths) in code_input_tuple.items():
            q = code_q[code]
            q = q.reshape(len(lengths), T_num)
            for index, patient in enumerate(patient_list):
                transformed_data[patient, :, code] = q[index]

        return transformed_data
