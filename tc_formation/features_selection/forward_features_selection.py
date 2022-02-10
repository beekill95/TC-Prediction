import numpy as np
import tensorflow.keras as keras


class ForwardFeaturesSelection:
    def __init__(self, model_fn, data_shape, nb_features_to_select):
        self._init_data_shape = data_shape
        self._nb_feature_channels = data_shape[-1]
        self._model_fn = model_fn
        self._nb_features_to_select = nb_features_to_select
        self._best_proposal = None
        self._best_proposal_score = 0.0

    def best_proposal(self):
        return self._best_proposal
    
    def best_proposal_score(self):
        return self._best_proposal_score

    def fit(self, training, validation, initial_features=None):
        best_objective_value = 0.0
        best_proposal = (np.zeros(self._nb_feature_channels) if initial_features is None
                         else initial_features)
        proposals = self._propose_feature_masks(best_proposal)
        
        while best_proposal.sum() < self._nb_features_to_select:
            print(f'Evaluating proposal with nb features: {best_proposal.sum() + 1}')
            proposal_updated = False

            for proposal in proposals:
                print('\nCurrent proposal:', proposal)
                masked_training = training.map(lambda X, y: (X[:, :, :] * proposal, y))
                masked_validation = validation.map(lambda X, y: (X[:, :, :] * proposal, y))

                model = self._model_fn(self._init_data_shape)
                model.fit(
                    masked_training,
                    epochs=50,
                    validation_data=masked_validation,
                    class_weight={1: 10., 0: 1.},
                    shuffle=True,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor='val_f1_score',
                            mode='max',
                            verbose=1,
                            patience=20,
                            restore_best_weights=True),
                    ],
                    verbose=0,
                )

                objective_value = model.evaluate(masked_validation)[4]
                if objective_value > best_objective_value:
                    print(f'Best proposal improve from {best_objective_value} to {objective_value}')
                    print(f'with proposal {proposal}')
                    best_objective_value = objective_value
                    best_proposal = proposal
                    proposal_updated = True


            if not proposal_updated:
                print('Proposal not updated. Stop!!')
                break
            print('== Propose next batch of masks from current best proposal', best_proposal)
            proposals = self._propose_feature_masks(best_proposal)

        # Save the best proposal.
        self._best_proposal = best_proposal
        self._best_proposal_score = best_objective_value

    def _propose_feature_masks(self, feature_mask):
        mask = ~(feature_mask > 0)
        proposals = []
        for i, m in enumerate(mask):
            if m:
                proposal = np.zeros(self._nb_feature_channels)
                proposal[i] = 1
                proposals.append(proposal + feature_mask)
                
        return proposals
