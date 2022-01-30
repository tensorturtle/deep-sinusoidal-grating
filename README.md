# DeepSinusoidalGrating
Exploratory analysis comparing deep artificial neural networks and humans in sinusoidal grating categorization task.

https://sci.bban.top/pdf/10.1016/j.actpsy.2018.08.016.pdf#view=FitH

## TODO

- [x] Validation per epoch
- [x] ipywidgets modification for distribution parameters
- [ ] Training accuracy Visualization 
- [ ] Training regime similar to humans (try with starting with pre-trained?)
- [x] Convolution visualization integration: https://jacobgil.github.io/deeplearning/filter-visualizations, https://github.com/fossasia/visdom
- [x] Hosting on blog solutions?
- [x] Count number of parameters
- [x] Saveable dataset

## Hypothesis

Artificial neural networks (ANNs) will excel in categorizing sinusoidal gratings, becase they are defined by just two parameters: the frequency and rotation. Neural networks come into their own when there are a large (unknown) number of parameters, so this task should be trivial. However, I predict that whereas humans perform better on the 'rule-based' categorization, ANNs will perform better on the 'information-integration' categorization, because the 'information-integraion' categorization relies on a linear distinction between the two categories, whereas the 'rule-based' categorization relies on a non-linear (probably cubic?) distinction between the two categories.

## Methodology

1. Generate two sets of sinusoidal gratings, one according to the rule-based categorization, and one according to the information-integration categorization.
2. Train a deep ANN on the dataset with the same training regimen as applied to humans.
3. Test the ANN on the dataset.
4. Train a deep ANN on the dataset with more training data.
5. Test the ANN on the dataset.
6. Compare the performance of the ANN to humans (data obtained from literature)

## Exploration Notes

AlexNet-like NN performance is sensitive to random seed. Sometimes, the nn won't even learn.
