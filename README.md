# Wine Quality Prediction - Classifier Comparison

I compared the accuracy of three classification models (SVM, KNN, and Gaussian Naive Bayes) to see which one best predicts wine quality.

## Dataset

Used the [UCI Wine Quality (Red)](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) dataset - 1,599 rows, 12 columns. No missing values.

The original `quality` column was a numeric score from 3–8. Since this was a classification task, I converted it into a binary target:

- Tested `quality >= 7` as "good" - caused serious class imbalance (only ~14% positive class)
- Switched to `quality >= 6` as "good" - much more evenly distributed (~53% / 47%)

This created a new `quality_binary` column used as the target.

## Preprocessing

- Checked correlation between features to look for multicollinearity - found only moderate correlations (e.g. free vs. total sulfur dioxide, density vs. alcohol/fixed acidity), nothing severe enough to justify dropping any features. All 11 features were kept.
- Split the data into train/test sets, then scaled features with `StandardScaler`, since several features (like `total sulfur dioxide` and `alcohol`) have naturally different numeric ranges, and distance-based models (SVM, KNN) would otherwise be skewed by larger-magnitude features.

## Model Tuning

Used `GridSearchCV` with 5-fold cross-validation to find the best parameters for each model, rather than testing values one at a time. Cross-validation matters here because it evaluates each parameter combination across multiple splits of the data, so the "best" result isn't just luck from one favorable train/test split.

## Results

| Model | Best Parameters | Cross-Validation Score | Test Accuracy |
|---|---|---|---|
| SVM | `C=10`, `kernel='rbf'` | 77.2% | 76.3% |
| KNN | `n_neighbors=13`, `weights='distance'` | 79.7% | 78.8% |
| Gaussian Naive Bayes | `var_smoothing=1.0` | 73.1% | 74.1% |

**KNN performed best overall**, beating SVM by ~2–3% and Naive Bayes by ~5%. It used `weights='distance'`, meaning closer neighbors had more influence on the prediction than farther ones - likely helping it handle borderline cases better than treating every neighbor equally.

**Naive Bayes came in last**, which fits its core weakness: it assumes all features are independent. Wine chemistry features aren't fully independent (e.g. alcohol and density are related), so that assumption likely held it back, even though the multicollinearity wasn't severe enough to justify dropping features outright.

One additional finding: when tuning `var_smoothing` for Naive Bayes, the best value landed right at the edge of the search range even after widening it further - suggesting the score had plateaued rather than genuinely improving. This points to the independence assumption being the real bottleneck for Naive Bayes here, not a lack of tuning.

## Conclusion

For this dataset, **KNN was the most effective of the three models**, likely because wine quality doesn't split cleanly along a linear boundary or fully independent features - something KNN's neighbor-based approach handles more naturally than SVM or Naive Bayes.
