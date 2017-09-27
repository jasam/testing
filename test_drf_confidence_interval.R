library(cvAUC)
library(h2o)

h2o.init(nthreads = -1)  #Start a local H2O cluster using nthreads = num available cores

# Load binary-response dataset
data(admissions)
train = as.h2o(x = admissions)

# Dimensions
dim(train)

# Columns
names(train)

# Identity the response column
ycol = "Y"

# Identify the predictor columns
xcols = setdiff(names(train), ycol)

# Convert response to factor (required by randomForest)
train[, ycol] = as.factor(train[, ycol])

# Train a default RF model with 50 trees
model <- h2o.randomForest(
    x = xcols,
    y = ycol,
    training_frame = train,
    seed = 1,
    ntrees = 50,
    keep_cross_validation_fold_assignment = TRUE,
    keep_cross_validation_predictions = TRUE,
    nfolds = 3
)

# get folds
folds = as.data.frame(h2o.cross_validation_fold_assignment(model))

pp = h2o.cross_validation_holdout_predictions(model)
predictions = as.data.frame(pp$p1)
labels = as.data.frame(train$Y)

res = ci.cvAUC(
    predictions = predictions$p1,
    labels = labels$Y,
    folds = folds[,1],
    confidence = 0.95
)

res