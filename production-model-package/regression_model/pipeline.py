from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

# to persist the model and the scaler
import joblib

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

# pipeline
from sklearn.pipeline import Pipeline

#import preprocessors
from sklearn.base import BaseEstimator, TransformerMixin


!pip install feature_engine

# for imputation
from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)

# for encoding categorical variables
from feature_engine.encoding import (
    RareLabelEncoder,
    OneHotEncoder
)

# set up the pipeline
titanic_pipe = Pipeline([

    # ===== IMPUTATION =====
    # impute categorical variables with string 'missing'
    ('categorical_imputation', CategoricalImputer(
    imputation_method='missing', variables=CATEGORICAL_VARIABLES)),
    # add missing indicator to numerical variables
    ('missing_indicator',AddMissingIndicator (variables=NUMERICAL_VARIABLES)),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(imputation_method='median',
                                       variables=NUMERICAL_VARIABLES)),


    # Extract first letter from cabin
    ('extract_letter', ExtractLetterTransformer(variables=CABIN)),


    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder', RareLabelEncoder(
        tol=0.05, n_categories=1, variables=CATEGORICAL_VARIABLES)),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder',OneHotEncoder( drop_last=True, variables=CATEGORICAL_VARIABLES )),

    # scale using standardization
    ('scaler', StandardScaler ()),

    # logistic regression (use C=0.0005 and random_state=0)
    ('Logit', LogisticRegression(C=0.0005, random_state=0)),
])
