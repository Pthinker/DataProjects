import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

import utils
from features import FeatureMapper, SimpleTransform


def feature_extractor():
    features = [('FullDescription-Bag of Words', 'FullDescription', CountVectorizer(max_features=500)),
                ('Title-Bag of Words', 'Title', CountVectorizer(max_features=100)),
                ('LocationRaw-Bag of Words', 'LocationRaw', CountVectorizer(max_features=100)),
                ('LocationNormalized-Bag of Words', 'LocationNormalized', CountVectorizer(max_features=100))]
    
    combined = FeatureMapper(features)
    return combined


def randomforest(train_df):
    features = feature_extractor()
    steps = [("extract_features", features),
             ("classify", RandomForestRegressor(n_estimators=400, 
                                                verbose=2,
                                                n_jobs=6,
                                                min_samples_split=20,
                                                random_state=3465345))]
    
    model = Pipeline(steps)
    model.fit(train_df, train_df["SalaryNormalized"])
    
    return model


def main():
    train_df = utils.get_train_df()

    model = randomforest(train_df)
    utils.save_model(model)


if __name__ == "__main__":
    main()

