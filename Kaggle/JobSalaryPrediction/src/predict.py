import pickle

import utils

def main():
    model = utils.load_model()
    
    valid_df = utils.get_valid_df()
    
    predictions = model.predict(valid_df)
    predictions = predictions.reshape(len(predictions), 1)
    
    utils.write_submission(predictions)


if __name__ == "__main__":
    main()

