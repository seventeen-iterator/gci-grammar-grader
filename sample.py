import pickle

with open('lstm_prod.pickle', 'rb') as f:
    predict, _ = pickle.load(f)

# playing around
print(
    predict([
        'I like playing computer games.',  # Correct sentence
        'SDFsdfsdf sdfdfsd sdfsdFSDF',     # Just bunch of letters
        'Computer games I like to play',   # Somewhat correct sentence
        'computars aint good'              # Incorrect sentence
    ])
)
