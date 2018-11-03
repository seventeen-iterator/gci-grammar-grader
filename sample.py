import pickle

with open('lstm_prod.pickle', 'rb') as f:
    predict, _ = pickle.load(f)

# playing around
print(
    predict([
        'I like playing computer games.',
        'SDFsdfsdf sdfdfsd sdfsdFSDF',
        'Computer games I like to play',
        'computars aint good'
    ])
)
