NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_HEIGHT = 400
IMG_WIDTH = 400

# DPAC Dataset
DPAC_ATT_CAT_COUNT = {
    "emotion": 7,
    "age": 5,
    "gender": 2
}

DPAC_EMOTION_LABEL_TO_IDX = {
    'Happiness': 0,
    'Anger': 1,
    'Surprise': 2,
    'Disgust': 3,
    'Fear': 4,
    'Sadness': 5,
    'Neutral': 6
}

DPAC_GENDER_LABEL_TO_IDX = {
    'Male': 0,
    'Female': 1
}

DPAC_AGE_LABEL_TO_IDX = {
    'Child': 0,
    'Adolescent': 1,
    'Young Adult': 2,
    'Middle-Aged Adult': 3,
    'Older Adult': 4
}

# IoG Dataset
IOG_ATT_CAT_COUNT = {
    "age": 7,
    "gender": 2
}

# EMOTIC Datasetrieval and Search Engi
EMOTIC_GENDER_CAT = ['Male', 'Female']
EMOTIC_EMOTION_CAT = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion',
                      'Confidence', 'Disapproval', 'Disconnection', 'Disquietment',
                      'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem',
                      'Excitement', 'Fatigue', 'Fear', 'Happiness', 'Pain', 'Peace',
                      'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise',
                      'Sympathy', 'Yearning']


# CAER Dataset
CAER_ATT_CAT_COUNT = {
    "emotion": 5
}


#dict(map(reversed, emotions_cat.items()))
# emotion_cat = {0: 'Affection',
#  1: 'Anger',
#  2: 'Annoyance',
#  3: 'Anticipation',
#  4: 'Aversion',
#  5: 'Confidence',
#  6: 'Disapproval',
#  7: 'Disconnection',
#  8: 'Disquietment',
#  9: 'Doubt/Confusion',
#  10: 'Embarrassment',
#  11: 'Engagement',
#  12: 'Esteem',
#  13: 'Excitement',
#  14: 'Fatigue',
#  15: 'Fear',
#  16: 'Happiness',
#  17: 'Pain',
#  18: 'Peace',
#  19: 'Pleasure',
#  20: 'Sadness',
#  21: 'Sensitivity',
#  22: 'Suffering',
#  23: 'Surprise',
#  24: 'Sympathy',
#  25: 'Yearning'
# }
