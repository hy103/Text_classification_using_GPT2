import pandas as pd
data = pd.read_csv("../code/sms_spam_collection/SMSSpamCollection.tsv", sep='\t', header = None, names = ["Label", "Text"])


def create_balanced_datset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state = 123
    )

    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

def random_split(df, train_ratio, validation_ratio):
    df = df.sample(
        frac =1, random_state = 123
    ).reset_index(drop = True)
    train_end = int(len(df)* train_ratio)
    validation_end = train_end + int(len(df)* validation_ratio)

    train_df = df[: train_end]
    validation_df = df[train_end : validation_end]
    test_df = df[validation_end: ]

    return train_df, validation_df, test_df


balanced_df = create_balanced_datset(data)

balanced_df["Label"] = balanced_df["Label"].map({"ham" : 0, "spam": 1})
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
print(train_df.shape, validation_df.shape, test_df.shape)
train_df.to_csv("../code/sms_spam_collection/train.csv", index = None)
validation_df.to_csv("../code/sms_spam_collection/val.csv", index = None)
test_df.to_csv("../code/sms_spam_collection/test.csv", index = None)
