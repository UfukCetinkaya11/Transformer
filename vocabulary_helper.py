def get_vocabulary(dataset, start_token, padding_token, end_token):
    # Create an empty set to store unique characters
    unique_chars_set = set()

    # Iterate through each sentence in the dataset
    for sentence in dataset:
        # Iterate through each character in the sentence
        for char in sentence:
            # Add the character to the set
            unique_chars_set.add(char)

    # Convert the set to a list to maintain the order of insertion
    unique_chars_list = list(unique_chars_set)
    unique_chars_list = (
        sorted(unique_chars_list))
    unique_chars_list.insert(0, start_token)
    unique_chars_list.extend([padding_token, end_token])

    return unique_chars_list
