import re
import difflib


def match_tag_sequences(predicted_tags, true_tags):
    def clean_tag_lines(lines):
        cleaned_lines = []
        for line in lines:
            # Remove quotes at the beginning or end of the lines
            line = re.sub(r'^"|"$', "", line)

            # Remove any character that is not Arabic or punctuation
            line = re.sub(r"[^\u0600-\u06FF\s\.\?!،؛]", "", line)

            # Print the cleaned line
            cleaned_lines.append(line)
        return cleaned_lines

    matcher = difflib.Differ()
    match_sequence = matcher.compare(
        clean_tag_lines(true_tags),
        clean_tag_lines(predicted_tags),
    )
    return list(match_sequence)


def pos_tagging_accuracy(predicted_tags, true_tags):
    # replace extra spaces. ChatGPT sometimes add spaces around its prediction tokens.
    predicted_tags = predicted_tags.replace(" ", "")
    true_tags = true_tags.replace(" ", "")
    # Convert the predicted and true tags to lists
    predicted_tags = predicted_tags.strip().split("\n")
    true_tags = true_tags.strip().split("\n")

    match_sequence = match_tag_sequences(
        predicted_tags=predicted_tags,
        true_tags=true_tags,
    )

    # print(len(match_sequence))
    # print(len(true_tags))
    # print(len(predicted_tags))

    # Initialize the count of correct tags
    correct_count = 0
    matched_index = 0
    for line in match_sequence:
        # if the current line only presents in the predicted sequence, just continue without any penalization.
        if line.startswith("+"):
            # remove that element from the predicted tags to have equal length lists
            predicted_tags.pop(matched_index)
            continue
        if line.startswith("?"):
            continue
        # if the current line is present in the true sequence but not in the predicted sequence, penalize (matched_index incremented and correct_count was not) and continue
        if line.startswith("-"):
            predicted_tags.insert(matched_index, "X")
            matched_index += 1
            continue
        # print(line)
        # print(predicted_tags[matched_index])
        # print(true_tags[matched_index])
        # print(f"{matched_index:=}")
        # print("*" * 100)
        # try:
        #     predicted_tag = predicted_tags[matched_index].split(":")[1].strip()
        # except Exception as e:
        #     print('predicted tags are:')
        #     print(predicted_tags)
        #     print('true tags are:')
        #     print(true_tags)
        #     print("Error occured at line:")
        #     print(line)
        #     print('Match Sequence:')
        #     print(match_sequence)
        #     raise e

        predicted_tags_tokens = predicted_tags[matched_index].split(":")

        if (
            len(predicted_tags_tokens) == 2
        ):  # usually, at the end of the sentence, chatGPT forget to tag the PUNCTUATIONs.
            predicted_tag = predicted_tags_tokens[1].strip()
            true_tag = true_tags[matched_index].split(":")[1].strip()
            if predicted_tag.upper() == true_tag.upper():
                correct_count += 1
        matched_index += 1

    # Calculate the accuracy and return it
    accuracy = correct_count / len(true_tags)
    return accuracy
