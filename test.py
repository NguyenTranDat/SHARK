# import pickle
# import torch

# # Load the data
# with open("src/example/dev_data.pkl", "rb") as f:
#     dev_data = pickle.load(f)

# with open("src/example/test_data.pkl", "rb") as f:
#     test_data = pickle.load(f)

# with open("src/example/train_data.pkl", "rb") as f:
#     train_data = pickle.load(f)


# # Define a function to find the max of each tensor in the data
# def find_max_values(data):
#     if not data:
#         return {}

#     sample_item = next(item for item in data if isinstance(item, dict))
#     max_values = {key: float("-inf") for key in sample_item if isinstance(sample_item[key], torch.Tensor)}

#     for item in data:
#         for key in max_values:
#             if isinstance(item[key], torch.Tensor):
#                 max_value = item[key].max().item()
#                 if max_value > max_values[key]:
#                     max_values[key] = max_value

#     return max_values


# # Find max values in each dataset
# max_dev_data = find_max_values(dev_data)
# max_test_data = find_max_values(test_data)
# max_train_data = find_max_values(train_data)

# print("Max values in dev_data:")
# print(max_dev_data)

# print("\nMax values in test_data:")
# print(max_test_data)

# print("\nMax values in train_data:")
# print(max_train_data)


import re


def tokenize_raw_words(text):
    tokens = []

    # Biểu thức chính quy để phân tách văn bản thành từ và các phần từ đặc biệt
    pattern = r"<<xReact\d+>>|<<oReact\d+>>|<<U\d+>>|[a-zA-Z0-9'-]+|[\.,!?;:]"

    # Sử dụng biểu thức chính quy để tách chuỗi thành các từ và các phần từ đặc biệt
    matches = re.findall(pattern, text)

    # Thêm từng phần từ vào danh sách token
    tokens.extend(matches)

    return tokens


raw_words = "<<U0>> Chandler : Alright , so I am back in high school , I am standing in the middle of the cafeteria , and I realize I am totally naked . <<U1>> All : Oh , yeah . Had that dream . <<U2>> Chandler : Then I look down , and I realize there is a phone ... there . <<U3>> Joey : Instead of ... ? <<U4>> Chandler : That is right . <<U5>> Joey : Never had that dream . <<U6>> Phoebe : No . <<U7>> Chandler : All of a sudden , the phone starts to ring ."
text = (
    "<<xReact0>> Chandler feels embarrassed, ashamed, nervous. <<xReact1>> All feels happy, excited, awake. <<xReact2>>"
)
words = tokenize_raw_words(raw_words)
print(words)

words = tokenize_raw_words(text)
print(words)
