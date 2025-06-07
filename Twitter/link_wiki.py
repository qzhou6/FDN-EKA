import pickle
# from common_function import filter_words,create_entity_description_dict,split_list_in_half,merge_dictionaries



# with open('/home/zhouqing/codes/COOLANT/idea02/data/eneity_data/all_nouns.pkl', 'rb') as f:
#     all_nouns = pickle.load(f)
    

# filter_nouns = filter_words(all_nouns)

# list1, list2 = split_list_in_half(filter_nouns)


# entity_description_dict1 = create_entity_description_dict(list1)

# with open('/home/zhouqing/codes/COOLANT/idea02/data/eneity_data/entity_description_dict1.pkl', 'wb') as f:
#     pickle.dump(entity_description_dict1, f)
# print("entity_description_dict1 已保存到文件 entity_description_dict1.pkl 中")

# entity_description_dict2 = create_entity_description_dict(list2)
# with open('/home/zhouqing/codes/COOLANT/idea02/data/eneity_data/entity_description_dict2.pkl', 'wb') as f:
#     pickle.dump(entity_description_dict2, f)
# print("entity_description_dict2 已保存到文件 entity_description_dict2.pkl 中")


# entity_description_dict = merge_dictionaries(entity_description_dict1, entity_description_dict2)

# with open('/home/zhouqing/codes/COOLANT/idea02/data/eneity_data/entity_description_dict.pkl', 'wb') as f:
#     pickle.dump(entity_description_dict, f)
# print("entity_description_dict 已保存到文件 entity_description_dict.pkl 中")



with open('/home/zhouqing/codes/COOLANT/idea02/data/eneity_data/entity_description_dict.pkl', 'rb') as f:
    entity_description_dict = pickle.load(f)







