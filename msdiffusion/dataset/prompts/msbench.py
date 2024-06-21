data_class_dict = {
    "backpack": "backpack",
    "cat": "cat",
    "cat2": "cat",
    "clock": "clock",
    "colorful_sneaker": "sneaker",
    "dog": "dog",
    "dog2": "dog",
    "dog5": "dog",
    "dog6": "dog",
    "dog7": "dog",
    "dog8": "dog",
    "duck_toy": "toy",
    "grey_sloth_plushie": "stuffed animal",
    "teapot": "teapot",
    "wolf_plushie": "stuffed animal",
    "cat1": "cat",
    "dog1": "dog",
    "lantern": "lantern",
    "flower": "flower",
    "teddybear": "stuffed animal",
    "forest": "forest",
    "mountain": "mountain",
    "barn": "barn",
    "waterfall": "waterfall",
    "cap": "cap",
    "cap1": "cap",
    "hat": "hat",
    "hat1": "hat",
    "hat2": "hat",
    "dress": "dress",
    "dress1": "dress",
    "shorts1": "shorts",
    "trousers": "trousers",
    "jeans": "jeans",
    "coat": "coat",
    "jacket": "jacket",
    "t-shirt": "t-shirt",
    "shirt": "shirt",
    "shirt1": "shirt",
    "sweater": "sweater",
}

two_subject_combinations = ["living_living", "living_object", "object_object", "living_upwearing", "living_midwearing",
                            "living_wholewearing", "midwearing_downwearing", "living_scene", "object_scene"]
three_subject_combinations = ["living_living_living", "object_object_object", "living_object_scene",
                              "upwearing_midwearing_downwearing"]

subject_combination_boxes = {
    "living_living": [[0., 0.25, 0.5, 0.75], [0.5, 0.25, 1., 0.75]],
    "living_object": [[0., 0.25, 0.5, 0.75], [0.5, 0.25, 1., 0.75]],
    "object_object": [[0., 0.25, 0.5, 0.75], [0.5, 0.25, 1., 0.75]],
    "living_upwearing": [[0.25, 0.25, 0.75, 0.75], [0.25, 0., 0.75, 0.25]],
    "living_midwearing": [[0.25, 0.25, 0.75, 0.75], [0.25, 0.25, 0.75, 0.75]],
    "living_wholewearing": [[0.25, 0.25, 0.75, 0.75], [0.25, 0.25, 0.75, 1.]],
    "midwearing_downwearing": [[0.25, 0.25, 0.75, 0.6], [0.25, 0.6, 0.75, 1.]],
    "living_scene": [[0.25, 0.25, 0.75, 0.75], [0., 0., 1., 1.]],
    "object_scene": [[0.25, 0.25, 0.75, 0.75], [0., 0., 1., 1.]],
    "living_living_living": [[0., 0.25, 0.35, 0.75], [0.35, 0.25, 0.65, 0.75], [0.65, 0.25, 1., 0.75]],
    "object_object_object": [[0., 0.25, 0.35, 0.75], [0.35, 0.25, 0.65, 0.75], [0.65, 0.25, 1., 0.75]],
    "living_object_scene": [[0., 0.25, 0.5, 0.75], [0.5, 0.25, 1., 0.75], [0., 0., 1., 1.]],
    "upwearing_midwearing_downwearing": [[0.25, 0., 0.75, 0.25], [0.25, 0.25, 0.75, 0.6], [0.25, 0.6, 0.75, 1.]],
}

normal_prompts_two_subjects = [
    "a {0} and a {1} in a room",
    "a {0} and a {1} in the snow",
    "a {0} and a {1} in the jungle",
    "a {0} and a {1} on the beach",
    "a {0} and a {1} on the grass",
    "a {0} and a {1} on a cobblestone street",
]

normal_prompts_three_subjects = [
    "a {0}, a {1}, and a {2} in a room",
    "a {0}, a {1}, and a {2} in the snow",
    "a {0}, a {1}, and a {2} in the jungle",
    "a {0}, a {1}, and a {2} on the beach",
    "a {0}, a {1}, and a {2} on the grass",
    "a {0}, a {1}, and a {2} on a cobblestone street",
]

scene_prompts_two_subjects = [
    "a {0} with a {1} in the background",
]

scene_prompts_three_subjects = [
    "a {0} and {1} with a {2} in the background",
]

wearing_prompts_two_subjects_0 = [
    "a {0} wearing a {1} in a room",
    "a {0} wearing a {1} in the snow",
    "a {0} wearing a {1} in the jungle",
    "a {0} wearing a {1} on the beach",
    "a {0} wearing a {1} on the grass",
    "a {0} wearing a {1} on a cobblestone street",
]

wearing_prompts_two_subjects_1 = [
    "a woman wearing a {0} and {1} in a room",
    "a woman wearing a {0} and {1} in the snow",
    "a woman wearing a {0} and {1} in the jungle",
    "a woman wearing a {0} and {1} on the beach",
    "a woman wearing a {0} and {1} on the grass",
    "a woman wearing a {0} and {1} on a cobblestone street"
]

wearing_prompts_three_subjects = [
    "a woman wearing a {0}, a {1}, and a {2} in a room",
    "a woman wearing a {0}, a {1}, and a {2} in the snow",
    "a woman wearing a {0}, a {1}, and a {2} in the jungle",
    "a woman wearing a {0}, a {1}, and a {2} on the beach",
    "a woman wearing a {0}, a {1}, and a {2} on the grass",
    "a woman wearing a {0}, a {1}, and a {2} on a cobblestone street"
]

subject_combination_prompts = {
    "living_living": normal_prompts_two_subjects,
    "living_object": normal_prompts_two_subjects,
    "object_object": normal_prompts_two_subjects,
    "living_upwearing": wearing_prompts_two_subjects_0,
    "living_midwearing": wearing_prompts_two_subjects_0,
    "living_wholewearing": wearing_prompts_two_subjects_0,
    "midwearing_downwearing": wearing_prompts_two_subjects_1,
    "living_scene": scene_prompts_two_subjects,
    "object_scene": scene_prompts_two_subjects,
    "living_living_living": normal_prompts_three_subjects,
    "object_object_object": normal_prompts_three_subjects,
    "living_object_scene": scene_prompts_three_subjects,
    "upwearing_midwearing_downwearing": wearing_prompts_three_subjects,
}
