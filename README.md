# recsys_spt2018
How to build recommendations:
1) run build_structures.py in order to convert ds into a more menagable format. you can specify where to find original data and where to save new structures.
2) run full_kernel_songs.py to build a kernel used in selected_KOMD_100.py. you shuld rename directories according to names given in previous file
3) run create_Ku.py specifing new directories' paths.
4) in utils_matrix modify load_test_set, by specifing the directory where to find the challenge set.
5) run words_similarity_builder to create S_titles.npy used in selected_KOMD_100.py and title_s_similaarity_0.py
6) change paths in titles_similarity_0.py, user_based_MSD_1.py, item_based_MSD_5_10_25.py, selected_KOMD_100.pyand run them.
