# recsys_spt2018
add challenge set and mdp/data containing data files to main directory//
How to build recommendations:
1) run build_structures.py
2) run full_kernel_songs.py
3) run create_Ku.py with parameters 1 and 0.5
4) run words_similarity_builder.py
5) run create_pl2title.py
6) run create_Pu.py
7) run titles_similarity_0.py
8) run user_based_MSD_1.py
9) run item_based_MSD_5_10_25.py with parameters 0.7 0.4 5
10) run item_based_MSD_5_10_25.py with parameters 0.7 0.4 10
11) run item_based_MSD_5_10_25.py with parameters 0.7 0.4 25
12) run selected_KOMD_100.py with parameters 100 50000
