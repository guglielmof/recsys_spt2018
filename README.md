# recsys_spt2018

Environment setting
* Put data in mdp/data inside main directory</br>
* Put challenge set in main directory</br>

How to build recommendations:
1) run build_structures.py
2) run full_kernel_songs.py
3) run create_Ku.py with parameters 1 and 0.5
4) run create_pl2title.py
5) run words_similarity_builder.py
6) run calc_P.py
7) run titles_similarity_0.py
8) run user_based_MSD_1.py
9) run item_based_MSD_5_10_25.py with parameters 0.7 0.4 5
10) run item_based_MSD_5_10_25.py with parameters 0.7 0.4 10
11) run item_based_MSD_5_10_25.py with parameters 0.7 0.4 25
12) run selected_KOMD_100.py with parameters 100 50000
13) run merge_csv.sh 

Estimated memory: 80Gb</br>
Estimated time: 20 hours</br>
Estimated disk space 150Gb</br>


Most of the steps above can be parallelized as shown in the following figure. The estimated time is calculated considering a sequential execution.

![alt text](https://raw.githubusercontent.com/guglielmof/recsys_spt2018/master/schema.png)
 
