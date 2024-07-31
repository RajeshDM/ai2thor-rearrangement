
import os
import compress_pickle
import prior
#:from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR

def install_procthor_dataset(revision="2022procthor", skip_consolidate=False):

    all_data = prior.load_dataset("rearrangement_episodes", revision=revision)

    for partition in ["val", "train"]:
        output_partition = f"mini_{partition}" if partition in ["val"] else partition

        print(f"{output_partition}...")

        num_episodes = 0

        '''
        current_dir = os.path.join(
            ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR,
            "data",
            "2022procthor",
            f"split_{output_partition}",
        )
        os.makedirs(current_dir, exist_ok=True)
        '''

        consolidated_data = {}

        for part, compressed_part_data in all_data[partition]:
            print(f"{part}")

            if not skip_consolidate:
                # each part is a compressed_pickle
                cur_data = compress_pickle.loads(
                    data=compressed_part_data, compression="gzip"
                )

                for scene in cur_data:
                    num_episodes += len(cur_data[scene])

                consolidated_data.update(cur_data)

            #with open(os.path.join(current_dir, f"{part}.pkl.gz"), "wb") as f:
            #    f.write(compressed_part_data)

        if not skip_consolidate:
            print(f"{output_partition}_consolidated")
            consolidated_file = os.path.join(
                ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR,
                "data",
                "2022procthor",
                f"{output_partition}_consolidated.pkl.gz",
            )
            '''
            compress_pickle.dump(
                obj=consolidated_data,
                path=consolidated_file,
                pickler_kwargs={"protocol": 4},  # Backwards compatible with python 3.6
            )
            '''

            print(
                f"{len(consolidated_data)} scenes and total {num_episodes} episodes for {output_partition}"
            )

    #print("Creating mini val houses file")
    #make_valid_houses_file(ctx, verbose=False)

    print("DONE")

def load_dataset(dataset_main, revision):

    if revision is None:
        return prior.load_dataset(dataset_main)

    return prior.load_dataset(dataset_main, revision)

def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified, same

def compare_datasets():
    # Load the datasets
    #data1 = load_dataset('procthor-10k', 'ab3cacd0fc17754d4c080a3fd50b18395fae8647')
    data1 = load_dataset('procthor-10k', None)
    data2 = load_dataset('procthor-10k', 'rearrangement-2022')
    #data3 = load_dataset('rearrangement_episodes', '2022procthor')

    split = 'train'
    base_and_proc_re_not_same = 0 
    base_and_re_proc_not_same = 0
    re_proc_and_proc_re_not_same = 0
    total = 0

    #for house_base, house_proc_re, house_re_proc in zip(data1[split], data2[split], data3[split]):
    for house_base, house_proc_re in zip(data1[split], data2[split]):

        #print (house_base)
        if house_base != house_proc_re :
            base_and_proc_re_not_same += 1
        #if house_base != house_re_proc :
        #    base_and_re_proc_not_same += 1
        #if house_proc_re != house_re_proc :
        #    re_proc_and_proc_re_not_same += 1 

        #print ("base house: ", house_base)
        #print ("proc_re house: ", house_proc_re)
        #print ("re_proc house: ", house_re_proc)

        added, removed, modified, same = dict_compare(house_base, house_proc_re)

        print ("Added: ", added)
        print ("Removed: ", removed)
        print ("Modified: ", modified)
        print ("Same: ", same)

        break

        total += 1

        if total > 2 :
            break

    print ("Base and Proc Re not same: ", base_and_proc_re_not_same)
    #print ("Base and Re Proc not same: ", base_and_re_proc_not_same)
    #print ("Re Proc and Proc Re not same: ", re_proc_and_proc_re_not_same)

    print ("Total: ", total)

if __name__ == "__main__":
    compare_datasets()
    #install_procthor_dataset()