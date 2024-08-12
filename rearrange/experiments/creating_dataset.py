from tasks import (
    consolidate_procthor_traincustom, 
    make_valid_houses_file_cutsom,
    make_procthor_mini_traincustom,
    split_data_custom
)

if __name__ == "__main__":
    #consolidate_procthor_traincustom()
    load_folder = '2022procthor_new'
    make_valid_houses_file_cutsom(1000, 'mini_train', load_folder)
    #make_procthor_mini_traincustom()
    #split_data_custom(mode='train', input_name='mini_train_consolidated')