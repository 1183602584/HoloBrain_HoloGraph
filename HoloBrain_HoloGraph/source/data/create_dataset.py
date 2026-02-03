from source.data.dataset import *

def create_dataset(data):
    
    if data == "HCP-A":
        dataset = HCPA_BoldSCDataset(
            bold_dir='./HCP-A-SC_FC/AAL_116/BOLD', 
            sc_dir='./HCP-A-SC_FC/ALL_SC'
        )
    elif data == "HCP-YA-WM":
        dataset = HCP_YA_SCDataset(
            bold_dir='./HCP-YA-SC_FC/Brainnetome_264/BOLD_interpolated', 
            sc_dir='./HCP-YA-SC_FC/HCP-YA-SC',
            label_path='../LR_label/LR_label.csv',
            scan="LR"
        )
    elif data == "HCP-YA-WM-RL":
        dataset = HCP_YA_SCDataset(
            bold_dir='./HCP-YA-SC_FC/Brainnetome_264/BOLD_interpolated', 
            sc_dir='./HCP-YA-SC_FC/HCP-YA-SC',
            label_path='../RL_label/RL_label.csv',
            scan="RL"
        )
    elif data == "HCP-YA":
        dataset = HCPYA_BoldSCDataset(
            bold_dir='./HCP-YA-SC_FC/AAL_116/BOLD', 
            sc_dir='./HCP-YA-SC_FC/HCP-YA-SC'
        )
    elif data == "HCP-YA-region":
        dataset = HCPYA_byregion(
            bold_dir='./HCP-YA-SC_FC/AAL_116/BOLD', 
            label_path='./region_label.txt'
        )
    elif data == "HCPA-region":
        dataset = HCPA_byregion(
            bold_dir='./HCP-A-SC_FC/AAL_116/BOLD', 
            label_path='./region_label.txt'
        )
        
        
    elif data == "ABIDE":
        dataset = ABIDE_BoldFCDataset(
            ts_dir="/home/hezhenkun/nilearn_data/ABIDE_pcp/cpac/nofilt_noglobal",
            phenotypic_csv="/home/hezhenkun/nilearn_data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv",
            atlas="aal",
            expected_n=116,
            fix_len=175,
            k=15,
            use_abs=True,
            cache_fc_path="/home/hezhenkun/nilearn_data/abide_fc_cache2.npy",  # 可选：强烈建议开
        )
    return dataset