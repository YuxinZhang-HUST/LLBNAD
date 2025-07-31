import os
import statistics
import pandas as pd

from util.aggregate_csv import read_csv, aggregate_csv

def gen_chapter2_results():
    #### Most Important Information

    ############## Chapter2: Zero-shot -- AdaCLIP
    sub_dir = [
        ### result subdir, method name, csv name
        # ['csvs', 'ViT-L-14-336-None-None-D0-L0', 'ViT-L-14-336-None-None-D0-L0'],
        ['csvs', 'ViT-L-14-336-SD-VL-D4-L5', 'ViT-L-14-336-SD-VL-D4-L5'],
        # ['csvs', 'ViT-L-14-336-None-VL-D4-L5', 'ViT-L-14-336-None-VL-D4-L5'],
        ['csvs', 'ViT-L-14-336-S-VL-D4-L5', 'ViT-L-14-336-S-VL-D4-L5'],
        # ['csvs', 'ViT-L-14-336-D-VL-D4-L5', 'ViT-L-14-336-D-VL-D4-L5'],
        ['csvs', 'ViT-L-14-336-SD-VL-D4-L5', 'ViT-L-14-336-SD-VL-D4-L5'],
        ['csvs', 'ViT-L-14-336-SD-None-D4-L5', 'ViT-L-14-336-SD-None-D4-L5'],
        ['csvs', 'ViT-L-14-336-SD-V-D4-L5', 'ViT-L-14-336-SD-V-D4-L5'],
        ['csvs', 'ViT-L-14-336-SD-L-D4-L5', 'ViT-L-14-336-SD-L-D4-L5'],
        ['csvs', 'ViT-L-14-336-SD-VL-D4-L5', 'ViT-L-14-336-SD-VL-D4-L5'],
        ['csvs', 'ViT-B-16-SD-VL-D4-L5', 'ViT-B-16-SD-VL-D4-L5'],
        ['csvs', 'ViT-B-32-SD-VL-D4-L5', 'ViT-B-32-SD-VL-D4-L5'],
        # ['csvs', 'ViT-L-14-SD-VL-D4-L5', 'ViT-L-14-SD-VL-D4-L5'],
        ['csvs', 'ViT-L-14-336-SD-VL-D4-L5', 'ViT-L-14-336-SD-VL-D4-L5'],
        ['csvs', 'ViT-L-14-336-SD-VL-D1-L1', 'ViT-L-14-336-SD-VL-D1-L1'],
        ['csvs', 'ViT-L-14-336-SD-VL-D1-L2', 'ViT-L-14-336-SD-VL-D1-L2'],
        ['csvs', 'ViT-L-14-336-SD-VL-D1-L3', 'ViT-L-14-336-SD-VL-D1-L3'],
        ['csvs', 'ViT-L-14-336-SD-VL-D1-L4', 'ViT-L-14-336-SD-VL-D1-L4'],
        ['csvs', 'ViT-L-14-336-SD-VL-D1-L5', 'ViT-L-14-336-SD-VL-D1-L5'],
        ['csvs', 'ViT-L-14-336-SD-VL-D1-L6', 'ViT-L-14-336-SD-VL-D1-L6'],
        ['csvs', 'ViT-L-14-336-SD-VL-D2-L1', 'ViT-L-14-336-SD-VL-D2-L1'],
        ['csvs', 'ViT-L-14-336-SD-VL-D2-L2', 'ViT-L-14-336-SD-VL-D2-L2'],
        ['csvs', 'ViT-L-14-336-SD-VL-D2-L3', 'ViT-L-14-336-SD-VL-D2-L3'],
        ['csvs', 'ViT-L-14-336-SD-VL-D2-L4', 'ViT-L-14-336-SD-VL-D2-L4'],
        ['csvs', 'ViT-L-14-336-SD-VL-D2-L5', 'ViT-L-14-336-SD-VL-D2-L5'],
        ['csvs', 'ViT-L-14-336-SD-VL-D2-L6', 'ViT-L-14-336-SD-VL-D2-L6'],
        ['csvs', 'ViT-L-14-336-SD-VL-D3-L1', 'ViT-L-14-336-SD-VL-D3-L1'],
        ['csvs', 'ViT-L-14-336-SD-VL-D3-L2', 'ViT-L-14-336-SD-VL-D3-L2'],
        ['csvs', 'ViT-L-14-336-SD-VL-D3-L3', 'ViT-L-14-336-SD-VL-D3-L3'],
        ['csvs', 'ViT-L-14-336-SD-VL-D3-L4', 'ViT-L-14-336-SD-VL-D3-L4'],
        ['csvs', 'ViT-L-14-336-SD-VL-D3-L5', 'ViT-L-14-336-SD-VL-D3-L5'],
        ['csvs', 'ViT-L-14-336-SD-VL-D3-L6', 'ViT-L-14-336-SD-VL-D3-L6'],
        ['csvs', 'ViT-L-14-336-SD-VL-D4-L1', 'ViT-L-14-336-SD-VL-D4-L1'],
        ['csvs', 'ViT-L-14-336-SD-VL-D4-L2', 'ViT-L-14-336-SD-VL-D4-L2'],
        ['csvs', 'ViT-L-14-336-SD-VL-D4-L3', 'ViT-L-14-336-SD-VL-D4-L3'],
        ['csvs', 'ViT-L-14-336-SD-VL-D4-L4', 'ViT-L-14-336-SD-VL-D4-L4'],
        ['csvs', 'ViT-L-14-336-SD-VL-D4-L5', 'ViT-L-14-336-SD-VL-D4-L5'],
        ['csvs', 'ViT-L-14-336-SD-VL-D4-L6', 'ViT-L-14-336-SD-VL-D4-L6'],
        ['csvs', 'ViT-L-14-336-SD-VL-D5-L1', 'ViT-L-14-336-SD-VL-D5-L1'],
        ['csvs', 'ViT-L-14-336-SD-VL-D5-L2', 'ViT-L-14-336-SD-VL-D5-L2'],
        ['csvs', 'ViT-L-14-336-SD-VL-D5-L3', 'ViT-L-14-336-SD-VL-D5-L3'],
        ['csvs', 'ViT-L-14-336-SD-VL-D5-L4', 'ViT-L-14-336-SD-VL-D5-L4'],
        ['csvs', 'ViT-L-14-336-SD-VL-D5-L5', 'ViT-L-14-336-SD-VL-D5-L5'],
        ['csvs', 'ViT-L-14-336-SD-VL-D5-L6', 'ViT-L-14-336-SD-VL-D5-L6'],
        ['csvs', 'ViT-L-14-336-SD-VL-D6-L1', 'ViT-L-14-336-SD-VL-D6-L1'],
        ['csvs', 'ViT-L-14-336-SD-VL-D6-L2', 'ViT-L-14-336-SD-VL-D6-L2'],
        ['csvs', 'ViT-L-14-336-SD-VL-D6-L3', 'ViT-L-14-336-SD-VL-D6-L3'],
        ['csvs', 'ViT-L-14-336-SD-VL-D6-L4', 'ViT-L-14-336-SD-VL-D6-L4'],
        ['csvs', 'ViT-L-14-336-SD-VL-D6-L5', 'ViT-L-14-336-SD-VL-D6-L5'],
        ['csvs', 'ViT-L-14-336-SD-VL-D6-L6', 'ViT-L-14-336-SD-VL-D6-L6'],

    ]

    return_average = True

    ##### for mvtec
    ######## data for read
    result_root = '../AdaCLIP/workspaces'

    name_formatting = '0s-pretrained-visa-{}-WO-HSF-{}.csv'

    dataset_list = [
        'mvtec',
        # 'visa',
        # 'car_cover',
        # 'phone_cover',
    ]

    required_metrics = [
        'auroc_im', 'f1_im',
        'auroc_px', 'f1_px'
    ]

    ######## data save
    save_root = './experiments'
    os.makedirs(save_root, exist_ok=True)

    dfs = {}
    for _sub_dir in sub_dir:
        csv_full_path = os.path.join(save_root, _sub_dir[2] + '.csv')
        df = read_csv(result_root, _sub_dir, dataset_list, name_formatting, required_metrics, return_average)
        if df is not None:
            df.to_csv(csv_full_path, header=True, float_format='%.2f')
            dfs[_sub_dir[2]] = df
            print(f'Aggregate csv fils to {csv_full_path}')

    #################  aggregation
    save_agg_root = './experiments/agg'
    save_agg_name = 'adaclip_ablation_mvtec'
    os.makedirs(save_agg_root, exist_ok=True)

    aggregate_csv(dfs, required_metrics, save_agg_root, save_agg_name)

    ##### for visa
    ######## data for read
    result_root = '../AdaCLIP/workspaces'

    name_formatting = '0s-pretrained-mvtec-{}-WO-HSF-{}.csv'

    dataset_list = [
        # 'mvtec',
        'visa',
        # 'car_cover',
        # 'phone_cover',
    ]

    required_metrics = [
        'auroc_im', 'f1_im',
        'auroc_px', 'f1_px'
    ]

    ######## data save
    save_root = './experiments'
    os.makedirs(save_root, exist_ok=True)

    dfs = {}
    for _sub_dir in sub_dir:
        csv_full_path = os.path.join(save_root, _sub_dir[2] + '.csv')
        df = read_csv(result_root, _sub_dir, dataset_list, name_formatting, required_metrics, return_average)
        if df is not None:
            df.to_csv(csv_full_path, header=True, float_format='%.2f')
            dfs[_sub_dir[2]] = df
            print(f'Aggregate csv fils to {csv_full_path}')

    #################  aggregation
    save_agg_root = './experiments/agg'
    save_agg_name = 'adaclip_ablation_visa'
    os.makedirs(save_agg_root, exist_ok=True)

    aggregate_csv(dfs, required_metrics, save_agg_root, save_agg_name)

def gen_chapter3_results():
    #### Most Important Information

    ############## Chapter3: Unsupervised -- IKD
    sub_dir = [
        ### result subdir, method name, csv name
        # ['AprilGANTrainer_configs_benchmark_april_gan_april_gan_256_5e_exp1', 'april_gan'],
        # ['WinCLIPTrainer_configs_benchmark_winclip_winclip_256_exp1', 'winclip', 'winclip'],
        ['CFATrainer_configs_benchmark_cfa_cfa_256_100e_exp1', 'cfa', 'cfa'],
        ['PatchCoreTrainer_configs_benchmark_patchcore_patchcore_256_exp1', 'patchcore', 'patchcore'],
        ['PyramidFlowTrainer_configs_benchmark_pyramidflow_pyramidflow_256_100e_exp1', 'pyramidflow', 'pyramidflow'],
        ['RDTrainer_configs_benchmark_rd_rd_256_100e_exp1', 'rd', 'rd'],
        ['SimpleNetTrainer_configs_benchmark_simplenet_simplenet_256_100e_exp1', 'simplenet', 'simplenet'],
        ['STFPMTrainer_configs_benchmark_stfpm_stfpm_256_50e_exp1', 'stfpm', 'stfpm'],
        ['UniADTrainer_configs_benchmark_uniad_uniad_256_100e_exp1', 'uniad', 'uniad'],

        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet32_beta2_Lpcs_exp1', 'ikd', 'ikd_mshrnet32_beta2_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet18_beta2_L2_exp1', 'ikd', 'ikd_mshrnet18_beta2_L2'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet18_beta2_Lps_exp1', 'ikd', 'ikd_mshrnet18_beta2_Lps'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet18_beta2_Lpcs_exp1', 'ikd', 'ikd_mshrnet18_beta2_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet32_beta2_L2_exp1', 'ikd', 'ikd_mshrnet32_beta2_L2'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet32_beta2_Lps_exp1', 'ikd', 'ikd_mshrnet32_beta2_Lps'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet32_beta2_Lpcs_exp1', 'ikd', 'ikd_mshrnet32_beta2_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet48_beta2_L2_exp1', 'ikd', 'ikd_mshrnet48_beta2_L2'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet48_beta2_Lps_exp1', 'ikd', 'ikd_mshrnet48_beta2_Lps'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet48_beta2_Lpcs_exp1', 'ikd', 'ikd_mshrnet48_beta2_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_timm_resnet18_beta2_L2_exp1', 'ikd', 'ikd_timm_resnet18_beta2_L2'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_timm_resnet18_beta2_Lps_exp1', 'ikd', 'ikd_timm_resnet18_beta2_Lps'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_timm_resnet18_beta2_Lpcs_exp1', 'ikd', 'ikd_timm_resnet18_beta2_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_timm_resnet34_beta2_L2_exp1', 'ikd', 'ikd_timm_resnet34_beta2_L2'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_timm_resnet34_beta2_Lps_exp1', 'ikd', 'ikd_timm_resnet34_beta2_Lps'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_timm_resnet34_beta2_Lpcs_exp1', 'ikd', 'ikd_timm_resnet34_beta2_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_timm_resnet50_beta2_L2_exp1', 'ikd', 'ikd_timm_resnet50_beta2_L2'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_timm_resnet50_beta2_Lps_exp1', 'ikd', 'ikd_timm_resnet50_beta2_Lps'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_timm_resnet50_beta2_Lpcs_exp1', 'ikd', 'ikd_timm_resnet50_beta2_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_timm_wide_resnet50_2_beta2_L2_exp1', 'ikd', 'ikd_timm_wide_resnet50_2_beta2_L2'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_timm_wide_resnet50_2_beta2_Lps_exp1', 'ikd', 'ikd_timm_wide_resnet50_2_beta2_Lps'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_timm_wide_resnet50_2_beta2_Lpcs_exp1', 'ikd', 'ikd_timm_wide_resnet50_2_beta2_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet32_beta-3_Lpcs_exp1', 'ikd', 'ikd_mshrnet32_beta-3_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet32_beta-2_Lpcs_exp1', 'ikd', 'ikd_mshrnet32_beta-2_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet32_beta-1_Lpcs_exp1', 'ikd', 'ikd_mshrnet32_beta-1_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet32_beta0_Lpcs_exp1', 'ikd', 'ikd_mshrnet32_beta0_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet32_beta1_Lpcs_exp1', 'ikd', 'ikd_mshrnet32_beta1_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet32_beta2_Lpcs_exp1', 'ikd', 'ikd_mshrnet32_beta2_Lpcs'],
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet32_beta3_Lpcs_exp1', 'ikd', 'ikd_mshrnet32_beta3_Lpcs'],
    ]

    return_average = False

    ######## data for read
    result_root = './runs'

    name_formatting = '{:}_{:}_last.csv'
    # name_formatting = '{:}_{:}_best.csv'


    dataset_list = [
        'mvtec',
        'visa',
        # 'car_cover',
        # 'phone_cover',
    ]

    # required_metrics = [
    #     'mAUROC_sp_max', 'mAP_sp_max', 'mF1_max_sp_max',
    #     'mAUROC_px', 'mAP_px', 'mF1_max_px', 'mAUPRO_px'
    # ]

    required_metrics = [
        'mAUROC_sp_max','mF1_max_sp_max',
        'mAUROC_px', 'mF1_max_px'
    ]

    ######## data save
    save_root = './experiments'
    os.makedirs(save_root, exist_ok=True)

    dfs = {}
    for _sub_dir in sub_dir:
        csv_full_path = os.path.join(save_root, _sub_dir[2] + '.csv')
        df = read_csv(result_root, _sub_dir, dataset_list, name_formatting, required_metrics, return_average)
        if df is not  None:
            df.to_csv(csv_full_path, header=True, float_format='%.2f')
            dfs[_sub_dir[2]] = df
            print(f'Aggregate csv fils to {csv_full_path}')

    #################  aggregation
    save_agg_root = './experiments/agg'
    save_agg_name = 'ikd_ablation'
    os.makedirs(save_agg_root, exist_ok=True)

    aggregate_csv(dfs, required_metrics, save_agg_root, save_agg_name)



def gen_chapter4_results():
    #### Most Important Information

    ############## Chapter3: Unsupervised -- IKD
    sub_dir = [
        ### result subdir, method name, csv name
        # ['AprilGANTrainer_configs_benchmark_april_gan_april_gan_256_5e_exp1', 'april_gan'],
        # ['WinCLIPTrainer_configs_benchmark_winclip_winclip_256_exp1', 'winclip', 'winclip'],
        ['DevNetTrainer_configs_benchmark_devnet_devnet_semi1_exp1', 'devnet', 'devnet_semi1'],
        ['DevNetTrainer_configs_benchmark_devnet_devnet_semi5_exp1', 'devnet', 'devnet_semi5'],
        ['DevNetTrainer_configs_benchmark_devnet_devnet_semi10_exp1', 'devnet', 'devnet_semi10'],

        ['SemiHRNetTrainer_configs_benchmark_semihrnet_semihrnet_semi1_exp1', 'semi_hrnet', 'semihrnet_semi1'],
        ['SemiHRNetTrainer_configs_benchmark_semihrnet_semihrnet_semi5_exp1', 'semi_hrnet', 'semihrnet_semi5'],
        ['SemiHRNetTrainer_configs_benchmark_semihrnet_semihrnet_semi10_exp1', 'semi_hrnet', 'semihrnet_semi10'],

        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_unsupervised_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_unsupervised_mshrnet32_gaTrue_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi1_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi1_mshrnet32_gaTrue_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi5_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi5_mshrnet32_gaTrue_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaFalse_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaFalse_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomFalse_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomFalse_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma0_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma0_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma0_5_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma0.5_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma1_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma1_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma1_5_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma1.5_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma2_5_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma2.5_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma3_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma3_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma3_5_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma3.5_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma4_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma4_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma2_L2_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma2_lossL2'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet18_semi10_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet18_gaTrue_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet32_gaTrue_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet48_semi10_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10_mshrnet48_gaTrue_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_timm_resnet18_semi10_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10_timm_resnet18_gaTrue_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_timm_resnet34_semi10_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10_timm_resnet34_gaTrue_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_timm_resnet50_semi10_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10_timm_resnet50_gaTrue_oomTrue_gamma2_lossLcdo'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_timm_wide_resnet50_2_semi10_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10_timm_wide_resnet50_2_gaTrue_oomTrue_gamma2_lossLcdo'],

    ]

    return_average = True

    ######## data for read
    result_root = './runs'

    name_formatting = '{:}_{:}_last.csv'
    # name_formatting = '{:}_{:}_best.csv'


    dataset_list = [
        'mvtec',
        'visa',
        # 'car_cover',
        # 'phone_cover',
    ]

    # required_metrics = [
    #     'mAUROC_sp_max', 'mAP_sp_max', 'mF1_max_sp_max',
    #     'mAUROC_px', 'mAP_px', 'mF1_max_px', 'mAUPRO_px'
    # ]

    required_metrics = [
        'mAUROC_sp_max','mF1_max_sp_max',
        'mAUROC_px', 'mF1_max_px'
    ]

    ######## data save
    save_root = './experiments'
    os.makedirs(save_root, exist_ok=True)

    dfs = {}
    for _sub_dir in sub_dir:
        csv_full_path = os.path.join(save_root, _sub_dir[2] + '.csv')
        df = read_csv(result_root, _sub_dir, dataset_list, name_formatting, required_metrics, return_average)
        if df is not  None:
            df.to_csv(csv_full_path, header=True, float_format='%.2f')
            dfs[_sub_dir[2]] = df
            print(f'Aggregate csv fils to {csv_full_path}')

    #################  aggregation
    save_agg_root = './experiments/agg'
    save_agg_name = 'cdo_ablation'
    os.makedirs(save_agg_root, exist_ok=True)

    aggregate_csv(dfs, required_metrics, save_agg_root, save_agg_name)



def gen_real_case_results():
    #### Most Important Information

    ############## Chapter3: Unsupervised -- IKD
    sub_dir = [
        ### result subdir, method name, csv name
        ['IKDTrainer_configs_benchmark_ikd_ikd_mshrnet32_beta0_Lps_exp1', 'ikd',
         'ikd'],

        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi1_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi1'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi5_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi5'],
        ['CDOTrainer_configs_benchmark_cdo_cdo_mshrnet32_semi10_gaTrue_oomTrue_gamma2_Lcdo_exp1', 'cdo',
         'cdo_semi10'],
    ]

    return_average = False

    ######## data for read
    result_root = './runs'

    # name_formatting = '{:}_{:}_last.csv'
    name_formatting = '{:}_{:}_best.csv'


    dataset_list = [
        # 'mvtec',
        'real_case',
        # 'car_cover',
        # 'phone_cover',
    ]

    # required_metrics = [
    #     'mAUROC_sp_max', 'mAP_sp_max', 'mF1_max_sp_max',
    #     'mAUROC_px', 'mAP_px', 'mF1_max_px', 'mAUPRO_px'
    # ]

    required_metrics = [
        'mAUROC_sp_max','mF1_max_sp_max',
        'mAUROC_px', 'mF1_max_px'
    ]

    ######## data save
    save_root = './experiments'
    os.makedirs(save_root, exist_ok=True)

    dfs = {}
    for _sub_dir in sub_dir:
        csv_full_path = os.path.join(save_root, _sub_dir[2] + '.csv')
        df = read_csv(result_root, _sub_dir, dataset_list, name_formatting, required_metrics, return_average)
        if df is not  None:
            df.to_csv(csv_full_path, header=True, float_format='%.2f')
            dfs[_sub_dir[2]] = df
            print(f'Aggregate csv fils to {csv_full_path}')

    #################  aggregation
    save_agg_root = './experiments/agg'
    save_agg_name = 'real_case'
    os.makedirs(save_agg_root, exist_ok=True)

    aggregate_csv(dfs, required_metrics, save_agg_root, save_agg_name)

# 计算每个setting下的各个指标I-AUROC...
def gen_FUIAD_results():
    sub_dir = [
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num7_iter1000_meanscores"
    ]

    return_average = False

    ######## data for read
    result_root = '/home/user/zyx/HUST-ADer4YX_FUIAD/runs'
    result_data = []
    for _sub_dir in sub_dir:
        file_path = os.path.join(result_root, _sub_dir)
        for filename in os.listdir(file_path):
            if filename.endswith("best.csv"):
                file_path = os.path.join(result_root, _sub_dir, filename)
                data = pd.read_csv(file_path)
                print(f"成功读取文件：{file_path}")
                print(data.head())

        # # 处理数据
        # new_df = pd.DataFrame()
        # new_df["file_path"] = [_sub_dir]
        # mean_values = data.iloc[:, 1:].mean(axis=1)
        # new_df["columns_mean"] = mean_values.mean()  # 所有列的整体平均值
        # result_data.append(new_df)

        # 计算每列平均值
        col_means = data.iloc[:, 1:].mean(axis=0)
        # 构建结果：文件路径 + 各列均值
        result_row = pd.DataFrame({"file_path": [_sub_dir]})
        result_row = pd.concat([result_row, pd.DataFrame(col_means).T], axis=1)
        result_data.append(result_row)

    if result_data:
        final_df = pd.concat(result_data, ignore_index=True)
        final_df.to_csv("processed_columns_means.csv", index=False)
        print("处理完成，结果已保存至 processed_columns_means.csv")
    else:
        print("未找到可处理的 CSV 文件")
    # 合并结果并保存
    if result_data:
        final_df = pd.concat(result_data, ignore_index=True)
        final_df.to_csv("/home/user/zyx/HUST-ADer4YX_FUIAD/experiments/result_mean.csv", index=False)
        print("处理完成，结果已保存至processed_results.csv")
    else:
        print("未找到可处理的文件")

def calculate_variance(data_list):
    """计算列表的方差，处理长度不足的情况"""
    if len(data_list) < 2:
        return 0.0
    return statistics.variance(data_list)

# 计算每个子模型在训练集上的结果
def gen_FUIAD_submodel_results():
    realiad = [
        'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser',
        'fire_hood', 'mint', 'mounts', 'pcb', 'phone_battery',
        'plastic_nut', 'plastic_plug', 'porcelain_doll', 'regulator', 'rolled_strip_base',
        'sim_card_set', 'switch', 'tape', 'terminalblock', 'toothbrush',
        'toy', 'toy_brick', 'transistor1', 'u_block', 'usb',
        'usb_adaptor', 'vcpill', 'wooden_beads', 'woodstick', 'zipper',
    ]
    sub_dir_num1 = [
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num1_iter1000_meanscores",
    ]

    sub_dir_num3 = [
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num3_iter1000_meanscores",
    ]

    sub_dir_num5 = [
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num5_iter1000_meanscores",
    ]

    sub_dir_num7 = [
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num7_iter1000_meanscores"
    ]

    sub_dir_all = [
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num7_iter1000_meanscores"
    ]
    sub_dir_all_f = [
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num7_iter1000_meanscores"
    ]

    sub_dir_all_n = [
        # "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num1_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num1_iter1000_meanscores",
        # "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num3_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num3_iter1000_meanscores",
        # "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num5_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num5_iter1000_meanscores",
        # "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f0_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f10_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num7_iter1000_meanscores",
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f40_t40_num7_iter1000_meanscores"
        ]
    sub_dir_test = [
        "DinomalyTrainer_configs_benchmark_dinomaly_dinomaly_f20_t40_num3_iter1000_meanscores",
    ]


    return_average = False

    ######## data for read
    result_root = '/home/user/zyx/HUST-ADer4YX_FUIAD/runs'
    results = []

    for sub_dir in sub_dir_all_n:
        # 读取 selected.csv 文件（假设文件名格式为 {sub_dir}_selected.csv）
        file_path = os.path.join(result_root, sub_dir)
        for filename in os.listdir(file_path):
            if filename.endswith("selected.csv"):
                selected_file_path = os.path.join(result_root, sub_dir, filename)
        selected_data = pd.read_csv(selected_file_path)

        for category in realiad:
            file_path = os.path.join(result_root, sub_dir, "stage1_results", category)
            # 存储当前类别的所有异常数（用于计算均值）
            category_anomaly_counts = []
            category_normal_counts = []
            for filename in os.listdir(file_path):
                file_path_csv = os.path.join(file_path, filename)
                data = pd.read_csv(file_path_csv)
                # 按anomaly_scores排序
                sorted_data = data.sort_values('anomaly_scores')
                # 计算40%的数据点数量
                forty_percent = int(0.4 * len(sorted_data))
                # 取最小的40%数据点
                bottom_40 = sorted_data.head(forty_percent)
                # 计算异常个数
                anomaly_count = bottom_40['anomalys'].sum()
                normal_count = forty_percent - anomaly_count
                category_anomaly_counts.append(anomaly_count)
                category_normal_counts.append(normal_count)
            # 计算平均值和方差
            avg_anomaly = sum(category_anomaly_counts) / len(category_anomaly_counts)
            avg_normal = sum(category_normal_counts) / len(category_normal_counts)
            var_anomaly = calculate_variance(category_anomaly_counts)
            var_normal = calculate_variance(category_normal_counts)

            # 从 selected.csv 中提取当前类别的 anomaly_num
            anomaly_num = None
            if selected_data is not None:
                matched_row = selected_data[selected_data['class_name'] == category]
                if not matched_row.empty:
                    anomaly_num = matched_row['anomaly_num'].iloc[0]            # 计算当前类别的平均异常数（如果至少有1个CSV文件）
            normal_num = forty_percent - anomaly_num

            results.append({
                'sub_dir': sub_dir,
                'category': category,
                'avg_anomaly_count': avg_anomaly,
                'avg_normal_count': avg_normal,
                'var_anomaly_count': var_anomaly,
                'var_normal_count': var_normal,
                'anomaly_num': anomaly_num,  # 来自 selected.csv 的对应值
                'normal_num': normal_num,
                'num_csv_files': len(category_anomaly_counts),  # 该类别有多少个CSV文件
            })

    # 转换为DataFrame并保存
    df_results = pd.DataFrame(results)
    output_file = "/home/user/zyx/HUST-ADer4YX_FUIAD/experiments/anomaly_counts_40_percent_all_n.csv"
    df_results.to_csv(output_file, index=False)

    print(f"结果已保存到 {output_file}")

# 得到计算平均值，先运行def gen_FUIAD_submodel_results()后，再运行这个函数
def gen_avg_from_FUIAD_submodel_results():

    # 读取原始数据文件
    input_file = "/home/user/zyx/HUST-ADer4YX_FUIAD/experiments/anomaly_counts_40_percent_all_n.csv"
    df = pd.read_csv(input_file)

    # 按 sub_dir 分组计算平均值
    columns_to_avg = [
        "avg_anomaly_count",
        "avg_normal_count",
        "var_anomaly_count",
        "var_normal_count",
        "anomaly_num",
        "normal_num",
    ]
    df_means = df.groupby("sub_dir")[columns_to_avg].mean().reset_index()

    # 重命名列名以明确是平均值
    df_means.columns = [
        "sub_dir",
        "mean_avg_anomaly",
        "mean_avg_normal",
        "mean_var_anomaly",
        "mean_var_normal",
        "mean_anomaly_num",
        "mean_normal_num",
    ]

    # 保存结果
    output_file = "/home/user/zyx/HUST-ADer4YX_FUIAD/experiments/anomaly_counts_40_percent_all_n_means.csv"
    df_means.to_csv(output_file, index=False)

    print(f"每个 sub_dir 的指标平均值已保存到 {output_file}")
    print("\n输出格式预览:")
    print(df_means.head())

# gen_chapter2_results()
# gen_chapter3_results()
# gen_chapter4_results()
# gen_real_case_results()
# gen_FUIAD_results()
# gen_FUIAD_submodel_results()
gen_avg_from_FUIAD_submodel_results()
