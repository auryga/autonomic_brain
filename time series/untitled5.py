    binary_mask_brs = dataset['BRS']>0
    binary_mask_icp = dataset['ICP']>0
    
    binary_mask_all = binary_mask_brs*binary_mask_icp
    
    dataset_all=dataset[binary_mask_all]
    