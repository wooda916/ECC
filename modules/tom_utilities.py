import numpy as np

def nano_to_days(full_dict, index_list, metric_shortlist):
    # convert timedeltas to 'days'
    for index in index_list:
        if index in metric_shortlist:
            print('full_dict\n', full_dict)
            for model in full_dict[index]:
                print(model)
                full_dict[index][model]['anomaly'] = (full_dict[index][model]['anomaly'] * 1/86400 * 1e-9).astype('float32')
                full_dict[index][model]['PDC']     = (full_dict[index][model]['PDC'] * 1/86400 * 1e-9).astype('float32')        
                full_dict[index][model]['EOC']     = (full_dict[index][model]['EOC'] * 1/86400 * 1e-9).astype('float32')
    return full_dict